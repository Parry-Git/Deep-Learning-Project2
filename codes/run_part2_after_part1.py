"""
Wait for Part 1 ablation summaries, then launch Part 2 experiments.

This is useful when Part 1 is already running in another terminal. Start this
script in a second terminal; it polls for valid Part 1 summary JSON files and
automatically starts codes/VGG_BatchNorm/run_part2_experiments.py afterward.

Usage:
    python codes/run_part2_after_part1.py --part1-epochs 100 --part2-workers 4
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PART2_RUNNER = PROJECT_ROOT / "codes" / "VGG_BatchNorm" / "run_part2_experiments.py"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "full_pipeline"
PART1_GROUPS = ["width", "activations", "optimizers", "losses", "regularization"]


def parse_args():
    parser = argparse.ArgumentParser(description="Wait for Part 1, then run Part 2")
    parser.add_argument("--part1-epochs", type=int, default=100)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-hours", type=float, default=18.0)
    parser.add_argument("--part2-workers", type=int, default=4)
    parser.add_argument("--part2-compare-epochs", type=int, default=100)
    parser.add_argument("--part2-sweep-epochs", type=int, default=40)
    parser.add_argument("--part2-landscape-epochs", type=int, default=5)
    parser.add_argument("--part2-landscape-n-items", type=int, default=5000)
    parser.add_argument("--part2-batch-size", type=int, default=128)
    parser.add_argument("--part2-mp-context", choices=["fork", "spawn", "forkserver"], default="forkserver")
    parser.add_argument("--part2-worker-timeout", type=int, default=120)
    parser.add_argument("--part2-prefetch-factor", type=int, default=2)
    parser.add_argument("--part2-no-amp", action="store_true")
    parser.add_argument("--part2-force", action="store_true")
    return parser.parse_args()


def summary_path(group, epochs):
    return PROJECT_ROOT / "checkpoints" / f"experiment_summary_ablation_{group}_e{epochs}.json"


def validate_part1_summary(group, epochs):
    command = [
        sys.executable,
        str(PROJECT_ROOT / "codes" / "part1_cifar10" / "summarize_experiments.py"),
        "--summary",
        str(summary_path(group, epochs)),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)


def missing_or_invalid_groups(epochs):
    pending = []
    for group in PART1_GROUPS:
        path = summary_path(group, epochs)
        if not path.exists():
            pending.append(group)
            continue
        try:
            validate_part1_summary(group, epochs)
        except subprocess.CalledProcessError:
            pending.append(group)
    return pending


def build_part2_command(args):
    command = [
        sys.executable,
        str(PART2_RUNNER),
        "--compare-epochs",
        str(args.part2_compare_epochs),
        "--sweep-epochs",
        str(args.part2_sweep_epochs),
        "--landscape-epochs",
        str(args.part2_landscape_epochs),
        "--landscape-n-items",
        str(args.part2_landscape_n_items),
        "--batch-size",
        str(args.part2_batch_size),
        "--workers",
        str(args.part2_workers),
        "--mp-context",
        args.part2_mp_context,
        "--worker-timeout",
        str(args.part2_worker_timeout),
        "--prefetch-factor",
        str(args.part2_prefetch_factor),
    ]
    if args.part2_no_amp:
        command.append("--no-amp")
    if args.part2_force:
        command.append("--force")
    return command


def run_command(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return process.wait()


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    deadline = time.time() + args.timeout_hours * 3600
    while True:
        pending = missing_or_invalid_groups(args.part1_epochs)
        if not pending:
            print("Part 1 summaries are complete and readable.")
            break
        if time.time() >= deadline:
            raise SystemExit(f"Timed out waiting for Part 1 summaries. Still pending: {pending}")
        print(f"Waiting for Part 1 summaries: {', '.join(pending)}")
        time.sleep(args.poll_seconds)

    command = build_part2_command(args)
    log_path = OUTPUT_DIR / "part2_after_part1.log"
    print("[cmd] " + " ".join(command))
    print(f"[log] {log_path}")
    code = run_command(command, log_path)
    if code != 0:
        raise SystemExit(f"Part 2 runner failed with exit code {code}; see {log_path}")
    print("Part 2 experiments completed.")


if __name__ == "__main__":
    main()
