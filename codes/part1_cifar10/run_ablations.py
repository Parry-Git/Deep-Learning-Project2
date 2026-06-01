"""
Run Part 1 ablation experiments sequentially and validate their summaries.

This script is intentionally a thin orchestration layer around train.py:
it keeps every training run reproducible from the command line, writes one log
file per ablation group, skips already completed groups by default, and emits
Markdown tables that can be copied into the report.

Usage:
    python codes/part1_cifar10/run_ablations.py --epochs 80 --workers 2
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = PROJECT_ROOT / "codes" / "part1_cifar10" / "train.py"
SUMMARY_SCRIPT = PROJECT_ROOT / "codes" / "part1_cifar10" / "summarize_experiments.py"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "outputs" / "part1_ablations"
DATA_DIR = PROJECT_ROOT / "data" / "cifar-10-batches-py"

EXPECTED_RUNS = {
    "width": {"width_small", "width_medium", "width_large"},
    "activations": {"act_relu", "act_gelu", "act_silu"},
    "optimizers": {"opt_sgd", "opt_adam", "opt_adamw"},
    "losses": {"loss_ce", "loss_ls005", "loss_ls010", "loss_focal"},
    "regularization": {"reg_dropout000", "reg_dropout010", "reg_dropout020", "reg_wd0001"},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run Part 1 ablation experiments")
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=list(EXPECTED_RUNS),
        default=list(EXPECTED_RUNS),
        help="Ablation groups to run in order",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--mp-context", choices=["fork", "spawn", "forkserver"], default="forkserver")
    parser.add_argument("--worker-timeout", type=int, default=120)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable AMP for every ablation run; use --no-amp to disable")
    parser.add_argument("--compile", action="store_true", help="Allow torch.compile in train.py")
    parser.add_argument("--save-ablation-checkpoints", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rerun groups even if valid summaries exist")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with later groups when one group fails",
    )
    return parser.parse_args()


def summary_path(group, epochs):
    return CHECKPOINT_DIR / f"experiment_summary_ablation_{group}_e{epochs}.json"


def markdown_path(group, epochs):
    return LOG_DIR / f"summary_ablation_{group}_e{epochs}.md"


def run_name(group, epochs):
    return f"ablation_{group}_e{epochs}"


def validate_summary(path, group, epochs):
    if not path.exists():
        raise FileNotFoundError(f"Missing summary: {path}")
    with open(path) as f:
        payload = json.load(f)

    if group not in payload:
        raise ValueError(f"{path} does not contain group {group!r}")

    expected = {f"ablation_{group}_e{epochs}_{name}" for name in EXPECTED_RUNS[group]}
    actual = set(payload[group])
    missing = expected - actual
    if missing:
        raise ValueError(f"{path} is missing runs: {sorted(missing)}")

    for run, result in payload[group].items():
        for key in ("best_acc", "best_epoch", "final_acc", "final_loss", "config", "artifacts"):
            if key not in result:
                raise ValueError(f"{path} -> {run} missing key {key!r}")
        if not (0.0 <= float(result["best_acc"]) <= 100.0):
            raise ValueError(f"{path} -> {run} has invalid best_acc={result['best_acc']}")
        if int(result["best_epoch"]) < 1:
            raise ValueError(f"{path} -> {run} has invalid best_epoch={result['best_epoch']}")

    return payload


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


def write_group_markdown(group, epochs):
    out_path = markdown_path(group, epochs)
    command = [
        sys.executable,
        str(SUMMARY_SCRIPT),
        "--summary",
        str(summary_path(group, epochs)),
    ]
    result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result.stdout)
    return out_path


def write_combined_markdown(groups, epochs):
    combined = LOG_DIR / f"summary_ablation_all_e{epochs}.md"
    parts = []
    for group in groups:
        path = markdown_path(group, epochs)
        if path.exists():
            parts.append(f"## {group}\n\n{path.read_text().strip()}\n")
    combined.write_text("\n\n".join(parts) + "\n")
    return combined


def build_train_command(args, group):
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--experiment",
        group,
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--run-name",
        run_name(group, args.epochs),
        "--workers",
        str(args.workers),
        "--prefetch-factor",
        str(args.prefetch_factor),
        "--worker-timeout",
        str(args.worker_timeout),
    ]
    if args.workers > 0 and args.persistent_workers:
        command.append("--persistent-workers")
    if args.workers > 0 and args.mp_context:
        command.extend(["--mp-context", args.mp_context])
    if args.amp:
        command.append("--amp")
    else:
        command.append("--no-amp")
    if not args.compile:
        command.append("--no-compile")
    if args.save_ablation_checkpoints:
        command.append("--save-ablation-checkpoints")
    return command


def main():
    args = parse_args()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_DIR.exists():
        raise SystemExit(
            f"CIFAR-10 data not found at {DATA_DIR}. "
            "Run: python codes/download.py --data --source torchvision"
        )

    planned = {group: sorted(EXPECTED_RUNS[group]) for group in args.groups}
    total_runs = sum(len(runs) for runs in planned.values())
    print("Ablation plan:")
    for group, runs in planned.items():
        print(f"  - {group}: {len(runs)} runs -> {', '.join(runs)}")
    print(f"Total planned runs: {total_runs}")

    completed = []
    failed = []
    started_at = time.time()

    for group in args.groups:
        s_path = summary_path(group, args.epochs)
        if not args.force and s_path.exists():
            try:
                validate_summary(s_path, group, args.epochs)
                md_path = write_group_markdown(group, args.epochs)
                print(f"[skip] {group}: valid summary exists -> {s_path}")
                print(f"[skip] {group}: markdown refreshed -> {md_path}")
                completed.append(group)
                continue
            except Exception as exc:
                print(f"[rerun] {group}: existing summary is invalid: {exc}")

        command = build_train_command(args, group)
        log_path = LOG_DIR / f"train_{run_name(group, args.epochs)}.log"
        print("\n" + "=" * 80)
        print(f"[run] group={group}")
        print("[cmd] " + " ".join(command))
        print(f"[log] {log_path}")
        print("=" * 80)

        code = run_command(command, log_path)
        if code != 0:
            failed.append((group, f"train.py exited with code {code}; see {log_path}"))
            if not args.continue_on_error:
                break
            continue

        try:
            validate_summary(s_path, group, args.epochs)
            md_path = write_group_markdown(group, args.epochs)
            print(f"[ok] {group}: summary validated -> {s_path}")
            print(f"[ok] {group}: markdown written -> {md_path}")
            completed.append(group)
        except Exception as exc:
            failed.append((group, str(exc)))
            if not args.continue_on_error:
                break

    combined = write_combined_markdown(completed, args.epochs)
    elapsed = time.time() - started_at

    print("\n" + "=" * 80)
    print(f"Completed groups: {completed}")
    print(f"Combined markdown: {combined}")
    print(f"Elapsed: {elapsed / 60:.1f} min")
    if failed:
        print("Failed groups:")
        for group, reason in failed:
            print(f"  - {group}: {reason}")
        raise SystemExit(1)
    print("All requested ablation groups completed and validated.")


if __name__ == "__main__":
    main()
