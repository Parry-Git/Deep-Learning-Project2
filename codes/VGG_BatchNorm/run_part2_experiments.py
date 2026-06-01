"""
Run Part 2 BatchNorm experiments sequentially and validate outputs.

The runner covers the two requirements from the project handout:
1. VGG-A vs VGG-A+BN training curves.
2. Loss landscape over multiple learning rates.

It also adds a learning-rate sensitivity sweep, which is useful for explaining
BN as an optimization stabilizer in the report.

Usage:
    python codes/VGG_BatchNorm/run_part2_experiments.py
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VGG_DIR = PROJECT_ROOT / "codes" / "VGG_BatchNorm"
TRAIN_SCRIPT = VGG_DIR / "train_vgg.py"
LANDSCAPE_SCRIPT = VGG_DIR / "VGG_Loss_Landscape.py"
SUMMARY_SCRIPT = VGG_DIR / "summarize_vgg_experiments.py"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "vgg_bn"
LOG_DIR = PROJECT_ROOT / "outputs" / "part2_bn"
DATA_DIR = PROJECT_ROOT / "data" / "cifar-10-batches-py"


def parse_args():
    parser = argparse.ArgumentParser(description="Run Part 2 VGG BatchNorm experiments")
    parser.add_argument("--compare-epochs", type=int, default=100)
    parser.add_argument("--sweep-epochs", type=int, default=40)
    parser.add_argument("--landscape-epochs", type=int, default=5)
    parser.add_argument("--landscape-n-items", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--mp-context", choices=["fork", "spawn", "forkserver"], default="forkserver")
    parser.add_argument("--worker-timeout", type=int, default=120)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--optimizer", choices=["adam", "adamw", "sgd"], default="adam")
    parser.add_argument("--lrs", type=float, nargs="+", default=[1e-4, 5e-4, 1e-3, 2e-3])
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable AMP for every Part 2 run; use --no-amp to disable")
    parser.add_argument("--force", action="store_true", help="Rerun existing valid outputs")
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument("--skip-lr-sweep", action="store_true")
    parser.add_argument("--skip-landscape", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


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


def train_log_path(run_name, model_name):
    return CHECKPOINT_DIR / f"train_log_{run_name}_{model_name}.json"


def landscape_log_path(run_name):
    return CHECKPOINT_DIR / f"loss_landscape_{run_name}.json"


def validate_train_log(path, model_name):
    if not path.exists():
        raise FileNotFoundError(f"Missing train log: {path}")
    with open(path) as f:
        payload = json.load(f)
    for key in ("model", "display_name", "params", "train_loss", "test_loss", "test_acc", "best_acc", "best_epoch"):
        if key not in payload:
            raise ValueError(f"{path} missing key {key!r}")
    if payload["model"] != model_name:
        raise ValueError(f"{path} model mismatch: expected {model_name}, got {payload['model']}")
    if not payload["test_acc"]:
        raise ValueError(f"{path} has empty test_acc")
    if not (0.0 <= float(payload["best_acc"]) <= 100.0):
        raise ValueError(f"{path} has invalid best_acc={payload['best_acc']}")
    if int(payload["best_epoch"]) < 1:
        raise ValueError(f"{path} has invalid best_epoch={payload['best_epoch']}")
    return payload


def validate_pair(run_name):
    logs = []
    for model_name in ("vgg_a", "vgg_bn"):
        path = train_log_path(run_name, model_name)
        validate_train_log(path, model_name)
        logs.append(path)
    return logs


def validate_landscape(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing landscape log: {path}")
    with open(path) as f:
        payload = json.load(f)
    if set(payload.get("models", {})) != {"vgg_a", "vgg_bn"}:
        raise ValueError(f"{path} must contain vgg_a and vgg_bn")
    for model_name, model_payload in payload["models"].items():
        curves = model_payload.get("curves", [])
        band = model_payload.get("band", {})
        if not curves:
            raise ValueError(f"{path} -> {model_name} has no curves")
        for key in ("steps", "min", "max", "mean"):
            if key not in band or not band[key]:
                raise ValueError(f"{path} -> {model_name} missing band {key!r}")
        lengths = {len(band[key]) for key in ("steps", "min", "max", "mean")}
        if len(lengths) != 1:
            raise ValueError(f"{path} -> {model_name} band arrays have mismatched lengths")
    return payload


def base_train_command(args, run_name, epochs, lr):
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--model",
        "both",
        "--epochs",
        str(epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(lr),
        "--optimizer",
        args.optimizer,
        "--run-name",
        run_name,
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
    return command


def landscape_command(args):
    run_name = "vgg_loss_landscape"
    command = [
        sys.executable,
        str(LANDSCAPE_SCRIPT),
        "--epochs",
        str(args.landscape_epochs),
        "--n-items",
        str(args.landscape_n_items),
        "--batch-size",
        str(args.batch_size),
        "--optimizer",
        args.optimizer,
        "--run-name",
        run_name,
        "--workers",
        str(args.workers),
        "--prefetch-factor",
        str(args.prefetch_factor),
        "--worker-timeout",
        str(args.worker_timeout),
        "--lrs",
        *[str(lr) for lr in args.lrs],
    ]
    if args.workers > 0 and args.persistent_workers:
        command.append("--persistent-workers")
    if args.workers > 0 and args.mp_context:
        command.extend(["--mp-context", args.mp_context])
    if args.amp:
        command.append("--amp")
    else:
        command.append("--no-amp")
    return run_name, command


def write_markdown(logs, landscape_log=None):
    output = LOG_DIR / "part2_summary.md"
    command = [
        sys.executable,
        str(SUMMARY_SCRIPT),
        "--logs",
        *[str(path) for path in logs],
    ]
    if landscape_log is not None:
        command.extend(["--landscape", str(landscape_log)])
    result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.stdout)
    return output


def run_train_task(args, run_name, epochs, lr):
    logs = [train_log_path(run_name, "vgg_a"), train_log_path(run_name, "vgg_bn")]
    if not args.force and all(path.exists() for path in logs):
        validate_pair(run_name)
        print(f"[skip] {run_name}: valid VGG logs exist")
        return logs

    command = base_train_command(args, run_name, epochs, lr)
    log_path = LOG_DIR / f"train_{run_name}.log"
    print("\n" + "=" * 80)
    print(f"[run] {run_name}")
    print("[cmd] " + " ".join(command))
    print(f"[log] {log_path}")
    print("=" * 80)
    code = run_command(command, log_path)
    if code != 0:
        raise RuntimeError(f"{run_name} failed with exit code {code}; see {log_path}")
    return validate_pair(run_name)


def main():
    args = parse_args()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_DIR.exists():
        raise SystemExit(
            f"CIFAR-10 data not found at {DATA_DIR}. "
            "Run: python codes/download.py --data --source torchvision"
        )

    all_logs = []
    landscape_log = None
    failed = []
    started_at = time.time()

    tasks = []
    if not args.skip_compare:
        tasks.append(("compare", f"vgg_compare_e{args.compare_epochs}", args.compare_epochs, 1e-3))
    if not args.skip_lr_sweep:
        for lr in args.lrs:
            lr_tag = f"{lr:g}".replace(".", "p").replace("-", "m")
            tasks.append(("lr_sweep", f"vgg_lr_{lr_tag}_e{args.sweep_epochs}", args.sweep_epochs, lr))

    for _, run_name, epochs, lr in tasks:
        try:
            all_logs.extend(run_train_task(args, run_name, epochs, lr))
        except Exception as exc:
            failed.append((run_name, str(exc)))
            if not args.continue_on_error:
                break

    if not failed or args.continue_on_error:
        if not args.skip_landscape:
            try:
                run_name, command = landscape_command(args)
                landscape_log = landscape_log_path(run_name)
                if not args.force and landscape_log.exists():
                    validate_landscape(landscape_log)
                    print(f"[skip] {run_name}: valid landscape log exists")
                else:
                    log_path = LOG_DIR / f"train_{run_name}.log"
                    print("\n" + "=" * 80)
                    print(f"[run] {run_name}")
                    print("[cmd] " + " ".join(command))
                    print(f"[log] {log_path}")
                    print("=" * 80)
                    code = run_command(command, log_path)
                    if code != 0:
                        raise RuntimeError(f"{run_name} failed with exit code {code}; see {log_path}")
                    validate_landscape(landscape_log)
            except Exception as exc:
                failed.append(("vgg_loss_landscape", str(exc)))

    if all_logs or landscape_log:
        markdown = write_markdown(all_logs, landscape_log if landscape_log and landscape_log.exists() else None)
        print(f"[ok] report table written -> {markdown}")

    elapsed = time.time() - started_at
    print("\n" + "=" * 80)
    print(f"Validated train logs: {len(all_logs)}")
    print(f"Elapsed: {elapsed / 60:.1f} min")
    if failed:
        print("Failed tasks:")
        for name, reason in failed:
            print(f"  - {name}: {reason}")
        raise SystemExit(1)
    print("All requested Part 2 experiments completed and validated.")


if __name__ == "__main__":
    main()
