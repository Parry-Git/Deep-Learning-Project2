"""
Print report-ready tables for VGG-A / VGG-A+BN experiments.

Usage:
    python codes/VGG_BatchNorm/summarize_vgg_experiments.py \
        --logs checkpoints/vgg_bn/train_log_vgg_compare_e100_vgg_a.json \
               checkpoints/vgg_bn/train_log_vgg_compare_e100_vgg_bn.json
"""

import argparse
import json
from pathlib import Path


def load_json(path):
    with open(path) as f:
        return json.load(f)


def format_config(config):
    fields = [
        ("epochs", config.get("epochs")),
        ("lr", config.get("lr")),
        ("optimizer", config.get("optimizer")),
        ("scheduler", config.get("scheduler")),
        ("batch_size", config.get("batch_size")),
        ("augment", not config.get("no_augment", False)),
        ("n_items", config.get("n_items")),
        ("seed", config.get("seed")),
    ]
    return ", ".join(f"{key}={value}" for key, value in fields if value is not None)


def print_train_table(logs):
    print("| Run | Model | Params | Best Acc (%) | Best Epoch | Final Acc (%) | Final Loss | Key Config |")
    print("|---|---|---:|---:|---:|---:|---:|---|")
    for path in logs:
        payload = load_json(path)
        run_name = payload["config"].get("run_name", Path(path).stem)
        final_acc = payload["test_acc"][-1]
        final_loss = payload["test_loss"][-1]
        print(
            f"| {run_name} | {payload['display_name']} | {payload['params']} | "
            f"{payload['best_acc']:.2f} | {payload['best_epoch']} | "
            f"{final_acc:.2f} | {final_loss:.4f} | {format_config(payload['config'])} |"
        )


def print_landscape_table(path):
    payload = load_json(path)
    print("| Model | LR Values | Steps | Mean Band Width | Max Band Width | Final Mean Loss |")
    print("|---|---|---:|---:|---:|---:|")
    for model_name, model_payload in payload["models"].items():
        curves = model_payload["curves"]
        lrs = ", ".join(f"{curve['lr']:g}" for curve in curves)
        band = model_payload["band"]
        widths = [hi - lo for lo, hi in zip(band["min"], band["max"])]
        mean_width = sum(widths) / len(widths)
        max_width = max(widths)
        final_mean = band["mean"][-1]
        print(
            f"| {model_payload['display_name']} | {lrs} | {len(band['steps'])} | "
            f"{mean_width:.4f} | {max_width:.4f} | {final_mean:.4f} |"
        )


def main():
    parser = argparse.ArgumentParser(description="Summarize VGG BatchNorm experiments")
    parser.add_argument("--logs", type=Path, nargs="*", default=[])
    parser.add_argument("--landscape", type=Path, default=None)
    args = parser.parse_args()

    if args.logs:
        print_train_table(args.logs)
    if args.landscape:
        if args.logs:
            print()
        print_landscape_table(args.landscape)


if __name__ == "__main__":
    main()
