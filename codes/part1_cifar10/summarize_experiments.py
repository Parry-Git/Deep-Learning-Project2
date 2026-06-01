"""
Print report-ready tables from Part 1 ablation summary JSON files.

Usage:
    python codes/part1_cifar10/summarize_experiments.py \
        --summary checkpoints/experiment_summary_ablation_e80.json
"""

import argparse
import json
from pathlib import Path


def format_config(config):
    fields = [
        ("channels", config.get("channels")),
        ("blocks", config.get("blocks_per_stage")),
        ("activation", config.get("activation")),
        ("loss", config.get("loss")),
        ("optimizer", config.get("optimizer")),
        ("lr", config.get("lr")),
        ("dropout", config.get("dropout")),
        ("weight_decay", config.get("weight_decay")),
        ("label_smoothing", config.get("label_smoothing")),
    ]
    return ", ".join(f"{key}={value}" for key, value in fields if value is not None)


def main():
    parser = argparse.ArgumentParser(description="Summarize Part 1 ablation experiments")
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    with open(args.summary) as f:
        results = json.load(f)

    print("| Group | Run | Best Acc (%) | Best Epoch | Final Acc (%) | Key Config |")
    print("|---|---:|---:|---:|---:|---|")
    for group, runs in results.items():
        for run_name, payload in runs.items():
            print(
                f"| {group} | {run_name} | "
                f"{payload['best_acc']:.2f} | {payload['best_epoch']} | "
                f"{payload['final_acc']:.2f} | {format_config(payload['config'])} |"
            )


if __name__ == "__main__":
    main()
