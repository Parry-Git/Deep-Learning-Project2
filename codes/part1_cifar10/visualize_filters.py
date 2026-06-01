"""
Visualize first-layer convolution filters from the Part 1 CIFAR-10 model.

Usage:
    python codes/part1_cifar10/visualize_filters.py \
        --checkpoint checkpoints/part1_best.pth --output pic/part1_conv1_filters.png
"""

import argparse
import math
import ssl
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import torch

ssl._create_default_https_context = ssl._create_unverified_context

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "codes"))

from part1_cifar10.model import build_model


def normalize_filter(weight):
    weight = weight.detach().cpu()
    min_value = weight.min()
    max_value = weight.max()
    return (weight - min_value) / (max_value - min_value + 1e-8)


def main():
    parser = argparse.ArgumentParser(description="Visualize first-layer Conv2d filters")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "part1_best.pth")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "pic" / "part1_conv1_filters.png")
    parser.add_argument("--num-filters", type=int, default=64)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--channels", type=int, nargs=3, default=[64, 128, 256])
    parser.add_argument("--blocks", type=int, nargs=3, default=[3, 3, 3])
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    model_config = {
        "blocks_per_stage": tuple(args.blocks),
        "channels": tuple(args.channels),
        "activation": args.activation,
        "dropout": args.dropout,
    }
    model, desc = build_model(model_config)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)

    weights = model.conv1.weight[: args.num_filters]
    rows = math.ceil(len(weights) / args.cols)
    fig, axes = plt.subplots(rows, args.cols, figsize=(args.cols * 1.25, rows * 1.25))
    axes = [axes] if len(weights) == 1 else axes.flatten()

    for idx, ax in enumerate(axes):
        ax.axis("off")
        if idx >= len(weights):
            continue
        image = normalize_filter(weights[idx]).permute(1, 2, 0).numpy()
        ax.imshow(image)

    fig.suptitle(f"First-layer Conv Filters | {desc}", fontsize=11)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
