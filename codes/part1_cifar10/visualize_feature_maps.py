"""
Visualize intermediate feature maps at each stage of CifarResNet.

Shows how representations evolve from low-level edges (stage 1) to
high-level semantic features (stage 3).

Usage:
    python codes/part1_cifar10/visualize_feature_maps.py
"""

import argparse
import ssl
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

ssl._create_default_https_context = ssl._create_unverified_context

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "codes"))

from part1_cifar10.model import build_model

CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def denormalize(img_tensor):
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    img = img_tensor.cpu() * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def extract_features(model, x):
    """Run forward pass and capture feature maps at each stage boundary."""
    features = {}
    x = model.act(model.bn1(model.conv1(x)))
    features["After stem (64ch, 32x32)"] = x.detach()

    blocks = list(model.stages.children())
    blocks_per_stage = len(blocks) // 3

    for stage_idx in range(3):
        start = stage_idx * blocks_per_stage
        end = start + blocks_per_stage
        for block in blocks[start:end]:
            x = block(x)
        ch = x.shape[1]
        sp = x.shape[2]
        features[f"After stage {stage_idx + 1} ({ch}ch, {sp}x{sp})"] = x.detach()

    return features


def plot_feature_maps(image, target, pred, features, output, num_channels=16):
    n_stages = len(features)
    fig, axes = plt.subplots(n_stages + 1, num_channels + 1, figsize=(num_channels * 1.0 + 1.5, (n_stages + 1) * 1.1))

    for ax in axes.flatten():
        ax.axis("off")

    axes[0, 0].imshow(denormalize(image))
    axes[0, 0].set_title("Input", fontsize=8)

    for stage_idx, (stage_name, fmap) in enumerate(features.items(), start=1):
        fmap = fmap[0].cpu()
        axes[stage_idx, 0].text(
            0.5, 0.5, stage_name, transform=axes[stage_idx, 0].transAxes,
            ha="center", va="center", fontsize=7, wrap=True,
        )
        for ch_idx in range(min(num_channels, fmap.shape[0])):
            ax = axes[stage_idx, ch_idx + 1]
            channel = fmap[ch_idx].numpy()
            ax.imshow(channel, cmap="viridis")

    color = "#15803d" if target == pred else "#b91c1c"
    fig.suptitle(
        f"Feature Map Evolution | True: {CLASSES[target]}, Pred: {CLASSES[pred]}",
        fontsize=12, color=color,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize feature maps")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "part1_best.pth")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "pic" / "part1_feature_maps.png")
    parser.add_argument("--num-channels", type=int, default=16)
    parser.add_argument("--image-index", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--channels", type=int, nargs=3, default=[64, 128, 256])
    parser.add_argument("--blocks", type=int, nargs=3, default=[3, 3, 3])
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    model_config = {
        "blocks_per_stage": tuple(args.blocks),
        "channels": tuple(args.channels),
        "activation": args.activation,
        "dropout": args.dropout,
    }
    model, desc = build_model(model_config)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    print(f"Model: {desc}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root=str(PROJECT_ROOT / "data"), train=False, download=False, transform=transform,
    )

    generator = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(len(dataset), generator=generator)[args.image_index].item()
    image, target = dataset[idx]
    input_tensor = image.unsqueeze(0).to(device)

    with torch.no_grad():
        features = extract_features(model, input_tensor)
        logits = model(input_tensor)
        pred = logits.argmax(1).item()

    plot_feature_maps(image, target, pred, features, args.output, args.num_channels)
    print(f"Image: index={idx}, true={CLASSES[target]}, pred={CLASSES[pred]}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
