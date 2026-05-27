"""
Visualize random CIFAR-10 predictions from a trained checkpoint.

Usage:
    python codes/part1_cifar10/visualize_predictions.py
    python codes/part1_cifar10/visualize_predictions.py --num-samples 24 --output pic/predictions.png
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
import torchvision
import torchvision.transforms as transforms

ssl._create_default_https_context = ssl._create_unverified_context

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "codes"))

from part1_cifar10.model import build_model


CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)
CAT_CLASS = CLASSES.index("cat")
DOG_CLASS = CLASSES.index("dog")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def denormalize(img):
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    img = img.cpu() * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def load_model(args, device):
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
    return model, desc


@torch.no_grad()
def predict_samples(model, dataset, indices, device, batch_size):
    images = []
    targets = []
    for idx in indices:
        image, target = dataset[idx]
        images.append(image)
        targets.append(target)

    inputs = torch.stack(images).to(device)
    logits_chunks = []
    for start in range(0, len(inputs), batch_size):
        logits_chunks.append(model(inputs[start:start + batch_size]))
    logits = torch.cat(logits_chunks, dim=0)

    probs = torch.softmax(logits, dim=1)
    confs, preds = probs.max(dim=1)
    return images, targets, preds.cpu().tolist(), confs.cpu().tolist()


def plot_predictions(images, targets, preds, confs, output, cols):
    n = len(images)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.45, rows * 2.85))
    axes = [axes] if n == 1 else axes.flatten()

    correct = 0
    for ax_idx, ax in enumerate(axes):
        ax.axis("off")
        if ax_idx >= n:
            continue

        target = targets[ax_idx]
        pred = preds[ax_idx]
        conf = confs[ax_idx]
        correct += int(target == pred)

        ax.imshow(denormalize(images[ax_idx]))
        title_color = "#15803d" if target == pred else "#b91c1c"
        label = f"Pred: {CLASSES[pred]}  {conf * 100:.1f}%\nTrue: {CLASSES[target]}"
        ax.text(
            0.5,
            -0.13,
            label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            color=title_color,
            fontsize=9,
            linespacing=1.35,
        )

    acc = 100.0 * correct / n
    fig.suptitle(f"Random CIFAR-10 Predictions | Sample Accuracy: {correct}/{n} ({acc:.1f}%)", fontsize=14)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.91, bottom=0.08, wspace=0.18, hspace=0.58)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    return correct, n, acc


def sample_indices(dataset, args):
    generator = torch.Generator()
    if args.seed is not None:
        generator.manual_seed(args.seed)

    if args.sampling == "random":
        n = min(args.num_samples, len(dataset))
        return torch.randperm(len(dataset), generator=generator)[:n].tolist()

    labels = dataset.targets
    selected = []
    for class_idx in range(len(CLASSES)):
        candidates = [idx for idx, label in enumerate(labels) if label == class_idx]
        count = 2 if class_idx in (CAT_CLASS, DOG_CLASS) else 1
        perm = torch.randperm(len(candidates), generator=generator)[:count].tolist()
        selected.extend(candidates[i] for i in perm)

    order = torch.randperm(len(selected), generator=generator).tolist()
    return [selected[i] for i in order]


def main():
    parser = argparse.ArgumentParser(description="Visualize random CIFAR-10 predictions")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "part1_best.pth")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "pic" / "predictions.png")
    parser.add_argument("--num-samples", type=int, default=12)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--sampling", choices=["balanced12", "random"], default="balanced12",
                        help="balanced12 samples 2 cats, 2 dogs, and 1 image from each other class")
    parser.add_argument("--channels", type=int, nargs=3, default=[64, 128, 256])
    parser.add_argument("--blocks", type=int, nargs=3, default=[3, 3, 3])
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    if args.num_samples <= 0:
        print("--num-samples must be positive")
        sys.exit(1)
    if args.cols <= 0:
        print("--cols must be positive")
        sys.exit(1)

    device = get_device()
    print(f"Device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root=str(PROJECT_ROOT / "data"),
        train=(args.split == "train"),
        download=False,
        transform=transform,
    )

    indices = sample_indices(dataset, args)

    model, desc = load_model(args, device)
    print(f"Model: {desc}")
    print(f"Loaded: {args.checkpoint}")

    images, targets, preds, confs = predict_samples(model, dataset, indices, device, args.batch_size)
    correct, total, acc = plot_predictions(images, targets, preds, confs, args.output, args.cols)
    print(f"Saved: {args.output}")
    print(f"Sample accuracy: {correct}/{total} ({acc:.1f}%)")


if __name__ == "__main__":
    main()
