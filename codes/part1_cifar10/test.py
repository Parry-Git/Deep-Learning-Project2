"""
Evaluate a trained model on CIFAR-10 test set.

Usage:
    python test.py                                    # use default checkpoint
    python test.py --checkpoint path/to/model.pth     # specify checkpoint

Cross-platform: works on CUDA, MPS, and CPU.
"""

import argparse
import ssl
import sys
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms

ssl._create_default_https_context = ssl._create_unverified_context

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "codes"))

from part1_cifar10.model import build_model


CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_testloader(batch_size=128, num_workers=0):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    testset = torchvision.datasets.CIFAR10(
        root=str(PROJECT_ROOT / "data"), train=False, download=False, transform=transform
    )
    return torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    class_correct = [0] * 10
    class_total = [0] * 10

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        preds = outputs.argmax(1)
        correct += preds.eq(targets).sum().item()
        total += inputs.size(0)

        for t, p in zip(targets, preds):
            class_total[t.item()] += 1
            if t == p:
                class_correct[t.item()] += 1

    overall_acc = 100.0 * correct / total
    return overall_acc, class_correct, class_total


def main():
    parser = argparse.ArgumentParser(description="Test CIFAR-10 model")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--channels", type=int, nargs=3, default=[64, 128, 256])
    parser.add_argument("--blocks", type=int, nargs=3, default=[3, 3, 3])
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    ckpt_path = args.checkpoint or str(PROJECT_ROOT / "checkpoints" / "part1_best.pth")
    if not Path(ckpt_path).exists():
        print(f"Checkpoint not found: {ckpt_path}")
        print("Please train the model first: python train.py")
        sys.exit(1)

    model_config = {
        "blocks_per_stage": tuple(args.blocks),
        "channels": tuple(args.channels),
        "activation": args.activation,
        "dropout": args.dropout,
    }
    model, desc = build_model(model_config)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    print(f"Model: {desc}")
    print(f"Loaded: {ckpt_path}")

    loader = get_testloader()
    overall_acc, class_correct, class_total = evaluate(model, loader, device)

    print(f"\nOverall Test Accuracy: {overall_acc:.2f}%")
    print(f"\nPer-class accuracy:")
    print(f"{'Class':<12} {'Correct':>7} {'Total':>7} {'Accuracy':>8}")
    print("-" * 38)
    for i in range(10):
        acc = 100.0 * class_correct[i] / class_total[i]
        print(f"{CLASSES[i]:<12} {class_correct[i]:>7} {class_total[i]:>7} {acc:>7.2f}%")


if __name__ == "__main__":
    main()
