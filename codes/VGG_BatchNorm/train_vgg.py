"""
Train VGG-A and VGG-A+BatchNorm on CIFAR-10.

Examples:
    python codes/VGG_BatchNorm/train_vgg.py --model both --epochs 20
    python codes/VGG_BatchNorm/train_vgg.py --model vgg_bn --n-items 5000
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

VGG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VGG_DIR.parents[1]
sys.path.insert(0, str(VGG_DIR))

from cifar_loaders import get_cifar_loaders
from models.vgg import VGG_A, VGG_A_BatchNorm, get_number_of_parameters


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_random_seeds(seed_value=0, device=None):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if device is not None and device.type == "cuda":
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.benchmark = True


def make_scaler(device, enabled):
    if not enabled or device.type != "cuda":
        return None
    if hasattr(torch, "GradScaler"):
        return torch.GradScaler("cuda")
    return torch.cuda.amp.GradScaler()


def build_model(name):
    if name == "vgg_a":
        return VGG_A(), "VGG-A"
    if name == "vgg_bn":
        return VGG_A_BatchNorm(), "VGG-A+BN"
    raise ValueError(f"Unknown model: {name}")


def build_optimizer(args, model):
    if args.optimizer == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    if args.optimizer == "adamw":
        return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, use_amp=False):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(1).eq(targets).sum().item()
        total += inputs.size(0)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp=False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(1).eq(targets).sum().item()
        total += inputs.size(0)
    return total_loss / total, 100.0 * correct / total


def plot_histories(histories, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for name, history in histories.items():
        epochs = range(1, len(history["train_loss"]) + 1)
        axes[0].plot(epochs, history["train_loss"], label=f"{name} train", linewidth=1.8)
        axes[0].plot(epochs, history["test_loss"], label=f"{name} test", linewidth=1.8, linestyle="--")
        axes[1].plot(epochs, history["test_acc"], label=f"{name} test", linewidth=2.0)

    axes[0].set_title("VGG CIFAR-10 Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross Entropy")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].set_title("VGG CIFAR-10 Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_one_model(args, model_name, train_loader, val_loader, device):
    model, display_name = build_model(model_name)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(args, model)
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    use_amp = args.amp and device.type == "cuda"
    scaler = make_scaler(device, enabled=use_amp)
    ckpt_dir = PROJECT_ROOT / "checkpoints" / "vgg_bn"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"best_{args.run_name}_{model_name}.pth"
    log_path = ckpt_dir / f"train_log_{args.run_name}_{model_name}.json"

    history = {
        "model": model_name,
        "display_name": display_name,
        "params": get_number_of_parameters(model),
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "lr": [],
        "best_acc": -1.0,
        "best_epoch": 0,
        "config": vars(args),
    }

    print(f"\nModel: {display_name} | params={history['params']:,}")
    print(f"{'Epoch':>5} {'LR':>10} {'Train Loss':>11} {'Train Acc':>10} {'Test Loss':>10} {'Test Acc':>9} {'Time':>7}")
    print("-" * 75)

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler=scaler, use_amp=use_amp
        )
        test_loss, test_acc = evaluate(model, val_loader, criterion, device, use_amp=use_amp)
        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["lr"].append(lr)

        improved = test_acc > history["best_acc"]
        if improved:
            history["best_acc"] = test_acc
            history["best_epoch"] = epoch
            torch.save(model.state_dict(), best_path)

        save_json(history, log_path)
        mark = "*" if improved else " "
        print(
            f"{epoch:5d} {lr:10.6f} {train_loss:11.4f} {train_acc:9.2f}% "
            f"{test_loss:10.4f} {test_acc:8.2f}% {mark} {time.time() - start:5.1f}s"
        )

    return history


def parse_args():
    parser = argparse.ArgumentParser(description="Train VGG-A with/without BatchNorm on CIFAR-10")
    parser.add_argument("--model", choices=["vgg_a", "vgg_bn", "both"], default="both")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", choices=["adam", "adamw", "sgd"], default="adam")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="none")
    parser.add_argument("--n-items", type=int, default=None, help="Optional train subset size")
    parser.add_argument("--val-items", type=int, default=None, help="Optional test subset size")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--worker-timeout", type=int, default=0)
    parser.add_argument("--mp-context", choices=["fork", "spawn", "forkserver"], default=None)
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable AMP on CUDA; use --no-amp to disable")
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--run-name", type=str, default="vgg_compare")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()
    if args.workers > 0 and device.type == "cuda" and args.mp_context is None:
        args.mp_context = "forkserver"
    set_random_seeds(args.seed, device)
    print(f"Device: {device}")
    print(f"AMP: {'enabled' if args.amp and device.type == 'cuda' else 'disabled'}")

    train_loader, val_loader = get_cifar_loaders(
        batch_size=args.batch_size,
        num_workers=args.workers,
        n_items=args.n_items,
        val_items=args.val_items,
        augment=not args.no_augment,
        pin_memory=False if args.no_pin_memory else None,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        worker_timeout=args.worker_timeout,
        mp_context=args.mp_context,
        seed=args.seed,
    )

    model_names = ["vgg_a", "vgg_bn"] if args.model == "both" else [args.model]
    histories = {}
    for model_name in model_names:
        set_random_seeds(args.seed, device)
        history = run_one_model(args, model_name, train_loader, val_loader, device)
        histories[history["display_name"]] = history

    plot_path = PROJECT_ROOT / "pic" / f"vgg_train_{args.run_name}.png"
    plot_histories(histories, plot_path)
    print(f"\nSaved plot: {plot_path}")


if __name__ == "__main__":
    main()
