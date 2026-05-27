"""
CIFAR-10 training script.

Usage:
    python train.py                          # default config
    python train.py --epochs 200 --lr 0.1    # custom hyperparams
    python train.py --amp                    # mixed precision (CUDA/MPS)
    python train.py --experiment all         # run ablation experiments

Features:
    - Data augmentation: RandomCrop, HorizontalFlip, Cutout
    - Label smoothing
    - CosineAnnealing LR schedule with warmup
    - Multiple optimizer choices (SGD, Adam, AdamW)
    - Automatic best-model checkpointing
    - Mixed precision (AMP) support for CUDA and MPS
    - Cross-platform: CUDA, MPS, CPU
"""

import argparse
import json
import os
import ssl
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

ssl._create_default_https_context = ssl._create_unverified_context

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "codes"))

from part1_cifar10.model import build_model


# ======================== Platform Helpers ========================


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def supports_amp(device):
    """Check if the device supports torch.autocast."""
    if device.type == "cuda":
        return True
    if device.type == "mps":
        return hasattr(torch, "autocast") and torch.__version__ >= "2.0"
    return False


def supports_channels_last(device):
    if device.type == "cuda":
        return True
    if device.type == "mps":
        return torch.__version__ >= "2.1"
    return True


def make_scaler(device):
    """Create GradScaler compatible with current PyTorch version."""
    if device.type != "cuda":
        return None
    if hasattr(torch, "GradScaler"):
        return torch.GradScaler("cuda")
    if hasattr(torch.cuda.amp, "GradScaler"):
        return torch.cuda.amp.GradScaler()
    return None


def try_compile(model, device):
    """Apply torch.compile if available and supported."""
    if not hasattr(torch, "compile"):
        return model
    if device.type == "mps":
        return model
    try:
        return torch.compile(model)
    except Exception:
        return model


def unwrap_model(model):
    """Get the underlying model from compiled wrapper for state_dict saving."""
    if hasattr(model, "_orig_mod"):
        return model._orig_mod
    return model


# ======================== Data ========================


class Cutout:
    def __init__(self, n_holes=1, length=16):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        h, w = img.shape[1], img.shape[2]
        mask = torch.ones(h, w)
        for _ in range(self.n_holes):
            y = torch.randint(0, h, (1,)).item()
            x = torch.randint(0, w, (1,)).item()
            y1 = max(0, y - self.length // 2)
            y2 = min(h, y + self.length // 2)
            x1 = max(0, x - self.length // 2)
            x2 = min(w, x + self.length // 2)
            mask[y1:y2, x1:x2] = 0.0
        img = img * mask.unsqueeze(0)
        return img


def get_dataloaders(batch_size=128, num_workers=0, data_dir=None):
    if data_dir is None:
        data_dir = str(PROJECT_ROOT / "data")

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        Cutout(n_holes=1, length=16),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=train_transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=False, transform=test_transform
    )

    use_pin = torch.cuda.is_available()
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=use_pin, persistent_workers=(num_workers > 0),
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_pin, persistent_workers=(num_workers > 0),
    )
    return trainloader, testloader


# ======================== Training ========================


def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, use_amp=False, epoch=0, total_epochs=0):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    n_batches = len(loader)
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
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

        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == n_batches:
            print(f"\r  [{epoch}/{total_epochs}] batch {batch_idx+1}/{n_batches} "
                  f"loss={total_loss/total:.4f} acc={100.*correct/total:.1f}%", end="", flush=True)
    print("\r" + " " * 80 + "\r", end="", flush=True)
    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp=False):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(1).eq(targets).sum().item()
        total += inputs.size(0)
    return total_loss / total, 100.0 * correct / total


def get_optimizer(name, params, lr, weight_decay, momentum=0.9):
    if name == "sgd":
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


def train(args):
    device = get_device()
    print(f"Device: {device}")

    model_config = {
        "blocks_per_stage": tuple(args.blocks),
        "channels": tuple(args.channels),
        "activation": args.activation,
        "dropout": args.dropout,
    }
    model, desc = build_model(model_config)
    model = model.to(device)

    if supports_channels_last(device):
        model = model.to(memory_format=torch.channels_last)

    model = try_compile(model, device)
    print(f"Model: {desc}")

    # AMP setup
    use_amp = args.amp and supports_amp(device)
    scaler = make_scaler(device) if use_amp else None
    if use_amp:
        print(f"AMP: enabled (dtype=float16, scaler={'yes' if scaler else 'no'})")

    trainloader, testloader = get_dataloaders(args.batch_size, args.workers)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = get_optimizer(args.optimizer, model.parameters(), args.lr, args.weight_decay)

    # Cosine annealing with warmup
    warmup_epochs = min(5, args.epochs // 4)
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs - warmup_epochs), eta_min=1e-6
    )
    scheduler_warmup = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=max(1, warmup_epochs)
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_epochs]
    )

    # Logging
    save_dir = PROJECT_ROOT / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / f"train_log_{args.run_name}.json"

    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": [], "lr": []}

    print(f"\n{'Epoch':>5} {'LR':>8} {'Train Loss':>10} {'Train Acc':>9} {'Test Loss':>9} {'Test Acc':>8} {'Time':>6}")
    print("-" * 65)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, trainloader, criterion, optimizer, device, scaler, use_amp, epoch, args.epochs)
        test_loss, test_acc = evaluate(model, testloader, criterion, device, use_amp)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["lr"].append(lr)

        is_best = test_acc > best_acc
        if is_best:
            best_acc = test_acc
            state = unwrap_model(model).state_dict()
            torch.save(state, save_dir / "part1_best.pth")

        marker = " *" if is_best else ""
        print(f"{epoch:>5} {lr:>8.6f} {train_loss:>10.4f} {train_acc:>8.2f}% {test_loss:>9.4f} {test_acc:>7.2f}%{marker} {elapsed:>5.1f}s")

    print(f"\nBest test accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {save_dir / 'part1_best.pth'}")

    # Save training log
    history["config"] = {**model_config, "optimizer": args.optimizer, "lr": args.lr,
                         "batch_size": args.batch_size, "epochs": args.epochs,
                         "label_smoothing": args.label_smoothing, "weight_decay": args.weight_decay}
    history["best_acc"] = best_acc
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Log saved to: {log_path}")

    return best_acc


def main():
    parser = argparse.ArgumentParser(description="Train CIFAR-10")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--optimizer", choices=["sgd", "adam", "adamw"], default="sgd")
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--activation", choices=["relu", "gelu", "silu"], default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--blocks", type=int, nargs=3, default=[3, 3, 3])
    parser.add_argument("--channels", type=int, nargs=3, default=[64, 128, 256])
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision (CUDA/MPS)")
    parser.add_argument("--run-name", type=str, default="default")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Run ablation: activations | optimizers | width | all")
    args = parser.parse_args()

    if args.experiment:
        run_experiments(args)
    else:
        train(args)


def run_experiments(args):
    """Run ablation experiments to compare different configurations."""
    results = {}

    experiments = {}
    if args.experiment in ("activations", "all"):
        experiments["activations"] = [
            {"activation": "relu", "run_name": "act_relu"},
            {"activation": "gelu", "run_name": "act_gelu"},
            {"activation": "silu", "run_name": "act_silu"},
        ]
    if args.experiment in ("optimizers", "all"):
        experiments["optimizers"] = [
            {"optimizer": "sgd", "lr": 0.1, "run_name": "opt_sgd"},
            {"optimizer": "adam", "lr": 1e-3, "run_name": "opt_adam"},
            {"optimizer": "adamw", "lr": 1e-3, "run_name": "opt_adamw"},
        ]
    if args.experiment in ("width", "all"):
        experiments["width"] = [
            {"channels": [32, 64, 128], "run_name": "width_small"},
            {"channels": [64, 128, 256], "run_name": "width_medium"},
            {"channels": [128, 256, 512], "run_name": "width_large"},
        ]

    for group_name, configs in experiments.items():
        print(f"\n{'='*60}")
        print(f"  Experiment group: {group_name}")
        print(f"{'='*60}")
        results[group_name] = {}
        for cfg in configs:
            run_name = cfg.pop("run_name")
            for k, v in cfg.items():
                setattr(args, k, v)
            args.run_name = run_name
            print(f"\n--- {run_name} ---")
            acc = train(args)
            results[group_name][run_name] = acc

    print(f"\n{'='*60}")
    print("  Experiment Summary")
    print(f"{'='*60}")
    for group, runs in results.items():
        print(f"\n[{group}]")
        for name, acc in runs.items():
            print(f"  {name}: {acc:.2f}%")

    summary_path = PROJECT_ROOT / "checkpoints" / "experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
