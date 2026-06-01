"""
CIFAR-10 training script.

Usage:
    python train.py                          # default config
    python train.py --epochs 200 --lr 0.1    # custom hyperparams
    python train.py --no-amp                 # disable mixed precision
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
import copy
import json
import os
import ssl
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def try_compile(model, device, enabled=True):
    """Apply torch.compile if available and supported."""
    if not enabled:
        return model
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


def get_dataloaders(
    batch_size=128,
    num_workers=0,
    data_dir=None,
    pin_memory=None,
    persistent_workers=False,
    prefetch_factor=2,
    worker_timeout=0,
    mp_context=None,
):
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

    use_pin = torch.cuda.is_available() if pin_memory is None else pin_memory
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": use_pin,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["timeout"] = worker_timeout
        if mp_context:
            loader_kwargs["multiprocessing_context"] = mp_context

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True,
        **loader_kwargs,
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False,
        **loader_kwargs,
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


class FocalLoss(nn.Module):
    """Multi-class focal loss for CIFAR-10 ablation experiments."""

    def __init__(self, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits,
            targets,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


def build_criterion(args):
    if args.loss == "ce":
        return nn.CrossEntropyLoss()
    if args.loss == "label_smoothing":
        return nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    if args.loss == "focal":
        return FocalLoss(gamma=args.focal_gamma, label_smoothing=args.label_smoothing)
    raise ValueError(f"Unknown loss: {args.loss}")


def save_history(history, log_path):
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)


def plot_history(history, plot_path):
    epochs = range(1, len(history["train_loss"]) + 1)
    if not history["train_loss"]:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history["test_loss"], label="Test Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train Acc", linewidth=2)
    axes[1].plot(epochs, history["test_acc"], label="Test Acc", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Accuracy Curves")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    best_epoch = history.get("best_epoch")
    best_acc = history.get("best_acc")
    if best_epoch and best_acc is not None:
        axes[1].scatter([best_epoch], [best_acc], color="#b91c1c", zorder=3)
        axes[1].annotate(
            f"best {best_acc:.2f}%",
            xy=(best_epoch, best_acc),
            xytext=(8, -12),
            textcoords="offset points",
            fontsize=9,
            color="#b91c1c",
        )

    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)


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

    mp_context = args.mp_context
    if mp_context is None and args.workers > 0 and device.type == "cuda":
        mp_context = "forkserver"

    # On Linux/WSL, the default worker start method is fork; forking after CUDA
    # init can hang, so CUDA + multi-worker defaults to forkserver above.
    trainloader, testloader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=not args.no_pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        worker_timeout=args.worker_timeout,
        mp_context=mp_context,
    )

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

    model = try_compile(model, device, enabled=not args.no_compile)
    print(f"Model: {desc}")

    # AMP setup
    use_amp = args.amp and supports_amp(device)
    scaler = make_scaler(device) if use_amp else None
    if use_amp:
        print(f"AMP: enabled (dtype=float16, scaler={'yes' if scaler else 'no'})")

    criterion = build_criterion(args)
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
    plot_path = save_dir / f"training_curves_{args.run_name}.png"
    best_path = None if args.no_save_checkpoint else save_dir / args.checkpoint_name

    best_acc = 0.0
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "lr": [],
        "best_acc": best_acc,
        "best_epoch": None,
        "artifacts": {
            "log": str(log_path),
            "plot": str(plot_path),
            "checkpoint": str(best_path) if best_path else None,
        },
        "config": {
            **model_config,
            "loss": args.loss,
            "optimizer": args.optimizer,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "label_smoothing": args.label_smoothing,
            "focal_gamma": args.focal_gamma,
            "weight_decay": args.weight_decay,
            "workers": args.workers,
            "amp": args.amp,
        },
    }

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
            history["best_acc"] = best_acc
            history["best_epoch"] = epoch
            if best_path is not None:
                state = unwrap_model(model).state_dict()
                torch.save(state, best_path)

        marker = " *" if is_best else ""
        print(f"{epoch:>5} {lr:>8.6f} {train_loss:>10.4f} {train_acc:>8.2f}% {test_loss:>9.4f} {test_acc:>7.2f}%{marker} {elapsed:>5.1f}s")
        save_history(history, log_path)

    print(f"\nBest test accuracy: {best_acc:.2f}%")
    if best_path is not None:
        print(f"Model saved to: {best_path}")
    else:
        print("Model checkpoint saving disabled for this run.")

    # Save training log and curves
    save_history(history, log_path)
    plot_history(history, plot_path)
    print(f"Log saved to: {log_path}")
    print(f"Curves saved to: {plot_path}")

    return history


def main():
    parser = argparse.ArgumentParser(description="Train CIFAR-10")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--optimizer", choices=["sgd", "adam", "adamw"], default="sgd")
    parser.add_argument("--loss", choices=["ce", "label_smoothing", "focal"], default="label_smoothing")
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--activation", choices=["relu", "gelu", "silu"], default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--blocks", type=int, nargs=3, default=[3, 3, 3])
    parser.add_argument("--channels", type=int, nargs=3, default=[64, 128, 256])
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--persistent-workers", action="store_true",
                        help="Keep DataLoader workers alive between epochs")
    parser.add_argument("--prefetch-factor", type=int, default=2,
                        help="Number of batches prefetched by each worker")
    parser.add_argument("--worker-timeout", type=int, default=0,
                        help="DataLoader worker timeout in seconds; 0 disables timeout")
    parser.add_argument("--mp-context", choices=["fork", "spawn", "forkserver"], default=None,
                        help="DataLoader multiprocessing start method")
    parser.add_argument("--no-pin-memory", action="store_true",
                        help="Disable DataLoader pin_memory")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable mixed precision when supported; use --no-amp to disable")
    parser.add_argument("--run-name", type=str, default="default")
    parser.add_argument("--checkpoint-name", type=str, default="part1_best.pth")
    parser.add_argument("--no-save-checkpoint", action="store_true",
                        help="Disable checkpoint saving; useful for large ablation sweeps")
    parser.add_argument("--save-ablation-checkpoints", action="store_true",
                        help="Save a separate best checkpoint for each ablation run")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Run ablation: baseline | activations | optimizers | width | losses | regularization | all")
    args = parser.parse_args()

    if args.experiment:
        run_experiments(args)
    else:
        train(args)


def run_experiments(args):
    """Run ablation experiments to compare different configurations."""
    results = {}
    base_run_name = args.run_name

    experiments = {}
    if args.experiment in ("baseline", "all"):
        experiments["baseline"] = [
            {"run_name": "baseline"},
        ]
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
    if args.experiment in ("losses", "all"):
        experiments["losses"] = [
            {"loss": "ce", "label_smoothing": 0.0, "run_name": "loss_ce"},
            {"loss": "label_smoothing", "label_smoothing": 0.05, "run_name": "loss_ls005"},
            {"loss": "label_smoothing", "label_smoothing": 0.1, "run_name": "loss_ls010"},
            {"loss": "focal", "label_smoothing": 0.0, "focal_gamma": 2.0, "run_name": "loss_focal"},
        ]
    if args.experiment in ("regularization", "all"):
        experiments["regularization"] = [
            {"dropout": 0.0, "weight_decay": 5e-4, "run_name": "reg_dropout000"},
            {"dropout": 0.1, "weight_decay": 5e-4, "run_name": "reg_dropout010"},
            {"dropout": 0.2, "weight_decay": 5e-4, "run_name": "reg_dropout020"},
            {"dropout": 0.1, "weight_decay": 1e-4, "run_name": "reg_wd0001"},
        ]

    if not experiments:
        raise ValueError(f"Unknown experiment group: {args.experiment}")

    for group_name, configs in experiments.items():
        print(f"\n{'='*60}")
        print(f"  Experiment group: {group_name}")
        print(f"{'='*60}")
        results[group_name] = {}
        for cfg in configs:
            run_name = f"{base_run_name}_{cfg['run_name']}"
            run_args = argparse.Namespace(**copy.deepcopy(vars(args)))
            run_args.experiment = None
            run_args.run_name = run_name
            run_args.no_save_checkpoint = not args.save_ablation_checkpoints
            if args.save_ablation_checkpoints:
                run_args.checkpoint_name = f"part1_best_{run_name}.pth"
            for k, v in cfg.items():
                if k != "run_name":
                    setattr(run_args, k, v)
            print(f"\n--- {run_name} ---")
            history = train(run_args)
            results[group_name][run_name] = {
                "best_acc": history["best_acc"],
                "best_epoch": history["best_epoch"],
                "final_acc": history["test_acc"][-1],
                "final_loss": history["test_loss"][-1],
                "config": history["config"],
                "artifacts": history["artifacts"],
            }

    print(f"\n{'='*60}")
    print("  Experiment Summary")
    print(f"{'='*60}")
    for group, runs in results.items():
        print(f"\n[{group}]")
        for name, payload in runs.items():
            print(
                f"  {name}: best={payload['best_acc']:.2f}% "
                f"@epoch {payload['best_epoch']} | final={payload['final_acc']:.2f}%"
            )

    summary_path = PROJECT_ROOT / "checkpoints" / f"experiment_summary_{base_run_name}.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
