"""
Loss landscape experiment for VGG-A with and without BatchNorm.

The script trains each architecture with several learning rates, records the
training loss at every optimization step, then plots the min/max loss band over
learning rates with matplotlib.fill_between.
"""

import argparse
import json
import os
import sys
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

from cifar_loaders import get_cifar_loader
from models.vgg import get_number_of_parameters
from train_vgg import build_model, get_device, make_scaler, save_json, set_random_seeds


def build_optimizer(args, model, lr):
    if args.optimizer == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    if args.optimizer == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    return optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)


def train_loss_curve(args, model_name, lr, device):
    set_random_seeds(args.seed, device)
    train_loader = get_cifar_loader(
        train=True,
        batch_size=args.batch_size,
        num_workers=args.workers,
        n_items=args.n_items,
        augment=False,
        shuffle=True,
        pin_memory=False if args.no_pin_memory else None,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        worker_timeout=args.worker_timeout,
        mp_context=args.mp_context,
        seed=args.seed,
    )

    model, display_name = build_model(model_name)
    model = model.to(device)
    optimizer = build_optimizer(args, model, lr)
    criterion = nn.CrossEntropyLoss()
    use_amp = args.amp and device.type == "cuda"
    scaler = make_scaler(device, enabled=use_amp)
    losses = []

    model.train()
    for _ in range(args.epochs):
        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            losses.append(float(loss.detach().cpu()))
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

    if args.save_checkpoints:
        ckpt_dir = PROJECT_ROOT / "checkpoints" / "vgg_bn" / "landscape_models"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_dir / f"{args.run_name}_{model_name}_lr{lr:g}.pth")

    return {
        "model": model_name,
        "display_name": display_name,
        "lr": lr,
        "params": get_number_of_parameters(model),
        "losses": losses,
    }


def make_band(curves):
    min_len = min(len(curve["losses"]) for curve in curves)
    values = np.array([curve["losses"][:min_len] for curve in curves], dtype=np.float64)
    return {
        "steps": list(range(1, min_len + 1)),
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
    }


def plot_landscape(results, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=False)
    colors = {
        "vgg_a": "#4C78A8",
        "vgg_bn": "#F58518",
    }

    def draw_panel(ax, start_step, title):
        for model_name, payload in results["models"].items():
            band = payload["band"]
            steps = np.array(band["steps"])
            mask = steps >= start_step
            min_curve = np.array(band["min"])[mask]
            max_curve = np.array(band["max"])[mask]
            mean_curve = np.array(band["mean"])[mask]
            panel_steps = steps[mask]
            label = payload["display_name"]
            color = colors.get(model_name, None)
            ax.fill_between(
                panel_steps,
                min_curve,
                max_curve,
                alpha=0.18,
                color=color,
                label=f"{label} lr range",
            )
            ax.plot(panel_steps, mean_curve, linewidth=2.0, color=color, label=f"{label} mean")

        ax.set_title(title)
        ax.set_xlabel("Optimization Step")
        ax.grid(alpha=0.25)

    draw_panel(axes[0], start_step=1, title="Full per-step loss band")
    draw_panel(axes[1], start_step=10, title="Zoom after early transient")
    axes[0].set_ylabel("Training Loss")
    axes[1].legend(fontsize=8, loc="upper right")
    fig.suptitle("VGG Loss Bands over Learning Rates", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot VGG-A loss landscape with and without BN")
    parser.add_argument("--models", nargs="+", choices=["vgg_a", "vgg_bn"], default=["vgg_a", "vgg_bn"])
    parser.add_argument("--lrs", type=float, nargs="+", default=[1e-3, 2e-3, 1e-4, 5e-4])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-items", type=int, default=5000)
    parser.add_argument("--optimizer", choices=["adam", "adamw", "sgd"], default="adam")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--worker-timeout", type=int, default=0)
    parser.add_argument("--mp-context", choices=["fork", "spawn", "forkserver"], default=None)
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable AMP on CUDA; use --no-amp to disable")
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--from-log", type=Path, default=None,
                        help="Only redraw the figure from an existing landscape JSON log")
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--run-name", type=str, default="vgg_loss_landscape")
    return parser.parse_args()


def main():
    args = parse_args()
    plot_path = PROJECT_ROOT / "pic" / f"loss_landscape_{args.run_name}.png"
    if args.from_log is not None:
        with args.from_log.open("r", encoding="utf-8") as f:
            results = json.load(f)
        plot_landscape(results, plot_path)
        print(f"Loaded log: {args.from_log}")
        print(f"Saved plot: {plot_path}")
        return

    device = get_device()
    if args.workers > 0 and device.type == "cuda" and args.mp_context is None:
        args.mp_context = "forkserver"
    print(f"Device: {device}")
    print(f"Learning rates: {args.lrs}")
    print(f"Train subset for landscape: {args.n_items}")

    results = {
        "config": vars(args),
        "models": {},
    }

    for model_name in args.models:
        curves = []
        display_name = None
        for lr in args.lrs:
            print(f"Running {model_name} lr={lr:g} ...")
            curve = train_loss_curve(args, model_name, lr, device)
            display_name = curve["display_name"]
            curves.append(curve)
        results["models"][model_name] = {
            "display_name": display_name,
            "curves": curves,
            "band": make_band(curves),
        }

    log_path = PROJECT_ROOT / "checkpoints" / "vgg_bn" / f"loss_landscape_{args.run_name}.json"
    save_json(results, log_path)
    plot_landscape(results, plot_path)
    print(f"Saved log: {log_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
