"""
Gradient predictiveness and effective beta-smoothness for VGG-A +/- BN.

Following Santurkar et al. (NeurIPS 2018):
  At each training step t, record the gradient g_t and parameters theta_t.
  After the optimizer updates to theta_{t+1}, compute g_{t+1}.
  - Gradient predictiveness: ||g_{t+1} - g_t||_2
  - Effective beta-smoothness: ||g_{t+1} - g_t||_2 / ||theta_{t+1} - theta_t||_2

This measures how predictive the current gradient is of the nearby landscape,
which is the core question behind why BN helps optimization.

Usage:
    python codes/VGG_BatchNorm/VGG_Gradient_Landscape.py
    python codes/VGG_BatchNorm/VGG_Gradient_Landscape.py --epochs 5 --n-items 5000
"""

import argparse
import json
import math
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

VGG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VGG_DIR.parents[1]
sys.path.insert(0, str(VGG_DIR))

from cifar_loaders import get_cifar_loader
from train_vgg import build_model, get_device, set_random_seeds


def _flat_grad(model):
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().reshape(-1))
        else:
            grads.append(torch.zeros(p.numel(), device=p.device))
    return torch.cat(grads)


def _flat_params(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def run_experiment(args, model_name, device):
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
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    records = {
        "steps": [],
        "grad_predictiveness": [],
        "beta_smoothness": [],
        "grad_norm": [],
        "loss": [],
    }

    prev_grad = None
    prev_params = None
    global_step = 0

    model.train()
    for epoch in range(args.epochs):
        for inputs, targets in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            cur_grad = _flat_grad(model).cpu()
            cur_params = _flat_params(model).cpu()
            cur_loss = loss.item()

            if prev_grad is not None and global_step % args.sample_every == 0:
                grad_diff = (cur_grad - prev_grad).norm().item()
                param_diff = (cur_params - prev_params).norm().item()
                beta = grad_diff / max(param_diff, 1e-15)

                if math.isfinite(grad_diff) and math.isfinite(beta):
                    records["steps"].append(global_step)
                    records["grad_predictiveness"].append(grad_diff)
                    records["beta_smoothness"].append(beta)
                    records["grad_norm"].append(cur_grad.norm().item())
                    records["loss"].append(cur_loss)

            prev_grad = cur_grad.clone()
            prev_params = cur_params.clone()

            optimizer.step()
            global_step += 1

    n = len(records["steps"])
    print(f"  {display_name}: {n} samples, final loss={records['loss'][-1]:.4f}")

    return {
        "model": model_name,
        "display_name": display_name,
        "records": records,
    }


def smooth(values, window=7):
    if len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_all(results, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    colors = {"vgg_a": "#4C78A8", "vgg_bn": "#F58518"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    metrics = [
        ("grad_predictiveness", "Gradient Predictiveness\n"
         r"$\|\nabla L(\theta_{t+1}) - \nabla L(\theta_t)\|_2$"),
        ("beta_smoothness", "Effective β-smoothness\n"
         r"$\frac{\|\nabla L(\theta_{t+1}) - \nabla L(\theta_t)\|_2}"
         r"{\|\theta_{t+1} - \theta_t\|_2}$"),
        ("loss", "Training Loss"),
    ]

    window = 11
    for ax, (key, title) in zip(axes, metrics):
        for model_name, payload in results["models"].items():
            records = payload["records"]
            steps = np.array(records["steps"])
            values = np.array(records[key])
            color = colors.get(model_name, None)
            label = payload["display_name"]

            ax.plot(steps, values, linewidth=0.4, color=color, alpha=0.2)
            sv = smooth(values, window)
            ss = steps[: len(sv)] if len(sv) < len(steps) else steps
            ax.plot(ss[: len(sv)], sv, linewidth=2.0, color=color, label=label)

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Optimization Step")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("VGG-A vs VGG-A+BN: Gradient Landscape Analysis", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_focused(results, output_path):
    """Two-panel figure for the report: gradient predictiveness + beta-smoothness."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    colors = {"vgg_a": "#4C78A8", "vgg_bn": "#F58518"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    metrics = [
        ("grad_predictiveness", "Gradient Predictiveness"),
        ("beta_smoothness", "Effective β-smoothness"),
    ]
    ylabels = [
        r"$\|\nabla L(\theta_{t+1}) - \nabla L(\theta_t)\|_2$",
        r"$\frac{\|\Delta \nabla L\|}{\|\Delta \theta\|}$",
    ]

    window = 11
    for ax, (key, title), ylabel in zip(axes, metrics, ylabels):
        for model_name, payload in results["models"].items():
            records = payload["records"]
            steps = np.array(records["steps"])
            values = np.array(records[key])
            color = colors.get(model_name, None)
            label = payload["display_name"]

            ax.plot(steps, values, linewidth=0.5, color=color, alpha=0.15)
            sv = smooth(values, window)
            ss = steps[: len(sv)]
            ax.plot(ss, sv, linewidth=2.2, color=color, label=label)

        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Optimization Step")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)

    fig.suptitle(
        "How BN Helps Optimization: Gradient Landscape Comparison",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure gradient predictiveness and beta-smoothness for VGG +/- BN"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["vgg_a", "vgg_bn"],
        default=["vgg_a", "vgg_bn"],
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="sgd")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--n-items", type=int, default=5000)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--persistent-workers", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--worker-timeout", type=int, default=0)
    parser.add_argument("--mp-context", choices=["fork", "spawn", "forkserver"], default=None)
    parser.add_argument("--no-pin-memory", action="store_true")
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--run-name", type=str, default="gradient_landscape")
    parser.add_argument(
        "--from-log",
        type=Path,
        default=None,
        help="Only redraw figures from an existing JSON log",
    )
    return parser.parse_args()


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main():
    args = parse_args()
    plot_path = PROJECT_ROOT / "pic" / f"gradient_landscape_{args.run_name}.png"
    plot_focused_path = PROJECT_ROOT / "pic" / f"gradient_landscape_focused_{args.run_name}.png"
    log_path = (
        PROJECT_ROOT / "checkpoints" / "vgg_bn" / f"gradient_landscape_{args.run_name}.json"
    )

    if args.from_log is not None:
        with args.from_log.open("r", encoding="utf-8") as f:
            results = json.load(f)
        plot_all(results, plot_path)
        plot_focused(results, plot_focused_path)
        print(f"Loaded log: {args.from_log}")
        print(f"Saved 3-panel plot: {plot_path}")
        print(f"Saved 2-panel plot: {plot_focused_path}")
        return

    device = get_device()
    print(f"Device: {device}")
    print(f"Train subset: {args.n_items}, sample every {args.sample_every} step(s)")

    results = {"config": vars(args), "models": {}}

    for model_name in args.models:
        print(f"\n=== {model_name} ===")
        payload = run_experiment(args, model_name, device)
        results["models"][model_name] = {
            "display_name": payload["display_name"],
            "records": payload["records"],
        }

    save_json(results, log_path)
    plot_all(results, plot_path)
    plot_focused(results, plot_focused_path)

    for model_name, payload in results["models"].items():
        records = payload["records"]
        gp = np.array(records["grad_predictiveness"])
        bs = np.array(records["beta_smoothness"])
        print(f"\n{payload['display_name']}:")
        print(f"  Mean grad predictiveness: {gp.mean():.4f}")
        print(f"  Mean beta-smoothness:     {bs.mean():.2f}")
        print(f"  Max  grad predictiveness: {gp.max():.4f}")
        print(f"  Max  beta-smoothness:     {bs.max():.2f}")

    print(f"\nSaved log:          {log_path}")
    print(f"Saved 3-panel plot: {plot_path}")
    print(f"Saved 2-panel plot: {plot_focused_path}")


if __name__ == "__main__":
    main()
