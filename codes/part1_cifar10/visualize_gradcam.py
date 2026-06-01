"""
Grad-CAM visualization for CifarResNet network interpretation.

Generates class activation maps showing which spatial regions the network
attends to when making predictions, following Selvaraju et al. (ICCV 2017).

Usage:
    python codes/part1_cifar10/visualize_gradcam.py
    python codes/part1_cifar10/visualize_gradcam.py --checkpoint checkpoints/part1_best.pth
"""

import argparse
import math
import ssl
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
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


class GradCAM:
    """Grad-CAM for CifarResNet: hooks into the last residual stage."""

    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        target_layer = self.model.stages[-1]

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    @torch.enable_grad()
    def __call__(self, input_tensor, target_class=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1)

        one_hot = torch.zeros_like(output)
        for i, c in enumerate(target_class):
            one_hot[i, c] = 1.0
        output.backward(gradient=one_hot)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)

        batch_cams = []
        for i in range(cam.shape[0]):
            c = cam[i]
            c_min, c_max = c.min(), c.max()
            if c_max - c_min > 1e-8:
                c = (c - c_min) / (c_max - c_min)
            else:
                c = torch.zeros_like(c)
            batch_cams.append(c.cpu().numpy())

        probs = torch.softmax(output.detach(), dim=1)
        confs, preds = probs.max(dim=1)

        return batch_cams, preds.cpu().tolist(), confs.cpu().tolist()


def overlay_cam(image_np, cam_np, alpha=0.45):
    cmap = plt.cm.jet
    heatmap = cmap(cam_np)[:, :, :3]
    overlaid = (1 - alpha) * image_np + alpha * heatmap
    return np.clip(overlaid, 0, 1)


def plot_gradcam_grid(images, targets, cams, preds, confs, output, cols=4):
    n = len(images)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 4.2, rows * 2.6))
    if rows == 1:
        axes = axes.reshape(1, -1)

    for idx in range(rows * cols):
        ax_orig = axes[idx // cols, (idx % cols) * 2]
        ax_cam = axes[idx // cols, (idx % cols) * 2 + 1]

        ax_orig.axis("off")
        ax_cam.axis("off")

        if idx >= n:
            continue

        img_np = denormalize(images[idx])
        cam_np = cams[idx]
        overlaid = overlay_cam(img_np, cam_np)

        ax_orig.imshow(img_np)
        ax_cam.imshow(overlaid)

        target = targets[idx]
        pred = preds[idx]
        conf = confs[idx]
        correct = target == pred
        color = "#15803d" if correct else "#b91c1c"

        label = f"Pred: {CLASSES[pred]} {conf * 100:.1f}%\nTrue: {CLASSES[target]}"
        ax_orig.set_title(label, fontsize=8, color=color, pad=3)
        ax_cam.set_title("Grad-CAM", fontsize=8, color="#444", pad=3)

    fig.suptitle("Network Interpretation via Grad-CAM", fontsize=13, y=1.01)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM visualization for CifarResNet")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "part1_best.pth")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "pic" / "part1_gradcam.png")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--channels", type=int, nargs=3, default=[64, 128, 256])
    parser.add_argument("--blocks", type=int, nargs=3, default=[3, 3, 3])
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

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
        root=str(PROJECT_ROOT / "data"), train=False, download=False, transform=transform
    )

    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(dataset), generator=generator)[: args.num_samples].tolist()

    images, targets = [], []
    for idx in indices:
        img, target = dataset[idx]
        images.append(img)
        targets.append(target)

    input_batch = torch.stack(images).to(device)

    gradcam = GradCAM(model)
    cams, preds, confs = gradcam(input_batch)

    plot_gradcam_grid(images, targets, cams, preds, confs, args.output, cols=args.cols)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
