"""
一键下载数据集和模型权重

方式一（推荐）: 从 ModelScope 下载（包含预训练权重）
    python download.py --all

方式二: 仅下载数据集（从官方源，不需要 modelscope 库）
    python download.py --data --source torchvision

助教复现步骤:
    1. pip install torch torchvision modelscope matplotlib
    2. python codes/download.py --all
    3. 按 README 运行训练/测试脚本
"""

import argparse
import os
import ssl
import tarfile
from pathlib import Path

# ============ 配置区（提交前确认） ============
MODELSCOPE_REPO_ID = "YOUR_ID/NNDL-PJ2"  # TODO: 替换为你的 ModelScope 仓库 ID

FILES = {
    "data": {
        "data/cifar-10-python.tar.gz": "data/cifar-10-python.tar.gz",
    },
    "checkpoints": {
        "checkpoints/part1_best.pth": "checkpoints/part1_best.pth",
        "checkpoints/part2_vgg_a.pth": "checkpoints/part2_vgg_a.pth",
        "checkpoints/part2_vgg_bn.pth": "checkpoints/part2_vgg_bn.pth",
    },
}
# =============================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def download_from_modelscope(repo_id: str, filename: str, local_path: Path):
    from modelscope.hub.file_download import model_file_download

    local_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {filename} -> {local_path}")
    downloaded = model_file_download(
        model_id=repo_id,
        file_path=filename,
        local_dir=str(local_path.parent),
    )
    actual = Path(downloaded) if downloaded else local_path
    if actual != local_path and actual.exists():
        actual.rename(local_path)


def download_via_torchvision():
    """Fallback: download CIFAR-10 directly from official source via torchvision."""
    import torchvision

    ssl._create_default_https_context = ssl._create_unverified_context
    data_dir = str(PROJECT_ROOT / "data")
    print("  Downloading CIFAR-10 via torchvision...")
    torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True)
    torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True)
    print("  Done.")


def extract_cifar10():
    tar_path = PROJECT_ROOT / "data" / "cifar-10-python.tar.gz"
    extract_dir = PROJECT_ROOT / "data"
    if (extract_dir / "cifar-10-batches-py").exists():
        print("  CIFAR-10 already extracted, skipping.")
        return
    if tar_path.exists():
        print("  Extracting CIFAR-10...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=extract_dir)
        print("  Done.")


def main():
    parser = argparse.ArgumentParser(description="Download data and checkpoints")
    parser.add_argument("--data", action="store_true", help="Download dataset only")
    parser.add_argument("--checkpoints", action="store_true", help="Download model weights only")
    parser.add_argument("--all", action="store_true", help="Download everything")
    parser.add_argument(
        "--source",
        choices=["modelscope", "torchvision"],
        default="modelscope",
        help="Data source: modelscope (default, includes checkpoints) or torchvision (data only)",
    )
    args = parser.parse_args()

    if not (args.data or args.checkpoints or args.all):
        args.all = True

    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Source: {args.source}")
    print()

    # Download data
    if args.all or args.data:
        print("[data]")
        data_ready = (PROJECT_ROOT / "data" / "cifar-10-batches-py").exists()
        if data_ready:
            print("  CIFAR-10 already exists, skipping.")
        elif args.source == "torchvision":
            download_via_torchvision()
        else:
            tar_path = PROJECT_ROOT / "data" / "cifar-10-python.tar.gz"
            if not tar_path.exists():
                download_from_modelscope(
                    MODELSCOPE_REPO_ID,
                    "data/cifar-10-python.tar.gz",
                    tar_path,
                )
            extract_cifar10()
        print()

    # Download checkpoints
    if args.all or args.checkpoints:
        print("[checkpoints]")
        if args.source == "torchvision":
            print("  Skipping checkpoints (not available from torchvision source).")
        else:
            for remote_path, local_rel in FILES["checkpoints"].items():
                local_path = PROJECT_ROOT / local_rel
                if local_path.exists():
                    print(f"  {local_rel} already exists, skipping.")
                    continue
                download_from_modelscope(MODELSCOPE_REPO_ID, remote_path, local_path)
        print()

    print("All done!")


if __name__ == "__main__":
    main()
