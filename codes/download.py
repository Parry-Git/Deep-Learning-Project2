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
import sys
import tarfile
import urllib.request
from pathlib import Path

# ============ 配置区（提交前确认） ============
DEFAULT_MODELSCOPE_REPO_ID = "ParryY/Deep-Learning-Project2"
MODELSCOPE_REPO_ID = os.getenv("MODELSCOPE_REPO_ID", DEFAULT_MODELSCOPE_REPO_ID)

FILES = {
    "data": {
        "data/cifar-10-python.tar.gz": "data/cifar-10-python.tar.gz",
    },
    "checkpoints": {
        "checkpoints/part1_best.pth": "checkpoints/part1_best.pth",
        "checkpoints/vgg_bn/best_vgg_compare_e100_vgg_a.pth": (
            "checkpoints/vgg_bn/best_vgg_compare_e100_vgg_a.pth"
        ),
        "checkpoints/vgg_bn/best_vgg_compare_e100_vgg_bn.pth": (
            "checkpoints/vgg_bn/best_vgg_compare_e100_vgg_bn.pth"
        ),
    },
}
# =============================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


def is_placeholder_repo(repo_id: str) -> bool:
    return not repo_id or repo_id.lower() in {"none", "placeholder"}


def print_modelscope_hint(repo_id: str):
    print("  ModelScope repo is not configured.")
    print(f"  Current repo id: {repo_id!r}")
    print("  Set MODELSCOPE_REPO_ID or pass --repo-id after uploading the files.")
    print("  Example: MODELSCOPE_REPO_ID=ParryY/Deep-Learning-Project2 python codes/download.py --all")


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


def download_cifar10_direct(tar_path: Path):
    """Download CIFAR-10 tarball directly from the official source."""
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    ssl._create_default_https_context = ssl._create_unverified_context
    print(f"  Downloading CIFAR-10 tarball from {CIFAR10_URL}...")
    urllib.request.urlretrieve(CIFAR10_URL, tar_path)
    print("  Done.")


def download_via_torchvision():
    """Fallback: download CIFAR-10 directly from official source via torchvision."""
    try:
        import torchvision
    except ModuleNotFoundError:
        print("  torchvision is not installed; downloading CIFAR-10 tarball directly.")
        download_cifar10_direct(PROJECT_ROOT / "data" / "cifar-10-python.tar.gz")
        extract_cifar10()
        return

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
    parser.add_argument(
        "--repo-id",
        default=MODELSCOPE_REPO_ID,
        help="ModelScope repo id, e.g. ParryY/Deep-Learning-Project2. Can also use MODELSCOPE_REPO_ID.",
    )
    args = parser.parse_args()
    repo_id = args.repo_id
    repo_is_placeholder = is_placeholder_repo(repo_id)

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
        elif args.source == "torchvision" or repo_is_placeholder:
            if args.source == "modelscope" and repo_is_placeholder:
                print_modelscope_hint(repo_id)
                print("  Falling back to torchvision for CIFAR-10 data.")
            download_via_torchvision()
        else:
            tar_path = PROJECT_ROOT / "data" / "cifar-10-python.tar.gz"
            if not tar_path.exists():
                download_from_modelscope(
                    repo_id,
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
        elif repo_is_placeholder:
            print_modelscope_hint(repo_id)
            print("  Skipping checkpoints. Train locally to create them, or configure a real ModelScope repo.")
            if args.checkpoints and not args.all:
                sys.exit(2)
        else:
            for remote_path, local_rel in FILES["checkpoints"].items():
                local_path = PROJECT_ROOT / local_rel
                if local_path.exists():
                    print(f"  {local_rel} already exists, skipping.")
                    continue
                download_from_modelscope(repo_id, remote_path, local_path)
        print()

    print("All done!")


if __name__ == "__main__":
    main()
