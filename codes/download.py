"""
一键下载数据集和模型权重（从 ModelScope）
用法: python download.py [--data] [--checkpoints] [--all]

助教复现步骤:
    1. pip install modelscope
    2. python download.py --all
    3. 按 README 运行训练/测试脚本
"""

import argparse
import os
import tarfile
from pathlib import Path

# ============ 配置区（提交前确认） ============
MODELSCOPE_REPO_ID = "YOUR_ID/NNDL-PJ2"  # TODO: 替换为你的 ModelScope 仓库 ID

# 文件映射: ModelScope仓库中的路径 -> 本地存放路径（相对于项目根目录）
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
    from modelscope.hub.api import HubApi
    from modelscope.hub.file_download import model_file_download

    local_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Downloading {filename} -> {local_path}")
    downloaded = model_file_download(
        model_id=repo_id,
        file_path=filename,
        local_dir=str(local_path.parent),
    )
    # model_file_download 可能把文件放在子目录中，做个兜底move
    actual = Path(downloaded) if downloaded else local_path
    if actual != local_path and actual.exists():
        actual.rename(local_path)


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
    parser = argparse.ArgumentParser(description="Download data and checkpoints from ModelScope")
    parser.add_argument("--data", action="store_true", help="Download dataset only")
    parser.add_argument("--checkpoints", action="store_true", help="Download model weights only")
    parser.add_argument("--all", action="store_true", help="Download everything")
    args = parser.parse_args()

    if not (args.data or args.checkpoints or args.all):
        args.all = True

    targets = []
    if args.all or args.data:
        targets.append("data")
    if args.all or args.checkpoints:
        targets.append("checkpoints")

    print(f"ModelScope Repo: {MODELSCOPE_REPO_ID}")
    print(f"Project Root: {PROJECT_ROOT}")
    print()

    for category in targets:
        print(f"[{category}]")
        for remote_path, local_rel in FILES[category].items():
            local_path = PROJECT_ROOT / local_rel
            if local_path.exists():
                print(f"  {local_rel} already exists, skipping.")
                continue
            download_from_modelscope(MODELSCOPE_REPO_ID, remote_path, local_path)
        print()

    if args.all or args.data:
        extract_cifar10()

    print("All done! You can now run the training/testing scripts.")


if __name__ == "__main__":
    main()
