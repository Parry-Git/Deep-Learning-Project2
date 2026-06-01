import random
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


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
        return img * mask.unsqueeze(0)


def build_transform(train=False, augment=True):
    if train and augment:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
            Cutout(n_holes=1, length=16),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def _subset_dataset(dataset, n_items=None, seed=0):
    if n_items is None or n_items <= 0 or n_items >= len(dataset):
        return dataset
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    return Subset(dataset, indices[:n_items])


def get_cifar_loader(
    train=True,
    batch_size=128,
    num_workers=0,
    n_items=None,
    data_dir=None,
    augment=True,
    shuffle=None,
    pin_memory=None,
    persistent_workers=False,
    prefetch_factor=2,
    worker_timeout=0,
    mp_context=None,
    seed=0,
):
    if data_dir is None:
        data_dir = PROJECT_ROOT / "data"
    transform = build_transform(train=train, augment=augment)
    dataset = torchvision.datasets.CIFAR10(
        root=str(data_dir),
        train=train,
        download=False,
        transform=transform,
    )
    dataset = _subset_dataset(dataset, n_items=n_items, seed=seed)

    if shuffle is None:
        shuffle = train
    generator = torch.Generator()
    generator.manual_seed(seed)
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

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        **loader_kwargs,
    )


def get_cifar_loaders(
    batch_size=128,
    num_workers=0,
    n_items=None,
    val_items=None,
    data_dir=None,
    augment=True,
    pin_memory=None,
    persistent_workers=False,
    prefetch_factor=2,
    worker_timeout=0,
    mp_context=None,
    seed=0,
):
    train_loader = get_cifar_loader(
        train=True,
        batch_size=batch_size,
        num_workers=num_workers,
        n_items=n_items,
        data_dir=data_dir,
        augment=augment,
        shuffle=True,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        worker_timeout=worker_timeout,
        mp_context=mp_context,
        seed=seed,
    )
    val_loader = get_cifar_loader(
        train=False,
        batch_size=batch_size,
        num_workers=num_workers,
        n_items=val_items,
        data_dir=data_dir,
        augment=False,
        shuffle=False,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        worker_timeout=worker_timeout,
        mp_context=mp_context,
        seed=seed + 1,
    )
    return train_loader, val_loader
