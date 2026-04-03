"""
CIFAR-10 task splitting for Continual Learning.

Divides CIFAR-10 into 5 sequential tasks, each containing 2 classes:
  Task 0: airplane (0), automobile (1)
  Task 1: bird (2),     cat (3)
  Task 2: deer (4),     dog (5)
  Task 3: frog (6),     horse (7)
  Task 4: ship (8),     truck (9)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# Each task contains 2 consecutive original CIFAR-10 class indices.
TASK_CLASSES: List[List[int]] = [
    [0, 1],  # Task 0 – airplane, automobile
    [2, 3],  # Task 1 – bird, cat
    [4, 5],  # Task 2 – deer, dog
    [6, 7],  # Task 3 – frog, horse
    [8, 9],  # Task 4 – ship, truck
]

NUM_TASKS = len(TASK_CLASSES)
CLASSES_PER_TASK = 2

# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

# Standard single-crop augmentation used during supervised training.
TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

# Two-view augmentation used for contrastive pre-training (SimCLR / SupCon style).
class TwoViewTransform:
    """Applies the same base transform twice to produce two augmented views."""

    _contrastive_base = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    def __call__(self, x):
        return self._contrastive_base(x), self._contrastive_base(x)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

class TaskSubset(Dataset):
    """A subset of CIFAR-10 restricted to the classes of a given task.

    Labels are remapped so that each task always uses labels {0, 1} for
    Task-IL evaluation.  The original CIFAR-10 labels are preserved in the
    ``global_labels`` attribute for Class-IL evaluation.
    """

    def __init__(
        self,
        cifar_dataset: datasets.CIFAR10,
        task_id: int,
        transform=None,
    ) -> None:
        self.task_id = task_id
        self.classes = TASK_CLASSES[task_id]
        self.transform = transform

        # Collect indices that belong to this task.
        targets = np.array(cifar_dataset.targets)
        mask = np.isin(targets, self.classes)
        indices = np.where(mask)[0]

        self.data = cifar_dataset.data[indices]       # (N, 32, 32, 3) uint8
        self.global_labels = targets[indices].tolist()  # original CIFAR-10 labels
        # Local labels within the task: class index 0 or 1.
        self.local_labels = [
            self.classes.index(lbl) for lbl in self.global_labels
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        from PIL import Image

        img = Image.fromarray(self.data[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img, self.local_labels[idx], self.global_labels[idx]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class CIFAR10Tasks:
    """Container for all task-split CIFAR-10 datasets.

    Parameters
    ----------
    data_root:
        Directory where CIFAR-10 will be downloaded / is already stored.
    contrastive:
        If ``True`` the training sets use :class:`TwoViewTransform` so each
        sample returns a pair of augmented views.  Useful during contrastive
        pre-training.
    download:
        Whether to download CIFAR-10 if not present.
    """

    def __init__(
        self,
        data_root: str | Path = "./data/cifar10",
        contrastive: bool = False,
        download: bool = True,
    ) -> None:
        self.data_root = Path(data_root)
        self.contrastive = contrastive

        train_transform = TwoViewTransform() if contrastive else TRAIN_TRANSFORM

        self._cifar_train = datasets.CIFAR10(
            root=str(self.data_root), train=True, download=download
        )
        self._cifar_test = datasets.CIFAR10(
            root=str(self.data_root), train=False, download=download
        )

        self.train_tasks: List[TaskSubset] = []
        self.test_tasks: List[TaskSubset] = []

        for task_id in range(NUM_TASKS):
            self.train_tasks.append(
                TaskSubset(self._cifar_train, task_id, transform=train_transform)
            )
            self.test_tasks.append(
                TaskSubset(self._cifar_test, task_id, transform=EVAL_TRANSFORM)
            )

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    def get_train(self, task_id: int) -> TaskSubset:
        """Return the training split for *task_id*."""
        return self.train_tasks[task_id]

    def get_test(self, task_id: int) -> TaskSubset:
        """Return the test split for *task_id*."""
        return self.test_tasks[task_id]

    @property
    def num_tasks(self) -> int:
        return NUM_TASKS

    @property
    def classes_per_task(self) -> int:
        return CLASSES_PER_TASK

    @property
    def total_classes(self) -> int:
        return NUM_TASKS * CLASSES_PER_TASK


def get_task_dataloaders(
    task_id: int,
    data_root: str | Path = "./data/cifar10",
    batch_size: int = 128,
    num_workers: int = 2,
    contrastive: bool = False,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """Return (train_loader, test_loader) for a single task.

    Parameters
    ----------
    task_id:
        Index in [0, NUM_TASKS).
    data_root:
        Root directory for CIFAR-10.
    batch_size:
        Mini-batch size.
    num_workers:
        DataLoader worker processes.
    contrastive:
        Use two-view augmentation for contrastive training.
    download:
        Download CIFAR-10 if not already present.
    """
    tasks = CIFAR10Tasks(data_root=data_root, contrastive=contrastive, download=download)

    train_loader = DataLoader(
        tasks.get_train(task_id),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        tasks.get_test(task_id),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader
