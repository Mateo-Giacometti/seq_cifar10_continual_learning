from dataclasses import dataclass
import random
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


@dataclass
class TaskConfig:
    """
    Configuration for sequential tasks on CIFAR-10.

    Attributes
    ----------
    n_tasks : int
        The number of sequential tasks to create.
    classes_per_task : int
        The number of classes per task.
    class_order : Optional[List[int]]
        The order of classes for each task. If None, a random order will be used.
    """

    n_tasks: int = 5
    classes_per_task: int = 2
    class_order: Optional[List[int]] = None


class CIFAR10TaskDataset(Dataset):
    def __init__(self, base_dataset: datasets.CIFAR10, indices: List[int], transform: Optional[Callable] = None, 
                 target_transform: Optional[Callable] = None,) -> None:
        """
        Dataset wrapper for a specific task in CIFAR-10.

        Parameters
        ----------
        base_dataset : datasets.CIFAR10
            The original CIFAR-10 dataset.
        indices : List[int]
            The indices of the samples that belong to this task.
        transform : Optional[Callable]
            The transformation to apply to the images.
        target_transform : Optional[Callable]
            The transformation to apply to the labels.
        """
        self.base_dataset = base_dataset
        self.indices = indices
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        base_idx = self.indices[idx]
        image, target = self.base_dataset[base_idx]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target


class SeqCIFAR10Builder:
    def __init__(self, root: str, config: TaskConfig) -> None:
        """
        Initialize the sequential CIFAR-10 builder.

        Parameters
        ----------
        root : str
            The root directory where the CIFAR-10 dataset is stored.
        config : TaskConfig
            The configuration for the sequential tasks.
        """
        self.root = root
        self.config = config

        if config.n_tasks * config.classes_per_task > 10:
            raise ValueError("n_tasks * classes_per_task must be <= 10 for CIFAR-10")

        if config.class_order is None:
            class_order = list(range(10))
        else:
            class_order = config.class_order
            if sorted(class_order) != list(range(10)):
                raise ValueError("class_order must be a permutation of [0..9]")

        self.class_order = class_order
        self.task_classes = [
            self.class_order[
                i * config.classes_per_task : (i + 1) * config.classes_per_task
            ]
            for i in range(config.n_tasks)
        ]

        self.train_base = datasets.CIFAR10(
            root=self.root, train=True, download=True, transform=None
        )
        self.test_base = datasets.CIFAR10(
            root=self.root, train=False, download=True, transform=None
        )

        self.train_indices_by_task = self._build_task_indices(self.train_base.targets)
        self.test_indices_by_task = self._build_task_indices(self.test_base.targets)

    def _build_task_indices(self, targets: List[int]) -> Dict[int, List[int]]:
        task_indices: Dict[int, List[int]] = {}
        for task_id, classes in enumerate(self.task_classes):
            class_set = set(classes)
            indices = [i for i, y in enumerate(targets) if y in class_set]
            task_indices[task_id] = indices
        return task_indices

    def get_task_dataset(
        self,
        task_id: int,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> Dataset:
        base = self.train_base if train else self.test_base
        indices = (
            self.train_indices_by_task[task_id]
            if train
            else self.test_indices_by_task[task_id]
        )
        return CIFAR10TaskDataset(
            base, indices, transform=transform, target_transform=target_transform
        )

    def build_task_loaders(
        self,
        batch_size: int,
        transform: Optional[Callable],
        train_shuffle: bool = True,
        num_workers: int = 0,
    ) -> Tuple[Dict[int, DataLoader], Dict[int, DataLoader]]:
        train_loaders: Dict[int, DataLoader] = {}
        test_loaders: Dict[int, DataLoader] = {}
        loader_kwargs = {
            "num_workers": num_workers,
            "pin_memory": torch.cuda.is_available(),
            "persistent_workers": num_workers > 0,
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = 4

        for task_id in range(self.config.n_tasks):
            train_ds = self.get_task_dataset(
                task_id, train=True, transform=transform, target_transform=None
            )
            test_ds = self.get_task_dataset(
                task_id, train=False, transform=transform, target_transform=None
            )
            train_loaders[task_id] = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=train_shuffle,
                **loader_kwargs,
            )
            test_loaders[task_id] = DataLoader(
                test_ds,
                batch_size=batch_size,
                shuffle=False,
                **loader_kwargs,
            )
        return train_loaders, test_loaders

    def summary_rows(self) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for task_id, classes in enumerate(self.task_classes):
            rows.append(
                {
                    "task_id": task_id,
                    "classes": classes,
                    "class_names": [CIFAR10_CLASSES[c] for c in classes],
                    "n_train": len(self.train_indices_by_task[task_id]),
                    "n_test": len(self.test_indices_by_task[task_id]),
                }
            )
        return rows


class ReservoirReplayBuffer:
    def __init__(self, capacity: int, seed: int = 42) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.capacity = capacity
        self.images: List[torch.Tensor] = []
        self.labels: List[int] = []
        self.seen = 0
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.images)

    def add(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        images_cpu = images.detach().cpu()
        labels_cpu = labels.detach().cpu()

        for image, label in zip(images_cpu, labels_cpu):
            self.seen += 1
            if len(self.images) < self.capacity:
                self.images.append(image.clone())
                self.labels.append(int(label.item()))
            else:
                j = self.rng.randint(0, self.seen - 1)
                if j < self.capacity:
                    self.images[j] = image.clone()
                    self.labels[j] = int(label.item())

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.images) == 0:
            raise ValueError("Cannot sample from an empty replay buffer")
        k = min(batch_size, len(self.images))
        idx = self.rng.sample(range(len(self.images)), k=k)
        batch_images = torch.stack([self.images[i] for i in idx], dim=0)
        batch_labels = torch.tensor([self.labels[i] for i in idx], dtype=torch.long)
        return batch_images, batch_labels


class TwoCropTransform:
    def __init__(self, base_transform: Callable) -> None:
        self.base_transform = base_transform

    def __call__(self, x):
        return self.base_transform(x), self.base_transform(x)


class ContrastiveTaskDataset(Dataset):
    def __init__(
        self, task_dataset: Dataset, two_crop_transform: TwoCropTransform
    ) -> None:
        self.task_dataset = task_dataset
        self.two_crop_transform = two_crop_transform

    def __len__(self) -> int:
        return len(self.task_dataset)

    def __getitem__(self, idx: int):
        image, label = self.task_dataset[idx]
        view1, view2 = self.two_crop_transform(image)
        return view1, view2, label


class LabelMapper:
    def __init__(self, task_classes: List[int]) -> None:
        self.mapping = {c: i for i, c in enumerate(task_classes)}

    def __call__(self, y: int) -> int:
        return self.mapping[int(y)]


def build_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    supcon_train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    return eval_transform, supcon_train_transform
