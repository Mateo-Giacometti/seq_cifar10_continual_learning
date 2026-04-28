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
        The order of classes for each task. If None, canonical CIFAR-10 order is used.
    task_classes : Optional[List[List[int]]]
        Explicit class split per task. If provided, it takes precedence over class_order.
    shuffle_class_order : bool
        Whether to shuffle canonical class order when class_order is None.
    seed : int
        Seed used when shuffle_class_order is enabled.
    allow_partial_last_task : bool
        Allow the final task to contain fewer than classes_per_task classes.
    """

    n_tasks: int = 5
    classes_per_task: int = 2
    class_order: Optional[List[int]] = None
    task_classes: Optional[List[List[int]]] = None
    shuffle_class_order: bool = False
    seed: int = 42
    allow_partial_last_task: bool = False

    def __post_init__(self) -> None:
        if self.n_tasks <= 0:
            raise ValueError("n_tasks must be > 0")
        if self.classes_per_task <= 0:
            raise ValueError("classes_per_task must be > 0")

        if self.class_order is not None and self.task_classes is not None:
            raise ValueError("Use either class_order or task_classes, not both")

        if self.class_order is not None:
            if sorted(self.class_order) != list(range(10)):
                raise ValueError("class_order must be a permutation of [0..9]")

        if self.task_classes is not None:
            if len(self.task_classes) == 0:
                raise ValueError("task_classes must contain at least one task")

            self.n_tasks = len(self.task_classes)

            flat: List[int] = []
            for i, task in enumerate(self.task_classes):
                if len(task) == 0:
                    raise ValueError(f"task_classes[{i}] cannot be empty")

                if not self.allow_partial_last_task or i < self.n_tasks - 1:
                    if len(task) != self.classes_per_task:
                        raise ValueError(
                            "All tasks must have classes_per_task classes unless allow_partial_last_task is True"
                        )
                else:
                    if len(task) > self.classes_per_task:
                        raise ValueError(
                            "Last task cannot have more than classes_per_task classes"
                        )

                for cls_id in task:
                    if cls_id < 0 or cls_id > 9:
                        raise ValueError("class ids must be in [0..9] for CIFAR-10")

                flat.extend(task)

            if len(set(flat)) != len(flat):
                raise ValueError(
                    "task_classes cannot contain duplicated classes across tasks"
                )


class CIFAR10TaskDataset(Dataset):
    def __init__(
        self,
        base_dataset: datasets.CIFAR10,
        indices: List[int],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
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
        """
        Returns the number of samples in the task.
        """
        return len(self.indices)

    def __getitem__(self, idx: int):
        """
        Returns the image and label for the given index.
        """
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

        self.class_order, self.task_classes = self._resolve_split()

        self.train_base = datasets.CIFAR10(
            root=self.root, train=True, download=True, transform=None
        )
        self.test_base = datasets.CIFAR10(
            root=self.root, train=False, download=True, transform=None
        )

        self.train_indices_by_task = self._build_task_indices(self.train_base.targets)
        self.test_indices_by_task = self._build_task_indices(self.test_base.targets)

    def _resolve_split(self) -> Tuple[List[int], List[List[int]]]:
        """
        Resolve the class split for the tasks.

        Returns
        -------
        class_order : List[int]
            The order of classes for each task.
        task_classes : List[List[int]]
            The class split for each task.
        """
        if self.config.task_classes is not None:
            task_classes = [list(task) for task in self.config.task_classes]
            class_order = [cls_id for task in task_classes for cls_id in task]
            return class_order, task_classes

        if self.config.class_order is not None:
            class_order = list(self.config.class_order)
        else:
            class_order = list(range(10))
            if self.config.shuffle_class_order:
                rng = random.Random(self.config.seed)
                rng.shuffle(class_order)

        required = self.config.n_tasks * self.config.classes_per_task
        if required > len(class_order) and not self.config.allow_partial_last_task:
            raise ValueError(
                "Requested tasks exceed available classes. "
                "Reduce n_tasks/classes_per_task or enable allow_partial_last_task."
            )

        task_classes: List[List[int]] = []
        for i in range(self.config.n_tasks):
            start = i * self.config.classes_per_task
            end = (i + 1) * self.config.classes_per_task
            classes = class_order[start:end]

            if len(classes) == self.config.classes_per_task:
                task_classes.append(classes)
                continue

            is_last_task = i == self.config.n_tasks - 1
            if (
                self.config.allow_partial_last_task
                and is_last_task
                and len(classes) > 0
            ):
                task_classes.append(classes)
                continue

            raise ValueError(
                "Could not build a valid task split with current configuration."
            )

        return class_order, task_classes

    def _build_task_indices(self, targets: List[int]) -> Dict[int, List[int]]:
        """
        Build the indices for each task.

        Parameters
        ----------
        targets : List[int]
            The labels for each sample.

        Returns
        -------
        task_indices : Dict[int, List[int]]
            The indices for each task.
        """
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
        """
        Get a task dataset.

        Parameters
        ----------
        task_id : int
            The ID of the task.
        train : bool
            Whether to get the training dataset.
        transform : Optional[Callable]
            The transformation to apply to the images.
        target_transform : Optional[Callable]
            The transformation to apply to the labels.

        Returns
        -------
        Dataset
            The task dataset.
        """
        base = self.train_base if train else self.test_base
        indices = (
            self.train_indices_by_task[task_id]
            if train
            else self.test_indices_by_task[task_id]
        )
        return CIFAR10TaskDataset(base, indices, transform=transform, target_transform=target_transform)

    def build_task_loaders(
        self,
        batch_size: int,
        transform: Optional[Callable],
        train_shuffle: bool = True,
        num_workers: int = 0,
        test_transform: Optional[Callable] = None,
        train_target_transform: Optional[Callable] = None,
        test_target_transform: Optional[Callable] = None,
        loader_kwargs: Optional[Dict[str, object]] = None,
    ) -> Tuple[Dict[int, DataLoader], Dict[int, DataLoader]]:
        """
        Build task loaders.

        Parameters
        ----------
        batch_size : int
            The batch size.
        transform : Optional[Callable]
            The transformation to apply to the images.
        train_shuffle : bool
            Whether to shuffle the training data.
        num_workers : int
            The number of worker processes.
        test_transform : Optional[Callable]
            The transformation to apply to the test images.
        train_target_transform : Optional[Callable]
            The transformation to apply to the training labels.
        test_target_transform : Optional[Callable]
            The transformation to apply to the test labels.
        loader_kwargs : Optional[Dict[str, object]]
            Additional arguments for the loader.

        Returns
        -------
        Tuple[Dict[int, DataLoader], Dict[int, DataLoader]]
            The training and test loaders.
        """
        train_loaders: Dict[int, DataLoader] = {}
        test_loaders: Dict[int, DataLoader] = {}

        resolved_loader_kwargs = {
            "num_workers": num_workers,
            "pin_memory": torch.cuda.is_available(),
            "persistent_workers": num_workers > 0,
        }
        if num_workers > 0:
            resolved_loader_kwargs["prefetch_factor"] = 4
        if loader_kwargs is not None:
            resolved_loader_kwargs.update(loader_kwargs)

        resolved_test_transform = (
            transform if test_transform is None else test_transform
        )

        for task_id in range(len(self.task_classes)):
            train_ds = self.get_task_dataset(
                task_id,
                train=True,
                transform=transform,
                target_transform=train_target_transform,
            )
            test_ds = self.get_task_dataset(
                task_id,
                train=False,
                transform=resolved_test_transform,
                target_transform=test_target_transform,
            )
            train_loaders[task_id] = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=train_shuffle,
                **resolved_loader_kwargs,
            )
            test_loaders[task_id] = DataLoader(
                test_ds,
                batch_size=batch_size,
                shuffle=False,
                **resolved_loader_kwargs,
            )
        return train_loaders, test_loaders

    def summary_rows(self) -> List[Dict[str, object]]:
        """
        Get a summary of the task datasets.

        Returns
        -------
        List[Dict[str, object]]
            A list of dictionaries containing the summary of the task datasets.
        """
        rows: List[Dict[str, object]] = []
        for task_id, classes in enumerate(self.task_classes):
            rows.append(
                {
                    "task_id": task_id,
                    "classes": classes,
                    "class_names": [
                        (
                            CIFAR10_CLASSES[c]
                            if 0 <= c < len(CIFAR10_CLASSES)
                            else f"class_{c}"
                        )
                        for c in classes
                    ],
                    "n_train": len(self.train_indices_by_task[task_id]),
                    "n_test": len(self.test_indices_by_task[task_id]),
                }
            )
        return rows


class ReservoirReplayBuffer:
    def __init__(self, capacity: int, seed: int = 42) -> None:
        """
        Initialize the replay buffer.

        Attributes
        ----------
        capacity : int
            The capacity of the buffer.
        seed : int
            The seed for the random number generator.
        images : List[torch.Tensor]
            The images in the buffer.
        labels : List[int]
            The labels in the buffer.
        seen : int
            The number of samples seen so far.
        rng : random.Random
            The random number generator.
        """
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.capacity = capacity
        self.images: List[torch.Tensor] = []
        self.labels: List[int] = []
        self.seen = 0
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        """
        Returns the number of samples in the buffer.
        """
        return len(self.images)

    def add(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Add samples to the buffer.

        Parameters
        ----------
        images : torch.Tensor
            The images to add to the buffer.
        labels : torch.Tensor
            The labels to add to the buffer.
        """
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
        """
        Sample a batch of images and labels from the buffer.

        Parameters
        ----------
        batch_size : int
            The number of samples to sample.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing the sampled images and labels.
        """
        if len(self.images) == 0:
            raise ValueError("Cannot sample from an empty replay buffer")
        k = min(batch_size, len(self.images))
        idx = self.rng.sample(range(len(self.images)), k=k)
        batch_images = torch.stack([self.images[i] for i in idx], dim=0)
        batch_labels = torch.tensor([self.labels[i] for i in idx], dtype=torch.long)
        return batch_images, batch_labels


class TwoCropTransform:
    def __init__(self, base_transform: Callable) -> None:
        """
        Initialize the two-crop transform.

        Parameters
        ----------
        base_transform : Callable
            The base transformation to apply to the images.
        """
        self.base_transform = base_transform

    def __call__(self, x):
        """
        Apply the two-crop transform to the given image.

        Parameters
        ----------
        x : torch.Tensor
            The image to apply the transform to.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing the two transformed images.
        """
        return self.base_transform(x), self.base_transform(x)


class ContrastiveTaskDataset(Dataset):
    def __init__(
        self, task_dataset: Dataset, two_crop_transform: TwoCropTransform
    ) -> None:
        """
        Initialize the contrastive task dataset.

        Parameters
        ----------
        task_dataset : Dataset
            The task dataset to apply the two-crop transform to.
        two_crop_transform : TwoCropTransform
            The two-crop transform to apply to the images.
        """
        self.task_dataset = task_dataset
        self.two_crop_transform = two_crop_transform

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(self.task_dataset)

    def __getitem__(self, idx: int):
        """
        Get the item at the given index.

        Parameters
        ----------
        idx : int
            The index of the item to get.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, int]
            A tuple containing the two transformed images and the label.
        """
        image, label = self.task_dataset[idx]
        view1, view2 = self.two_crop_transform(image)
        return view1, view2, label


class LabelMapper:
    def __init__(self, task_classes: List[int]) -> None:
        """
        Initialize the label mapper.

        Parameters
        ----------
        task_classes : List[int]
            The task classes to map the labels to.
        """
        self.mapping = {c: i for i, c in enumerate(task_classes)}

    def __call__(self, y: int) -> int:
        """
        Apply the label mapping to the given label.

        Parameters
        ----------
        y : int
            The label to apply the mapping to.

        Returns
        -------
        int
            The mapped label.
        """
        return self.mapping[int(y)]


@dataclass
class TransformConfig:
    """
    Configuration for evaluation and SupCon augmentation pipelines.

    Parameters
    ----------
    mean : Tuple[float, float, float]
        The mean of the images.
    std : Tuple[float, float, float]
        The standard deviation of the images.
    crop_size : int
        The size of the crops.
    crop_scale : Tuple[float, float]
        The scale of the crops.
    hflip_prob : float
        The probability of horizontal flip.
    color_jitter_values : Tuple[float, float, float, float]
        The values of the color jitter.
    color_jitter_prob : float
        The probability of color jitter.
    grayscale_prob : float
        The probability of grayscale.
    """
    mean: Tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
    std: Tuple[float, float, float] = (0.2470, 0.2435, 0.2616)
    crop_size: int = 32
    crop_scale: Tuple[float, float] = (0.2, 1.0)
    hflip_prob: float = 0.5
    color_jitter_values: Tuple[float, float, float, float] = (0.4, 0.4, 0.4, 0.1)
    color_jitter_prob: float = 0.8
    grayscale_prob: float = 0.2


def build_transforms(
    config: Optional[TransformConfig] = None,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Build the transformation pipelines for evaluation and SupCon training.

    Parameters
    ----------
    config : Optional[TransformConfig]
        The configuration for the transformations.

    Returns
    -------
    Tuple[transforms.Compose, transforms.Compose]
        A tuple containing the evaluation and SupCon training transformation pipelines.
    """
    cfg = TransformConfig() if config is None else config

    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(cfg.mean, cfg.std),
        ]
    )

    supcon_train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=cfg.crop_size, scale=cfg.crop_scale),
            transforms.RandomHorizontalFlip(p=cfg.hflip_prob),
            transforms.RandomApply(
                [transforms.ColorJitter(*cfg.color_jitter_values)],
                p=cfg.color_jitter_prob,
            ),
            transforms.RandomGrayscale(p=cfg.grayscale_prob),
            transforms.ToTensor(),
            transforms.Normalize(cfg.mean, cfg.std),
        ]
    )

    return eval_transform, supcon_train_transform
