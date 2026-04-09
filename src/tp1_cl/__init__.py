"""Utilities for TP1 Continual Learning sections 4.1 to 4.3."""

from .data import (
    CIFAR10_CLASSES,
    CIFAR10TaskDataset,
    ContrastiveTaskDataset,
    LabelMapper,
    ReservoirReplayBuffer,
    SeqCIFAR10Builder,
    TaskConfig,
    TransformConfig,
    TwoCropTransform,
    build_transforms,
)
from .models import (
    ContinualClassifier,
    SupConLoss,
    SupConNetwork,
    build_cifar_resnet,
    build_cifar_resnet18,
)
from .train import (
    evaluate_class_il,
    evaluate_linear,
    evaluate_task_il,
    extract_features,
    set_seed,
    train_ewc,
    train_linear_head,
    train_naive_finetuning,
    train_supcon,
)

__all__ = [
    "CIFAR10_CLASSES",
    "CIFAR10TaskDataset",
    "ContrastiveTaskDataset",
    "LabelMapper",
    "ReservoirReplayBuffer",
    "SeqCIFAR10Builder",
    "TaskConfig",
    "TransformConfig",
    "TwoCropTransform",
    "build_transforms",
    "ContinualClassifier",
    "SupConLoss",
    "SupConNetwork",
    "build_cifar_resnet",
    "build_cifar_resnet18",
    "evaluate_class_il",
    "evaluate_linear",
    "evaluate_task_il",
    "extract_features",
    "set_seed",
    "train_ewc",
    "train_linear_head",
    "train_naive_finetuning",
    "train_supcon",
]
