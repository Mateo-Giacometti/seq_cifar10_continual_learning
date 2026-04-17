"""Utilities for TP1 Continual Learning sections 4.1 to 4.3."""

from .checkpoints import ensure_checkpoint_dir, load_checkpoint, save_checkpoint
from .config import ConfigNode, load_config
from .bootstrap import (
    build_initial_classifier_from_linear_head,
    build_task0_ewc_terms,
    build_task0_replay_buffer,
    task0_baseline_payload,
)
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
from .methods import train_co2l as train_co2l_method
from .methods import train_er_ace as train_er_ace_method
from .methods import train_ewc as train_ewc_method
from .methods import train_lwf as train_lwf_method
from .methods import train_naive_finetuning as train_naive_finetuning_method
from .models import (
    ContinualClassifier,
    SupConLoss,
    SupConNetwork,
    AsymmetricSupConLoss,
    IRDLoss,
    build_cifar_resnet,
    build_cifar_resnet18,
)
from .train import (
    evaluate_class_il,
    evaluate_linear,
    evaluate_task_il,
    evaluate_taskwise_class_il,
    evaluate_taskwise_task_il,
    extract_features,
    set_seed,
    train_ewc,
    train_linear_head,
    train_naive_finetuning,
    train_supcon,
)
from .viz import (
    compute_forgetting_matrix,
    plot_cl_metrics,
    plot_embedding_snapshots,
    plot_forgetting_by_task,
    plot_forgetting_curves,
    plot_linear_history,
    plot_methods_over_tasks,
    plot_supcon_loss,
)

__all__ = [
    "ConfigNode",
    "load_config",
    "ensure_checkpoint_dir",
    "load_checkpoint",
    "save_checkpoint",
    "build_task0_replay_buffer",
    "build_task0_ewc_terms",
    "build_initial_classifier_from_linear_head",
    "task0_baseline_payload",
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
    "AsymmetricSupConLoss",
    "IRDLoss",
    "build_cifar_resnet",
    "build_cifar_resnet18",
    "evaluate_class_il",
    "evaluate_linear",
    "evaluate_task_il",
    "evaluate_taskwise_class_il",
    "evaluate_taskwise_task_il",
    "extract_features",
    "set_seed",
    "train_ewc",
    "train_co2l_method",
    "train_er_ace_method",
    "train_ewc_method",
    "train_lwf_method",
    "train_linear_head",
    "train_naive_finetuning",
    "train_naive_finetuning_method",
    "train_supcon",
    "plot_supcon_loss",
    "plot_embedding_snapshots",
    "plot_linear_history",
    "plot_cl_metrics",
    "plot_methods_over_tasks",
    "plot_forgetting_curves",
    "plot_forgetting_by_task",
    "compute_forgetting_matrix",
]
