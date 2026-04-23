from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn

from .data import ReservoirReplayBuffer
from .models import ContinualClassifier
from .train import _compute_fisher_diagonal, _snapshot_parameters


def build_task0_replay_buffer(
    train_loader: torch.utils.data.DataLoader,
    capacity: int,
    seed: int,
) -> ReservoirReplayBuffer:
    """Build a replay buffer populated with Task 0 samples."""
    buffer = ReservoirReplayBuffer(capacity=capacity, seed=seed)
    for batch in train_loader:
        if len(batch) == 2:
            images, labels = batch
        elif len(batch) == 3:
            images, _, labels = batch
        else:
            raise ValueError("Task 0 loader batch must be (img,label) or (view1,view2,label)")
        buffer.add(images, labels)
    return buffer


def build_initial_classifier_from_linear_head(
    backbone: nn.Module,
    feat_dim: int,
    num_classes: int,
    linear_head: nn.Linear,
    task0_classes: List[int],
    device: torch.device,
) -> ContinualClassifier:
    """Create a global classifier initialized from Task 0 linear head.

    The two-class linear head from section 4.2 is mapped to global class
    indices in `task0_classes`. Remaining classes are initialized to zero.
    """
    model = ContinualClassifier(
        backbone=deepcopy(backbone),
        feat_dim=feat_dim,
        num_classes=num_classes,
    ).to(device)

    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.zero_()
        for local_idx, global_cls in enumerate(task0_classes):
            model.classifier.weight[global_cls].copy_(linear_head.weight[local_idx])
            model.classifier.bias[global_cls].copy_(linear_head.bias[local_idx])

    return model


def task0_baseline_payload(
    task0_acc: float,
    n_tasks: int,
    task0_id: int = 0,
) -> Dict[str, object]:
    """Create baseline metrics entry for histories before CL tasks start."""
    row_class = [float("nan")] * n_tasks
    row_task = [float("nan")] * n_tasks
    row_class[task0_id] = task0_acc
    row_task[task0_id] = task0_acc
    return {
        "task_id": float(task0_id),
        "train_loss": float("nan"),
        "class_il": task0_acc,
        "task_il": task0_acc,
        "taskwise_class_il_row": row_class,
        "taskwise_task_il_row": row_task,
    }


def build_task0_ewc_terms(
    model: nn.Module,
    train_loader_task0: torch.utils.data.DataLoader,
    device: torch.device,
    fisher_max_batches: Optional[int] = None,
    fisher_loss_mode: Literal["ce", "nll_true", "nll_pred"] = "ce",
) -> List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]:
    fisher_diag = _compute_fisher_diagonal(
        model=model,
        loader=train_loader_task0,
        device=device,
        max_batches=fisher_max_batches,
        fisher_loss_mode=fisher_loss_mode,
    )
    params_star = _snapshot_parameters(model)
    return [(fisher_diag, params_star)]
