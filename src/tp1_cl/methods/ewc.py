from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ..models import ContinualClassifier
from ..train import (
    _compute_fisher_diagonal,
    _resolve_task_ids,
    _snapshot_parameters,
    _train_task_classifier,
    evaluate_class_il,
    evaluate_task_il,
)


def train_ewc(
    backbone: nn.Module,
    feat_dim: int,
    train_loaders: Dict[int, torch.utils.data.DataLoader],
    test_loaders: Dict[int, torch.utils.data.DataLoader],
    task_classes: List[List[int]],
    device: torch.device,
    num_classes: int = 10,
    epochs_per_task: int = 10,
    lr: float = 0.03,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    ewc_lambda: float = 10.0,
    fisher_max_batches: Optional[int] = None,
    task_ids: Optional[List[int]] = None,
    verbose: bool = True,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    selected_task_ids = _resolve_task_ids(train_loaders, task_ids)

    model = ContinualClassifier(
        backbone=deepcopy(backbone),
        feat_dim=feat_dim,
        num_classes=num_classes,
    ).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    ewc_terms: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]] = []
    history: Dict[str, List[float]] = {
        "task_id": [],
        "train_loss": [],
        "class_il": [],
        "task_il": [],
    }

    seen_task_ids: List[int] = []
    for task_id in selected_task_ids:
        avg_loss = _train_task_classifier(
            model=model,
            train_loader=train_loaders[task_id],
            optimizer=optimizer,
            device=device,
            epochs=epochs_per_task,
            task_id=task_id,
            verbose=verbose,
            ewc_terms=ewc_terms,
            ewc_lambda=ewc_lambda,
        )

        fisher_diag = _compute_fisher_diagonal(
            model,
            train_loaders[task_id],
            device=device,
            max_batches=fisher_max_batches,
        )
        params_star = _snapshot_parameters(model)
        ewc_terms.append((fisher_diag, params_star))

        seen_task_ids.append(task_id)
        class_il_acc = evaluate_class_il(model, test_loaders, seen_task_ids, device)
        task_il_acc = evaluate_task_il(
            model,
            test_loaders,
            task_classes,
            seen_task_ids,
            device,
        )

        history["task_id"].append(float(task_id))
        history["train_loss"].append(avg_loss)
        history["class_il"].append(class_il_acc)
        history["task_il"].append(task_il_acc)

        if verbose:
            print(
                f"Task {task_id} done (EWC) | Class-IL={class_il_acc:.2f}% | "
                f"Task-IL={task_il_acc:.2f}%"
            )

    return model, history
