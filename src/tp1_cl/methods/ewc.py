from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Literal, Optional, Tuple

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
    evaluate_taskwise_class_il,
    evaluate_taskwise_task_il,
)
from ._utils import append_baseline_row


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
    fisher_loss_mode: Literal["ce", "nll_true", "nll_pred"] = "ce",
    task_ids: Optional[List[int]] = None,
    initial_model: Optional[ContinualClassifier] = None,
    initial_seen_task_ids: Optional[List[int]] = None,
    initial_ewc_terms: Optional[List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]] = None,
    baseline_payload: Optional[Dict[str, object]] = None,
    verbose: bool = True,
) -> Tuple[nn.Module, Dict[str, object]]:
    """Train with Elastic Weight Consolidation (Kirkpatrick et al. 2017).

    Note: ``ewc_lambda`` absorbs the 1/2 factor from the paper's formulation.
    Our ``ewc_lambda=5.0`` is equivalent to the paper's ``λ=10.0``.
    """
    selected_task_ids = _resolve_task_ids(train_loaders, task_ids)

    model = (
        deepcopy(initial_model).to(device)
        if initial_model is not None
        else ContinualClassifier(
            backbone=deepcopy(backbone),
            feat_dim=feat_dim,
            num_classes=num_classes,
        ).to(device)
    )

    ewc_terms: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]] = (
        [] if initial_ewc_terms is None else list(initial_ewc_terms)
    )
    n_tasks = len(task_classes)
    history: Dict[str, object] = {
        "task_id": [],
        "train_loss": [],
        "class_il": [],
        "task_il": [],
        "taskwise_class_il_matrix": [],
        "taskwise_task_il_matrix": [],
    }

    append_baseline_row(history, baseline_payload)

    seen_task_ids: List[int] = [] if initial_seen_task_ids is None else list(initial_seen_task_ids)
    for task_id in selected_task_ids:
        # Reset optimizer per task to avoid momentum carry-over from previous
        # task distributions.
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

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
            fisher_loss_mode=fisher_loss_mode,
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
        history["taskwise_class_il_matrix"].append(
            evaluate_taskwise_class_il(
                model=model,
                test_loaders=test_loaders,
                seen_task_ids=seen_task_ids,
                device=device,
                n_tasks=n_tasks,
            )
        )
        history["taskwise_task_il_matrix"].append(
            evaluate_taskwise_task_il(
                model=model,
                test_loaders=test_loaders,
                task_classes=task_classes,
                seen_task_ids=seen_task_ids,
                device=device,
                n_tasks=n_tasks,
            )
        )

        if verbose:
            print(
                f"Task {task_id} done (EWC) | Class-IL={class_il_acc:.2f}% | "
                f"Task-IL={task_il_acc:.2f}%"
            )

    return model, history
