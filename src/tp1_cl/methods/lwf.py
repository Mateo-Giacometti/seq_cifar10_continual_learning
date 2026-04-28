from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models import ContinualClassifier
from ..train import (
    _resolve_task_ids,
    evaluate_class_il,
    evaluate_task_il,
    evaluate_taskwise_class_il,
    evaluate_taskwise_task_il,
)
from ._utils import append_baseline_row


def _distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """
    Compute the distillation loss.

    Parameters
    ----------
    student_logits : torch.Tensor
        The student logits.
    teacher_logits : torch.Tensor
        The teacher logits.
    temperature : float
        The temperature.

    Returns
    -------
    torch.Tensor
        The distillation loss.
    """
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (
        temperature**2
    )


def train_lwf(
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
    temperature: float = 2.0,
    alpha: float = 1.0,
    task_ids: Optional[List[int]] = None,
    initial_model: Optional[ContinualClassifier] = None,
    initial_seen_task_ids: Optional[List[int]] = None,
    initial_teacher_model: Optional[nn.Module] = None,
    baseline_payload: Optional[Dict[str, object]] = None,
    verbose: bool = True,
) -> Tuple[nn.Module, Dict[str, object]]:
    """
    Train the Learning without Forgetting (LwF) model.

    Parameters
    ----------
    backbone : nn.Module
        The backbone.
    feat_dim : int
        The feature dimension.
    train_loaders : Dict[int, DataLoader]
        The training loaders.
    test_loaders : Dict[int, DataLoader]
        The test loaders.
    task_classes : List[List[int]]
        The task classes.
    device : torch.device
        The device.
    num_classes : int
        The number of classes.
    epochs_per_task : int
        The number of epochs per task.
    lr : float
        The learning rate.
    momentum : float
        The momentum.
    weight_decay : float
        The weight decay.
    temperature : float
        The temperature.
    alpha : float
        The alpha.
    task_ids : Optional[List[int]]
        The task ids.
    initial_model : Optional[ContinualClassifier]
        The initial model.
    initial_seen_task_ids : Optional[List[int]]
        The initial seen task ids.
    initial_teacher_model : Optional[nn.Module]
        The initial teacher model.
    baseline_payload : Optional[Dict[str, object]]
        The baseline payload.
    verbose : bool
        Whether to print verbose output.

    Returns
    -------
    Tuple[nn.Module, Dict[str, object]]
        The trained model and the history.
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
    criterion = nn.CrossEntropyLoss()

    n_tasks = len(task_classes)
    history: Dict[str, object] = {
        "task_id": [],
        "train_loss": [],
        "class_il": [],
        "task_il": [],
        "distill_loss": [],
        "taskwise_class_il_matrix": [],
        "taskwise_task_il_matrix": [],
    }

    append_baseline_row(history, baseline_payload, extra_keys={"distill_loss": 0.0})

    seen_task_ids: List[int] = [] if initial_seen_task_ids is None else list(initial_seen_task_ids)

    if initial_teacher_model is not None:
        teacher_model: Optional[nn.Module] = deepcopy(initial_teacher_model).to(device)
    elif initial_model is not None and len(seen_task_ids) > 0:
        teacher_model = deepcopy(initial_model).to(device)
    else:
        teacher_model = None

    if teacher_model is not None:
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False

    for task_id in selected_task_ids:
        old_class_ids = [
            cls
            for prev_task_id in seen_task_ids
            for cls in task_classes[prev_task_id]
        ]
        old_class_idx = (
            torch.tensor(old_class_ids, device=device, dtype=torch.long)
            if len(old_class_ids) > 0
            else None
        )

        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

        avg_loss = 0.0
        avg_distill = 0.0
        for epoch in range(1, epochs_per_task + 1):
            model.train()
            running_loss = 0.0
            running_distill = 0.0

            for images, labels in train_loaders[task_id]:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                logits = model(images)
                ce_loss = criterion(logits, labels)

                distill_loss = torch.tensor(0.0, device=device)
                if teacher_model is not None and old_class_idx is not None:
                    with torch.no_grad():
                        teacher_logits = teacher_model(images)[:, old_class_idx]
                    student_logits = logits[:, old_class_idx]
                    distill_loss = _distillation_loss(
                        student_logits=student_logits,
                        teacher_logits=teacher_logits,
                        temperature=temperature,
                    )

                loss = ce_loss + alpha * distill_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_distill += distill_loss.item()

            avg_loss = running_loss / max(1, len(train_loaders[task_id]))
            avg_distill = running_distill / max(1, len(train_loaders[task_id]))

            if verbose:
                print(
                    f"Task {task_id} | Epoch {epoch:02d}/{epochs_per_task} | "
                    f"Loss = {avg_loss:.4f} | Distill = {avg_distill:.4f}"
                )

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
        history["distill_loss"].append(avg_distill)
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

        teacher_model = deepcopy(model).to(device).eval()
        for param in teacher_model.parameters():
            param.requires_grad = False

        if verbose:
            print(
                f"Task {task_id} done (LwF) | Class-IL = {class_il_acc:.2f}% | "
                f"Task-IL = {task_il_acc:.2f}%"
            )

    return model, history
