from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models import ContinualClassifier
from ..train import _resolve_task_ids, evaluate_class_il, evaluate_task_il


def _distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
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
    criterion = nn.CrossEntropyLoss()

    history: Dict[str, List[float]] = {
        "task_id": [],
        "train_loss": [],
        "class_il": [],
        "task_il": [],
        "distill_loss": [],
    }

    seen_task_ids: List[int] = []
    teacher_model: Optional[nn.Module] = None

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
                    f"train_loss={avg_loss:.4f} | distill={avg_distill:.4f}"
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

        teacher_model = deepcopy(model).to(device).eval()
        for param in teacher_model.parameters():
            param.requires_grad = False

        if verbose:
            print(
                f"Task {task_id} done (LwF) | Class-IL={class_il_acc:.2f}% | "
                f"Task-IL={task_il_acc:.2f}%"
            )

    return model, history
