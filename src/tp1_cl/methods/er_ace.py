from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data import ReservoirReplayBuffer
from ..models import ContinualClassifier
from ..train import (
    _resolve_task_ids,
    evaluate_class_il,
    evaluate_task_il,
    evaluate_taskwise_class_il,
    evaluate_taskwise_task_il,
)


def _cross_entropy_on_class_subset(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_ids: torch.Tensor,
) -> torch.Tensor:
    if class_ids.numel() == 0:
        return torch.tensor(0.0, device=logits.device)

    selected_logits = logits.index_select(dim=1, index=class_ids)
    matches = labels.unsqueeze(1).eq(class_ids.unsqueeze(0))
    if not torch.all(matches.any(dim=1)):
        raise ValueError("Some labels are not present in provided class subset")

    local_targets = matches.float().argmax(dim=1)
    return F.cross_entropy(selected_logits, local_targets)


def train_er_ace(
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
    buffer_size: int = 200,
    replay_batch_size: int = 128,
    incoming_weight: float = 1.0,
    replay_weight: float = 1.0,
    task_ids: Optional[List[int]] = None,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[nn.Module, Dict[str, object]]:
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

    buffer = ReservoirReplayBuffer(capacity=buffer_size, seed=seed)
    n_tasks = len(task_classes)
    history: Dict[str, object] = {
        "task_id": [],
        "train_loss": [],
        "incoming_loss": [],
        "replay_loss": [],
        "class_il": [],
        "task_il": [],
        "taskwise_class_il_matrix": [],
        "taskwise_task_il_matrix": [],
    }

    seen_task_ids: List[int] = []

    for task_id in selected_task_ids:
        old_class_ids = sorted(
            {
                cls
                for prev_task_id in seen_task_ids
                for cls in task_classes[prev_task_id]
            }
        )
        old_class_tensor = torch.tensor(old_class_ids, device=device, dtype=torch.long)

        avg_total_loss = 0.0
        avg_incoming_loss = 0.0
        avg_replay_loss = 0.0

        for epoch in range(1, epochs_per_task + 1):
            model.train()
            running_total_loss = 0.0
            running_incoming_loss = 0.0
            running_replay_loss = 0.0

            for images_curr, labels_curr in train_loaders[task_id]:
                images_curr = images_curr.to(device, non_blocking=True)
                labels_curr = labels_curr.to(device, non_blocking=True)

                logits_curr = model(images_curr)
                curr_class_tensor = labels_curr.unique(sorted=True)
                loss_incoming = _cross_entropy_on_class_subset(
                    logits_curr,
                    labels_curr,
                    curr_class_tensor,
                )

                loss_replay = torch.tensor(0.0, device=device)
                if len(buffer) > 0 and replay_batch_size > 0:
                    images_bf, labels_bf = buffer.sample(replay_batch_size)
                    images_bf = images_bf.to(device, non_blocking=True)
                    labels_bf = labels_bf.to(device, non_blocking=True)

                    replay_class_tensor = torch.unique(
                        torch.cat([old_class_tensor, curr_class_tensor], dim=0),
                        sorted=True,
                    )
                    logits_bf = model(images_bf)
                    loss_replay = _cross_entropy_on_class_subset(
                        logits_bf,
                        labels_bf,
                        replay_class_tensor,
                    )

                loss = incoming_weight * loss_incoming + replay_weight * loss_replay

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                running_total_loss += loss.item()
                running_incoming_loss += loss_incoming.item()
                running_replay_loss += loss_replay.item()

            avg_total_loss = running_total_loss / max(1, len(train_loaders[task_id]))
            avg_incoming_loss = running_incoming_loss / max(1, len(train_loaders[task_id]))
            avg_replay_loss = running_replay_loss / max(1, len(train_loaders[task_id]))

            if verbose:
                print(
                    f"ER-ACE | Task {task_id} | Epoch {epoch:02d}/{epochs_per_task} | "
                    f"loss={avg_total_loss:.4f} | incoming={avg_incoming_loss:.4f} | "
                    f"replay={avg_replay_loss:.4f}"
                )

        for images, labels in train_loaders[task_id]:
            buffer.add(images, labels)

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
        history["train_loss"].append(avg_total_loss)
        history["incoming_loss"].append(avg_incoming_loss)
        history["replay_loss"].append(avg_replay_loss)
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
                f"Task {task_id} done (ER-ACE) | Class-IL={class_il_acc:.2f}% | "
                f"Task-IL={task_il_acc:.2f}%"
            )

    return model, history
