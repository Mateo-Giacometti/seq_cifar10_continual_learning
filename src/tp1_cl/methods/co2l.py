from __future__ import annotations

import math
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ..data import ReservoirReplayBuffer
from ..models import AsymmetricSupConLoss, ContinualClassifier, IRDLoss, SupConNetwork
from ..train import _resolve_task_ids, evaluate_class_il, evaluate_task_il


class _TensorDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return self.labels.size(0)

    def __getitem__(self, idx: int):
        return self.images[idx], self.labels[idx]


def _make_co2l_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        if epochs == warmup_epochs:
            return 1.0
        progress = float(epoch - warmup_epochs) / float(epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _train_linear_head_balanced(
    backbone: nn.Module,
    feat_dim: int,
    train_loader: DataLoader,
    test_loaders: Dict[int, DataLoader],
    task_classes: List[List[int]],
    seen_task_ids: List[int],
    device: torch.device,
    num_classes: int,
    epochs: int,
    lr: float,
    milestones: List[int],
    gamma: float,
) -> ContinualClassifier:
    model = ContinualClassifier(
        backbone=deepcopy(backbone),
        feat_dim=feat_dim,
        num_classes=num_classes,
    ).to(device)
    model.backbone.eval()
    for p in model.backbone.parameters():
        p.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.classifier.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=0.0,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=milestones,
        gamma=gamma,
    )

    for _ in range(epochs):
        model.classifier.train()
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.no_grad():
                feats = model.backbone(images)
            logits = model.classifier(feats)
            loss = criterion(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()

    class_il = evaluate_class_il(model, test_loaders, seen_task_ids, device)
    task_il = evaluate_task_il(model, test_loaders, task_classes, seen_task_ids, device)
    model.classifier.eval()
    model.class_il_metric = class_il  # type: ignore[attr-defined]
    model.task_il_metric = task_il  # type: ignore[attr-defined]
    return model


def train_co2l(
    backbone: nn.Module,
    feat_dim: int,
    train_loaders: Dict[int, DataLoader],
    test_loaders: Dict[int, DataLoader],
    task_classes: List[List[int]],
    device: torch.device,
    num_classes: int = 10,
    proj_dim: int = 128,
    buffer_size: int = 200,
    lr: float = 0.5,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    temperature: float = 0.5,
    kappa: float = 0.2,
    kappa_star: float = 0.01,
    lambda_ird: float = 1.0,
    warmup_epochs: int = 10,
    epochs_task0: int = 500,
    epochs_per_task: int = 100,
    eval_linear_epochs: int = 100,
    eval_linear_lr: float = 1.0,
    eval_linear_milestones: Optional[List[int]] = None,
    eval_linear_gamma: float = 0.2,
    task_ids: Optional[List[int]] = None,
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[SupConNetwork, Dict[str, List[float]]]:
    selected_task_ids = _resolve_task_ids(train_loaders, task_ids)
    model = SupConNetwork(
        backbone=deepcopy(backbone),
        feat_dim=feat_dim,
        proj_dim=proj_dim,
    ).to(device)

    asym_loss = AsymmetricSupConLoss(temperature=temperature)
    ird_loss = IRDLoss(kappa=kappa, kappa_star=kappa_star)
    buffer = ReservoirReplayBuffer(capacity=buffer_size, seed=seed)

    history: Dict[str, List[float]] = {
        "task_id": [],
        "train_loss": [],
        "class_il": [],
        "task_il": [],
        "ird_loss": [],
    }

    seen_task_ids: List[int] = []
    past_model: Optional[SupConNetwork] = None

    for task_id in selected_task_ids:
        epochs = epochs_task0 if task_id == selected_task_ids[0] else epochs_per_task
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        scheduler = _make_co2l_scheduler(optimizer, epochs=epochs, warmup_epochs=warmup_epochs)

        avg_epoch_loss = 0.0
        avg_epoch_ird = 0.0

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            running_ird = 0.0

            for images_curr, labels_curr in train_loaders[task_id]:
                images_curr = images_curr.to(device, non_blocking=True)
                labels_curr = labels_curr.to(device, non_blocking=True)

                if len(buffer) > 0:
                    images_buf, labels_buf = buffer.sample(images_curr.size(0))
                    images_buf = images_buf.to(device, non_blocking=True)
                    labels_buf = labels_buf.to(device, non_blocking=True)
                    images_all = torch.cat([images_curr, images_buf], dim=0)
                    labels_all = torch.cat([labels_curr, labels_buf], dim=0)
                    current_mask = torch.zeros(images_all.size(0), dtype=torch.bool, device=device)
                    current_mask[: images_curr.size(0)] = True
                else:
                    images_all = images_curr
                    labels_all = labels_curr
                    current_mask = torch.ones(images_all.size(0), dtype=torch.bool, device=device)

                noise1 = 0.001 * torch.randn_like(images_all)
                noise2 = 0.001 * torch.randn_like(images_all)
                view1 = images_all + noise1
                view2 = images_all + noise2

                batch_images = torch.cat([view1, view2], dim=0)
                _, proj_all = model(batch_images)
                bsz_total = labels_all.size(0)
                proj_views = proj_all.view(2, bsz_total, -1).permute(1, 0, 2).contiguous()

                loss_sup = asym_loss(proj_views, labels_all, current_mask)
                loss = loss_sup
                loss_ird_batch = torch.tensor(0.0, device=device)

                if past_model is not None:
                    with torch.no_grad():
                        _, proj_past = past_model(batch_images)
                    loss_ird_batch = ird_loss(proj_all, proj_past)
                    loss = loss + lambda_ird * loss_ird_batch

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                running_ird += loss_ird_batch.item()

            scheduler.step()
            avg_epoch_loss = running_loss / max(1, len(train_loaders[task_id]))
            avg_epoch_ird = running_ird / max(1, len(train_loaders[task_id]))

            if verbose and (epoch % 10 == 0 or epoch == epochs):
                print(
                    f"Co2L | Task {task_id} | Epoch {epoch:03d}/{epochs} | "
                    f"loss={avg_epoch_loss:.4f} | ird={avg_epoch_ird:.4f}"
                )

        for images, labels in train_loaders[task_id]:
            buffer.add(images, labels)

        seen_task_ids.append(task_id)

        if len(buffer) > 0:
            all_images = torch.stack(buffer.images, dim=0)
            all_labels = torch.tensor(buffer.labels, dtype=torch.long)
            eval_ds = _TensorDataset(all_images, all_labels)
            eval_loader = DataLoader(
                eval_ds,
                batch_size=train_loaders[task_id].batch_size or 256,
                shuffle=True,
            )
        else:
            eval_loader = train_loaders[task_id]

        if eval_linear_milestones is None:
            eval_linear_milestones = [60, 75, 90]

        eval_model = _train_linear_head_balanced(
            backbone=model.backbone,
            feat_dim=feat_dim,
            train_loader=eval_loader,
            test_loaders=test_loaders,
            task_classes=task_classes,
            seen_task_ids=seen_task_ids,
            device=device,
            num_classes=num_classes,
            epochs=eval_linear_epochs,
            lr=eval_linear_lr,
            milestones=eval_linear_milestones,
            gamma=eval_linear_gamma,
        )
        class_il_acc = float(eval_model.class_il_metric)  # type: ignore[attr-defined]
        task_il_acc = float(eval_model.task_il_metric)  # type: ignore[attr-defined]

        history["task_id"].append(float(task_id))
        history["train_loss"].append(avg_epoch_loss)
        history["class_il"].append(class_il_acc)
        history["task_il"].append(task_il_acc)
        history["ird_loss"].append(avg_epoch_ird)

        past_model = deepcopy(model).to(device).eval()
        for p in past_model.parameters():
            p.requires_grad = False

        if verbose:
            print(
                f"Task {task_id} done (Co2L) | Class-IL={class_il_acc:.2f}% | "
                f"Task-IL={task_il_acc:.2f}%"
            )

    return model, history
