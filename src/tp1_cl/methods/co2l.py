from __future__ import annotations

import math
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from ..data import ReservoirReplayBuffer
from ..models import AsymmetricSupConLoss, ContinualClassifier, IRDLoss, SupConNetwork
from ..train import (
    _resolve_task_ids,
    evaluate_class_il,
    evaluate_task_il,
    evaluate_taskwise_class_il,
    evaluate_taskwise_task_il,
)


def _append_baseline_row(history: Dict[str, object], baseline: Optional[Dict[str, object]]) -> None:
    if baseline is None:
        return
    history["task_id"].append(float(baseline["task_id"]))
    history["train_loss"].append(float(baseline.get("train_loss", float("nan"))))
    history["class_il"].append(float(baseline["class_il"]))
    history["task_il"].append(float(baseline["task_il"]))
    history["ird_loss"].append(0.0)
    history["taskwise_class_il_matrix"].append(list(baseline["taskwise_class_il_row"]))
    history["taskwise_task_il_matrix"].append(list(baseline["taskwise_task_il_row"]))


def _clone_replay_buffer(source: ReservoirReplayBuffer, seed: int) -> ReservoirReplayBuffer:
    out = ReservoirReplayBuffer(capacity=source.capacity, seed=seed)
    out.images = [img.clone() for img in source.images]
    out.labels = list(source.labels)
    out.seen = source.seen
    return out


class _TensorDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return self.labels.size(0)

    def __getitem__(self, idx: int):
        return self.images[idx], self.labels[idx]


def _make_balanced_sampler(labels: torch.Tensor) -> WeightedRandomSampler:
    classes, counts = labels.unique(return_counts=True)
    weight_per_class = 1.0 / counts.float()
    weights = weight_per_class[
        (labels.unsqueeze(1) == classes.unsqueeze(0)).int().argmax(dim=1)
    ]
    return WeightedRandomSampler(
        weights=weights.tolist(),
        num_samples=len(weights),
        replacement=True,
    )


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


def _augment_buffer_batch(images: torch.Tensor) -> torch.Tensor:
    flip_mask = torch.rand(images.size(0), device=images.device) > 0.5
    out = images.clone()
    out[flip_mask] = torch.flip(images[flip_mask], dims=[-1])
    return out


def _train_linear_head_balanced(
    backbone: nn.Module,
    feat_dim: int,
    buffer_images: torch.Tensor,
    buffer_labels: torch.Tensor,
    test_loaders: Dict[int, DataLoader],
    task_classes: List[List[int]],
    seen_task_ids: List[int],
    device: torch.device,
    num_classes: int,
    epochs: int,
    lr: float,
    milestones: List[int],
    gamma: float,
    batch_size: int = 256,
) -> ContinualClassifier:
    model = ContinualClassifier(
        backbone=deepcopy(backbone),
        feat_dim=feat_dim,
        num_classes=num_classes,
    ).to(device)
    model.backbone.eval()
    for p in model.backbone.parameters():
        p.requires_grad = False

    eval_ds = _TensorDataset(buffer_images, buffer_labels)
    sampler = _make_balanced_sampler(buffer_labels)
    eval_loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
    )

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
        for images, labels in eval_loader:
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
    train_loaders_eval: Optional[Dict[int, DataLoader]] = None,
    task_ids: Optional[List[int]] = None,
    seed: int = 42,
    max_replay_batch_size: Optional[int] = None,
    initial_model: Optional[SupConNetwork] = None,
    initial_seen_task_ids: Optional[List[int]] = None,
    initial_past_model: Optional[SupConNetwork] = None,
    initial_buffer: Optional[ReservoirReplayBuffer] = None,
    baseline_payload: Optional[Dict[str, object]] = None,
    verbose: bool = True,
) -> Tuple[SupConNetwork, Dict[str, object]]:
    if eval_linear_milestones is None:
        eval_linear_milestones = [60, 75, 90]

    n_tasks = len(task_classes)
    selected_task_ids = _resolve_task_ids(train_loaders, task_ids)
    model = (
        deepcopy(initial_model).to(device)
        if initial_model is not None
        else SupConNetwork(
            backbone=deepcopy(backbone),
            feat_dim=feat_dim,
            proj_dim=proj_dim,
        ).to(device)
    )

    asym_loss = AsymmetricSupConLoss(temperature=temperature)
    ird_loss = IRDLoss(kappa=kappa, kappa_star=kappa_star)
    if initial_buffer is not None:
        buffer = _clone_replay_buffer(initial_buffer, seed=seed)
    else:
        buffer = ReservoirReplayBuffer(capacity=buffer_size, seed=seed)

    history: Dict[str, object] = {
        "task_id": [],
        "train_loss": [],
        "class_il": [],
        "task_il": [],
        "ird_loss": [],
        "taskwise_class_il_matrix": [],
        "taskwise_task_il_matrix": [],
    }

    _append_baseline_row(history, baseline_payload)

    seen_task_ids: List[int] = [] if initial_seen_task_ids is None else list(initial_seen_task_ids)
    past_model: Optional[SupConNetwork] = (
        None if initial_past_model is None else deepcopy(initial_past_model).to(device)
    )
    if past_model is not None:
        past_model.eval()
        for p in past_model.parameters():
            p.requires_grad = False

    for task_id in selected_task_ids:
        epochs = epochs_task0 if task_id == 0 else epochs_per_task
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        scheduler = _make_co2l_scheduler(
            optimizer,
            epochs=epochs,
            warmup_epochs=warmup_epochs,
        )

        epoch_losses: List[float] = []
        epoch_irds: List[float] = []

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            running_ird = 0.0

            for batch in train_loaders[task_id]:
                if len(batch) == 3:
                    view1_curr, view2_curr, labels_curr = batch
                    view1_curr = view1_curr.to(device, non_blocking=True)
                    view2_curr = view2_curr.to(device, non_blocking=True)
                    labels_curr = labels_curr.to(device, non_blocking=True)
                    bsz_curr = labels_curr.size(0)
                elif len(batch) == 2:
                    images_curr, labels_curr = batch
                    images_curr = images_curr.to(device, non_blocking=True)
                    labels_curr = labels_curr.to(device, non_blocking=True)
                    bsz_curr = labels_curr.size(0)
                    noise1 = 0.001 * torch.randn_like(images_curr)
                    noise2 = 0.001 * torch.randn_like(images_curr)
                    view1_curr = images_curr + noise1
                    view2_curr = images_curr + noise2
                else:
                    raise ValueError("train loader batch must be (img,label) or (view1,view2,label)")

                if len(buffer) > 0:
                    replay_batch = bsz_curr
                    if max_replay_batch_size is not None:
                        replay_batch = min(replay_batch, max_replay_batch_size)

                    images_buf, labels_buf = buffer.sample(replay_batch)
                    images_buf = images_buf.to(device, non_blocking=True)
                    labels_buf = labels_buf.to(device, non_blocking=True)

                    view1_buf = _augment_buffer_batch(images_buf)
                    view2_buf = _augment_buffer_batch(images_buf)

                    view1_all = torch.cat([view1_curr, view1_buf], dim=0)
                    view2_all = torch.cat([view2_curr, view2_buf], dim=0)
                    labels_all = torch.cat([labels_curr, labels_buf], dim=0)
                    current_mask = torch.zeros(
                        bsz_curr + replay_batch,
                        dtype=torch.bool,
                        device=device,
                    )
                    current_mask[:bsz_curr] = True
                else:
                    view1_all = view1_curr
                    view2_all = view2_curr
                    labels_all = labels_curr
                    current_mask = torch.ones(bsz_curr, dtype=torch.bool, device=device)

                batch_images = torch.cat([view1_all, view2_all], dim=0)
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
            n_batches = max(1, len(train_loaders[task_id]))
            epoch_loss = running_loss / n_batches
            epoch_ird = running_ird / n_batches
            epoch_losses.append(epoch_loss)
            epoch_irds.append(epoch_ird)

            if verbose and (epoch % 10 == 0 or epoch == epochs):
                print(
                    f"Co2L | Task {task_id} | Epoch {epoch:03d}/{epochs} | "
                    f"loss={epoch_loss:.4f} | ird={epoch_ird:.4f}"
                )

        past_model = deepcopy(model).to(device).eval()
        for p in past_model.parameters():
            p.requires_grad = False

        buffer_source = train_loaders_eval if train_loaders_eval is not None else train_loaders
        for images, labels in buffer_source[task_id]:
            buffer.add(images, labels)

        seen_task_ids.append(task_id)

        buffer_images = torch.stack(buffer.images, dim=0)
        buffer_labels = torch.tensor(buffer.labels, dtype=torch.long)
        eval_model = _train_linear_head_balanced(
            backbone=model.backbone,
            feat_dim=feat_dim,
            buffer_images=buffer_images,
            buffer_labels=buffer_labels,
            test_loaders=test_loaders,
            task_classes=task_classes,
            seen_task_ids=seen_task_ids,
            device=device,
            num_classes=num_classes,
            epochs=eval_linear_epochs,
            lr=eval_linear_lr,
            milestones=eval_linear_milestones,
            gamma=eval_linear_gamma,
            batch_size=256,
        )
        class_il_acc = float(eval_model.class_il_metric)  # type: ignore[attr-defined]
        task_il_acc = float(eval_model.task_il_metric)  # type: ignore[attr-defined]

        history["task_id"].append(float(task_id))
        history["train_loss"].append(sum(epoch_losses) / len(epoch_losses))
        history["class_il"].append(class_il_acc)
        history["task_il"].append(task_il_acc)
        history["ird_loss"].append(sum(epoch_irds) / len(epoch_irds))
        history["taskwise_class_il_matrix"].append(
            evaluate_taskwise_class_il(
                model=eval_model,
                test_loaders=test_loaders,
                seen_task_ids=seen_task_ids,
                device=device,
                n_tasks=n_tasks,
            )
        )
        history["taskwise_task_il_matrix"].append(
            evaluate_taskwise_task_il(
                model=eval_model,
                test_loaders=test_loaders,
                task_classes=task_classes,
                seen_task_ids=seen_task_ids,
                device=device,
                n_tasks=n_tasks,
            )
        )

        if verbose:
            print(
                f"Task {task_id} done (Co2L) | "
                f"Class-IL={class_il_acc:.2f}% | Task-IL={task_il_acc:.2f}%"
            )

    return model, history
