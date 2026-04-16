from copy import deepcopy
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .models import ContinualClassifier, SupConLoss, SupConNetwork


def set_seed(seed: int = 42) -> None:
    """
    Set the random seed for reproducibility.

    Parameters
    ----------
    seed : int
        The seed value to use for random number generation.

    Returns
    -------
    None

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


@torch.no_grad()
def extract_features(
    backbone: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_samples: int = 2000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    backbone.eval()
    feats_list = []
    labels_list = []
    collected = 0

    for images, labels in loader:
        images = images.to(device)
        feats = backbone(images).detach().cpu()
        feats_list.append(feats)
        labels_list.append(labels.cpu())
        collected += labels.size(0)
        if collected >= max_samples:
            break

    features = torch.cat(feats_list, dim=0)[:max_samples]
    out_labels = torch.cat(labels_list, dim=0)[:max_samples]
    return features, out_labels


def train_supcon(
    model: SupConNetwork,
    train_loader: torch.utils.data.DataLoader,
    eval_loader_for_snapshots: torch.utils.data.DataLoader,
    criterion: SupConLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    snapshot_epochs: Dict[int, str],
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
) -> Tuple[List[float], Dict[str, Tuple[torch.Tensor, torch.Tensor]]]:
    losses: List[float] = []
    snapshots: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    start_feat, start_lab = extract_features(
        model.backbone, eval_loader_for_snapshots, device=device
    )
    snapshots["inicio"] = (start_feat, start_lab)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for view1, view2, labels in train_loader:
            view1 = view1.to(device, non_blocking=True)
            view2 = view2.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            images = torch.cat([view1, view2], dim=0)
            _, projections = model(images)

            bsz = labels.size(0)
            projections = projections.view(2, bsz, -1).permute(1, 0, 2).contiguous()

            loss = criterion(projections, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        losses.append(epoch_loss)
        if scheduler is not None:
            scheduler.step()
        print(f"SupCon | Epoch {epoch:02d}/{epochs} | Loss = {epoch_loss:.4f}")

        if epoch in snapshot_epochs:
            tag = snapshot_epochs[epoch]
            feat, lab = extract_features(
                model.backbone, eval_loader_for_snapshots, device=device
            )
            snapshots[tag] = (feat, lab)

    if "final" not in snapshots:
        feat, lab = extract_features(
            model.backbone, eval_loader_for_snapshots, device=device
        )
        snapshots["final"] = (feat, lab)

    return losses, snapshots


@torch.no_grad()
def evaluate_linear(
    backbone: nn.Module,
    linear_head: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    backbone.eval()
    linear_head.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        features = backbone(images)
        logits = linear_head(features)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    if total == 0:
        return 0.0
    return 100.0 * correct / total


def train_linear_head(
    backbone: nn.Module,
    linear_head: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1.0,
    scheduler_milestones: Optional[List[int]] = None,
    scheduler_gamma: float = 0.2,
    report_best: bool = False,
) -> Tuple[Dict[str, List[float]], float]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        linear_head.parameters(), lr=lr, momentum=0.9, weight_decay=0.0
    )
    if scheduler_milestones is None:
        scheduler_milestones = sorted(
            {
                max(1, int(epochs * 0.60)),
                max(1, int(epochs * 0.75)),
                max(1, int(epochs * 0.90)),
            }
        )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=scheduler_milestones,
        gamma=scheduler_gamma,
    )
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "test_acc": [],
    }

    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    for epoch in range(1, epochs + 1):
        linear_head.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.no_grad():
                features = backbone(images)
            logits = linear_head(features)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / max(1, len(train_loader))
        train_acc = 100.0 * correct / max(1, total)
        test_acc = evaluate_linear(backbone, linear_head, test_loader, device=device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        scheduler.step()

        print(
            f"Linear Eval | epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.2f}% | test_acc={test_acc:.2f}%"
        )

    metric = max(history["test_acc"]) if report_best else history["test_acc"][-1]
    return history, metric


def _resolve_task_ids(
    loaders: Dict[int, torch.utils.data.DataLoader], task_ids: Optional[List[int]]
) -> List[int]:
    available_task_ids = sorted(loaders.keys())
    if task_ids is None:
        return available_task_ids

    unknown = [task_id for task_id in task_ids if task_id not in loaders]
    if len(unknown) > 0:
        raise ValueError(f"Unknown task ids requested: {unknown}")
    return task_ids


@torch.no_grad()
def evaluate_class_il(
    model: nn.Module,
    test_loaders: Dict[int, torch.utils.data.DataLoader],
    seen_task_ids: List[int],
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total = 0

    for task_id in seen_task_ids:
        loader = test_loaders[task_id]
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        return 0.0
    return 100.0 * correct / total


@torch.no_grad()
def evaluate_task_il(
    model: nn.Module,
    test_loaders: Dict[int, torch.utils.data.DataLoader],
    task_classes: List[List[int]],
    seen_task_ids: List[int],
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total = 0

    for task_id in seen_task_ids:
        allowed_classes = torch.tensor(task_classes[task_id], device=device)
        loader = test_loaders[task_id]

        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            task_logits = logits[:, allowed_classes]
            task_preds_local = task_logits.argmax(dim=1)
            task_preds_global = allowed_classes[task_preds_local]

            correct += (task_preds_global == labels).sum().item()
            total += labels.size(0)

    if total == 0:
        return 0.0
    return 100.0 * correct / total


@torch.no_grad()
def evaluate_taskwise_class_il(
    model: nn.Module,
    test_loaders: Dict[int, torch.utils.data.DataLoader],
    seen_task_ids: List[int],
    device: torch.device,
    n_tasks: Optional[int] = None,
) -> List[float]:
    """Per-task Class-IL accuracy for all seen tasks.

    Returns a dense list of length `n_tasks` with NaN for tasks not yet seen.
    """
    if n_tasks is None:
        n_tasks = len(test_loaders)

    out = [float("nan")] * n_tasks
    model.eval()

    for task_id in seen_task_ids:
        loader = test_loaders[task_id]
        correct = 0
        total = 0

        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        out[task_id] = 0.0 if total == 0 else 100.0 * correct / total

    return out


@torch.no_grad()
def evaluate_taskwise_task_il(
    model: nn.Module,
    test_loaders: Dict[int, torch.utils.data.DataLoader],
    task_classes: List[List[int]],
    seen_task_ids: List[int],
    device: torch.device,
    n_tasks: Optional[int] = None,
) -> List[float]:
    """Per-task Task-IL accuracy for all seen tasks.

    Returns a dense list of length `n_tasks` with NaN for tasks not yet seen.
    """
    if n_tasks is None:
        n_tasks = len(task_classes)

    out = [float("nan")] * n_tasks
    model.eval()

    for task_id in seen_task_ids:
        allowed_classes = torch.tensor(task_classes[task_id], device=device)
        loader = test_loaders[task_id]
        correct = 0
        total = 0

        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            task_logits = logits[:, allowed_classes]
            task_preds_local = task_logits.argmax(dim=1)
            task_preds_global = allowed_classes[task_preds_local]

            correct += (task_preds_global == labels).sum().item()
            total += labels.size(0)

        out[task_id] = 0.0 if total == 0 else 100.0 * correct / total

    return out


def _ewc_penalty(
    model: nn.Module,
    ewc_terms: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]],
) -> torch.Tensor:
    if len(ewc_terms) == 0:
        return torch.tensor(0.0, device=next(model.parameters()).device)

    penalty = torch.tensor(0.0, device=next(model.parameters()).device)
    named_params = dict(model.named_parameters())

    for fisher_diag, params_star in ewc_terms:
        for name, fisher_value in fisher_diag.items():
            param = named_params[name]
            penalty = (
                penalty + (fisher_value * (param - params_star[name]).pow(2)).sum()
            )

    return penalty


def _compute_fisher_diagonal(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    model.eval()
    fisher_diag = {
        name: torch.zeros_like(param, device=device)
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    n_batches = 0
    for batch_idx, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        model.zero_grad(set_to_none=True)
        logits = model(images)
        loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.grad is None:
                continue
            fisher_diag[name] += param.grad.detach().pow(2)

        n_batches += 1

    normalizer = max(1, n_batches)
    for name in fisher_diag:
        fisher_diag[name] /= normalizer

    return fisher_diag


def _snapshot_parameters(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def _train_task_classifier(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    task_id: int,
    verbose: bool,
    ewc_terms: Optional[
        List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]
    ] = None,
    ewc_lambda: float = 0.0,
) -> float:
    criterion = nn.CrossEntropyLoss()
    avg_loss = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if ewc_terms is not None and len(ewc_terms) > 0 and ewc_lambda > 0:
                loss = loss + ewc_lambda * _ewc_penalty(model, ewc_terms)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(1, len(train_loader))
        if verbose:
            print(
                f"Task {task_id} | Epoch {epoch:02d}/{epochs} | "
                f"train_loss={avg_loss:.4f}"
            )

    return avg_loss


def train_naive_finetuning(
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

        if verbose:
            print(
                f"Task {task_id} done | Class-IL={class_il_acc:.2f}% | "
                f"Task-IL={task_il_acc:.2f}%"
            )

    return model, history


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
