from copy import deepcopy
import random
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

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
    """
    Extract features from a backbone network for a given data loader.

    Parameters
    ----------
    backbone : nn.Module
        The backbone network to extract features from.
    loader : torch.utils.data.DataLoader
        The data loader containing the data to extract features from.
    device : torch.device
        The device to use for feature extraction.
    max_samples : int
        The maximum number of samples to extract features from.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        A tuple containing the features and the labels.
    """
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
    """
    Train the SupCon model.

    Parameters
    ----------
    model : SupConNetwork
        The SupCon model to train.
    train_loader : torch.utils.data.DataLoader
        The data loader containing the training data.
    eval_loader_for_snapshots : torch.utils.data.DataLoader
        The data loader containing the evaluation data for snapshots.
    criterion : SupConLoss
        The SupCon loss function.
    optimizer : torch.optim.Optimizer
        The optimizer to use for training.
    device : torch.device
        The device to use for training.
    epochs : int
        The number of epochs to train for.
    snapshot_epochs : Dict[int, str]
        A dictionary containing the epoch numbers and the names of the snapshots.
    scheduler : torch.optim.lr_scheduler._LRScheduler | None, optional
        The learning rate scheduler, by default None.

    Returns
    -------
    Tuple[List[float], Dict[str, Tuple[torch.Tensor, torch.Tensor]]]
        A tuple containing the losses and the snapshots.
    """ 
    losses: List[float] = []
    snapshots: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    start_feat, start_lab = extract_features(model.backbone, eval_loader_for_snapshots, device=device)
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
    """
    Evaluate the linear head on the given data loader.

    Parameters
    ----------
    backbone : nn.Module
        The backbone network to evaluate.
    linear_head : nn.Module
        The linear head to evaluate.
    loader : torch.utils.data.DataLoader
        The data loader containing the data to evaluate on.
    device : torch.device
        The device to use for evaluation.

    Returns
    -------
    float
        The accuracy of the linear head on the given data loader.
    """
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
    """
    Train the linear head.

    Parameters
    ----------
    backbone : nn.Module
        The backbone network to train.
    linear_head : nn.Module
        The linear head to train.
    train_loader : torch.utils.data.DataLoader
        The data loader containing the training data.
    test_loader : torch.utils.data.DataLoader
        The data loader containing the test data.
    device : torch.device
        The device to use for training.
    epochs : int
        The number of epochs to train for.
    lr : float
        The learning rate.
    scheduler_milestones : Optional[List[int]]
        The learning rate scheduler milestones.
    scheduler_gamma : float
        The learning rate scheduler gamma.
    report_best : bool
        Whether to report the best accuracy.

    Returns
    -------
    Tuple[Dict[str, List[float]], float]
        A tuple containing the history and the best accuracy.
    """  
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
            f"Linear Eval | Epoch {epoch:02d}/{epochs} | "
            f"Train Loss = {train_loss:.4f} | Train Acc = {train_acc:.2f}% | Test Acc = {test_acc:.2f}%"
        )

    metric = max(history["test_acc"]) if report_best else history["test_acc"][-1]
    return history, metric


def _resolve_task_ids(
    loaders: Dict[int, torch.utils.data.DataLoader], task_ids: Optional[List[int]]
) -> List[int]:
    """
    Resolve the task ids.

    Parameters
    ----------
    loaders : Dict[int, torch.utils.data.DataLoader]
        A dictionary containing the task ids and the data loaders.
    task_ids : Optional[List[int]]
        The task ids to resolve.

    Returns
    -------
    List[int]
        The resolved task ids.
    """  
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
    """
    Evaluate the class-incremental learning model.

    Parameters
    ----------
    model : nn.Module
        The model to evaluate.
    test_loaders : Dict[int, torch.utils.data.DataLoader]
        A dictionary containing the task ids and the data loaders.
    seen_task_ids : List[int]
        The task ids that have been seen.
    device : torch.device
        The device to use for evaluation.

    Returns
    -------
    float
        The accuracy of the model on the given data loader.
    """  
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
    """
    Evaluate the task-incremental learning model.

    Parameters
    ----------
    model : nn.Module
        The model to evaluate.
    test_loaders : Dict[int, torch.utils.data.DataLoader]
        A dictionary containing the task ids and the data loaders.
    task_classes : List[List[int]]
        A list containing the task classes.
    seen_task_ids : List[int]
        The task ids that have been seen.
    device : torch.device
        The device to use for evaluation.

    Returns
    -------
    float
        The accuracy of the model on the given data loader.
    """  
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
    """
    Evaluate the taskwise class-incremental learning model.

    Parameters
    ----------
    model : nn.Module
        The model to evaluate.
    test_loaders : Dict[int, torch.utils.data.DataLoader]
        A dictionary containing the task ids and the data loaders.
    seen_task_ids : List[int]
        The task ids that have been seen.
    device : torch.device
        The device to use for evaluation.
    n_tasks : Optional[int], optional
        The number of tasks, by default None.

    Returns
    -------
    List[float]
        A list containing the accuracy of the model on the given data loader.
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
    """
    Evaluate the taskwise task-incremental learning model.

    Parameters
    ----------
    model : nn.Module
        The model to evaluate.
    test_loaders : Dict[int, torch.utils.data.DataLoader]
        A dictionary containing the task ids and the data loaders.
    task_classes : List[List[int]]
        A list containing the task classes.
    seen_task_ids : List[int]
        The task ids that have been seen.
    device : torch.device
        The device to use for evaluation.
    n_tasks : Optional[int], optional
        The number of tasks, by default None.

    Returns
    -------
    List[float]
        A list containing the accuracy of the model on the given data loader.
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


@torch.no_grad()
def get_confusion_matrix(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int = 10,
) -> np.ndarray:
    """
    Compute the confusion matrix for the given model and data loader.

    Parameters
    ----------
    model : nn.Module
        The model to evaluate.
    loader : torch.utils.data.DataLoader
        The data loader containing the data to evaluate on.
    device : torch.device
        The device to use for evaluation.
    num_classes : int, optional
        The number of classes, by default 10.

    Returns
    -------
    np.ndarray
        The confusion matrix.
    """  

    model.eval()
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        output = model(images)
        if isinstance(output, (tuple, list)):
            logits = output[0]
        else:
            logits = output

        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

    return confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))


def _ewc_penalty(
    model: nn.Module,
    ewc_terms: List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]],
) -> torch.Tensor:
    """
    Compute the EWC penalty for the given model and EWC terms.

    Parameters
    ----------
    model : nn.Module
        The model to evaluate.
    ewc_terms : List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]
        A list containing the EWC terms.

    Returns
    -------
    torch.Tensor
        The EWC penalty.
    """  
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
    fisher_loss_mode: Literal["ce", "nll_true", "nll_pred"] = "ce",
) -> Dict[str, torch.Tensor]:
    """
    Compute the Fisher diagonal for the given model and data loader.

    Parameters
    ----------
    model : nn.Module
        The model to evaluate.
    loader : torch.utils.data.DataLoader
        The data loader containing the data to evaluate on.
    device : torch.device
        The device to use for evaluation.
    max_batches : Optional[int], optional
        The maximum number of batches to use for evaluation, by default None.
    fisher_loss_mode : Literal["ce", "nll_true", "nll_pred"], optional
        The Fisher loss mode, by default "ce".

    Returns
    -------
    Dict[str, torch.Tensor]
        The Fisher diagonal.
    """  
    if fisher_loss_mode not in {"ce", "nll_true", "nll_pred"}:
        raise ValueError(
            "fisher_loss_mode must be one of {'ce', 'nll_true', 'nll_pred'}"
        )

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
        if fisher_loss_mode == "ce":
            loss = nn.functional.cross_entropy(logits, labels)
        else:
            log_probs = nn.functional.log_softmax(logits, dim=1)
            if fisher_loss_mode == "nll_true":
                targets = labels
            else:
                targets = logits.argmax(dim=1)
            loss = -log_probs.gather(1, targets.unsqueeze(1)).mean()
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
    """
    Snapshot the parameters of the given model.

    Parameters
    ----------
    model : nn.Module
        The model to snapshot.

    Returns
    -------
    Dict[str, torch.Tensor]
        The snapshot of the parameters.
    """  
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
    """
    Train the continual classifier.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    train_loader : torch.utils.data.DataLoader
        The data loader containing the data to train on.
    optimizer : torch.optim.Optimizer
        The optimizer to use for training.
    device : torch.device
        The device to use for training.
    epochs : int
        The number of epochs to train for.
    task_id : int
        The task id.
    verbose : bool
        Whether to print the training progress.
    ewc_terms : Optional[List[Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]], optional]
        A list containing the EWC terms, by default None.
    ewc_lambda : float, optional
        The EWC lambda parameter, by default 0.0.

    Returns
    -------
    float
        The average loss.
    """  
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
                f"Train Loss = {avg_loss:.4f}"
            )

    return avg_loss


