import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from .models import SupConNetwork, SupConLoss


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

    return 100.0 * correct / total


def train_linear_head(
    backbone: nn.Module,
    linear_head: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1.0,
) -> Tuple[Dict[str, List[float]], float]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        linear_head.parameters(), lr=lr, momentum=0.9, weight_decay=0.0
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[60, 75, 90], gamma=0.2
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

        train_loss = running_loss / len(train_loader)
        train_acc = 100.0 * correct / total
        test_acc = evaluate_linear(backbone, linear_head, test_loader, device=device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        scheduler.step()

        print(
            f"Linear Eval | epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.2f}% | test_acc={test_acc:.2f}%"
        )

    final_test_acc = history["test_acc"][-1]
    return history, final_test_acc
