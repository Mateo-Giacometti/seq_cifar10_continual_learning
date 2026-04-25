from __future__ import annotations

import math
import warnings
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
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

from ._utils import append_baseline_row

def _clone_replay_buffer(source: ReservoirReplayBuffer, seed: int) -> ReservoirReplayBuffer:
    """
    Clones a replay buffer.

    Parameters
    ----------
    source : ReservoirReplayBuffer
        The replay buffer to clone.
    seed : int
        The seed to use for cloning.
   
    Returns
    -------
    ReservoirReplayBuffer
        The cloned replay buffer.
    """
    out = ReservoirReplayBuffer(capacity=source.capacity, seed=seed)
    out.images = [img.clone() for img in source.images]
    out.labels = list(source.labels)
    out.seen = source.seen
    return out


class _TensorDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Initialize the TensorDataset.
        
        Parameters
        ----------
        images : torch.Tensor
            The images.
        labels : torch.Tensor
            The labels.
        """
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        """ 
        Return the number of samples.
        """
        return self.labels.size(0)

    def __getitem__(self, idx: int):
        """
        Return the sample at the given index.
        """
        return self.images[idx], self.labels[idx]


def _make_balanced_sampler(labels: torch.Tensor) -> WeightedRandomSampler:
    """
    Make a balanced sampler.
    
    Parameters
    ----------
    labels : torch.Tensor
        The labels.
    
    Returns
    -------
    WeightedRandomSampler
        The balanced sampler.
    """
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
    """
    Make a scheduler for Co2L.
    
    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer.
    epochs : int
        The number of epochs.
    warmup_epochs : int
        The number of warmup epochs.
   
    Returns
    -------
    torch.optim.lr_scheduler.LambdaLR
        The scheduler.
    """
    def lr_lambda(epoch: int) -> float:
        """
        Calculate the learning rate.
        Parameters
        ----------
        epoch : int
            The current epoch.
        Returns
        -------
        float
            The learning rate.
        """
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        if epochs == warmup_epochs:
            return 1.0
        progress = float(epoch - warmup_epochs) / float(epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _random_resized_crop_batch(
    images: torch.Tensor,
    scale: Tuple[float, float] = (0.2, 1.0),
    ratio: Tuple[float, float] = (0.75, 1.3333333333333333),
) -> torch.Tensor:
    """
    Perform random resized crop on a batch of images.
    
    Parameters
    ----------
    images : torch.Tensor
        The images.
    scale : Tuple[float, float]
        The scale.
    ratio : Tuple[float, float]
        The ratio.
    
    Returns
    -------
    torch.Tensor
        The cropped images.
    """
    if images.dim() != 4:
        raise ValueError("Expected images with shape (B, C, H, W)")

    bsz, _, height, width = images.shape
    out = images.clone()
    area = float(height * width)

    for i in range(bsz):
        cropped = False
        for _ in range(10):
            target_area = area * torch.empty((), device=images.device).uniform_(
                scale[0], scale[1]
            ).item()
            aspect_ratio = torch.empty((), device=images.device).uniform_(
                ratio[0], ratio[1]
            ).item()

            crop_h = int(round(math.sqrt(target_area / aspect_ratio)))
            crop_w = int(round(math.sqrt(target_area * aspect_ratio)))

            if 0 < crop_h <= height and 0 < crop_w <= width:
                top = int(
                    torch.randint(0, height - crop_h + 1, (1,), device=images.device).item()
                )
                left = int(
                    torch.randint(0, width - crop_w + 1, (1,), device=images.device).item()
                )
                crop = images[i : i + 1, :, top : top + crop_h, left : left + crop_w]
                out[i] = F.interpolate(
                    crop,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )[0]
                cropped = True
                break

        if not cropped:
            out[i] = images[i]

    return out


def _augment_buffer_batch(images: torch.Tensor) -> torch.Tensor:
    """
    Augment a batch of images.

    Parameters
    ----------
    images : torch.Tensor
        The images.
    
    Returns
    -------
    torch.Tensor
        The augmented images.
    """
    if images.dim() != 4:
        raise ValueError("Expected images with shape (B, C, H, W)")
    if images.size(0) == 0:
        return images

    out = _random_resized_crop_batch(images, scale=(0.2, 1.0))

    flip_mask = torch.rand(out.size(0), device=out.device) < 0.5
    if torch.any(flip_mask):
        out[flip_mask] = torch.flip(out[flip_mask], dims=[-1])

    jitter_mask = torch.rand(out.size(0), device=out.device) < 0.8
    if torch.any(jitter_mask):
        idx = torch.where(jitter_mask)[0]
        x = out[idx]

        brightness = torch.empty((x.size(0), 1, 1, 1), device=x.device, dtype=x.dtype).uniform_(
            0.6, 1.4
        )
        x = x * brightness

        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        contrast = torch.empty((x.size(0), 1, 1, 1), device=x.device, dtype=x.dtype).uniform_(
            0.6, 1.4
        )
        x = (x - mean) * contrast + mean

        coeff = torch.tensor([0.2989, 0.5870, 0.1140], device=x.device, dtype=x.dtype).view(
            1, 3, 1, 1
        )
        gray = (x * coeff).sum(dim=1, keepdim=True)
        saturation = torch.empty((x.size(0), 1, 1, 1), device=x.device, dtype=x.dtype).uniform_(
            0.6, 1.4
        )
        x = (x - gray) * saturation + gray

        out[idx] = x

    grayscale_mask = torch.rand(out.size(0), device=out.device) < 0.2
    if torch.any(grayscale_mask):
        x = out[grayscale_mask]
        coeff = torch.tensor([0.2989, 0.5870, 0.1140], device=x.device, dtype=x.dtype).view(
            1, 3, 1, 1
        )
        gray = (x * coeff).sum(dim=1, keepdim=True)
        out[grayscale_mask] = gray.repeat(1, 3, 1, 1)

    return out


def _extract_images_labels(batch: object) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract images and labels from a batch.

    Parameters
    ----------
    batch : object
        The batch.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        The images and labels.
    """
    if not isinstance(batch, (tuple, list)):
        raise ValueError("Expected batch to be a tuple/list")

    if len(batch) == 2:
        images, labels = batch
        if not isinstance(images, torch.Tensor) or not isinstance(labels, torch.Tensor):
            raise ValueError("Batch must contain torch.Tensor images and labels")
        return images, labels

    if len(batch) == 3:
        view1, _, labels = batch
        if not isinstance(view1, torch.Tensor) or not isinstance(labels, torch.Tensor):
            raise ValueError("Batch must contain torch.Tensor views and labels")
        return view1, labels

    raise ValueError("Batch must be (images, labels) or (view1, view2, labels)")


def _collect_images_labels_from_loader(loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collect images and labels from a loader.

    Parameters
    ----------
    loader : DataLoader
        The loader.
    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        The images and labels.
    """
    images_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    for batch in loader:
        images, labels = _extract_images_labels(batch)
        images_list.append(images.detach().cpu())
        labels_list.append(labels.detach().cpu())

    if not images_list:
        raise ValueError("Cannot collect data from an empty loader")

    return torch.cat(images_list, dim=0), torch.cat(labels_list, dim=0)


def _train_linear_head_balanced(
    backbone: nn.Module,
    feat_dim: int,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
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
    """
    Train a linear head on balanced data.

    Parameters
    ----------
    backbone : nn.Module
        The backbone.
    feat_dim : int
        The feature dimension.
    train_images : torch.Tensor
        The training images.
    train_labels : torch.Tensor
        The training labels.
    test_loaders : Dict[int, DataLoader]
        The test loaders.
    task_classes : List[List[int]]
        The task classes.
    seen_task_ids : List[int]
        The seen task ids.
    device : torch.device
        The device.
    num_classes : int
        The number of classes.
    epochs : int
        The number of epochs.
    lr : float
        The learning rate.
    milestones : List[int]
        The milestones.
    gamma : float
        The gamma.
    batch_size : int
        The batch size.
    
    Returns
    -------
    ContinualClassifier
        The trained classifier.
    """
    model = ContinualClassifier(
        backbone=deepcopy(backbone),
        feat_dim=feat_dim,
        num_classes=num_classes,
    ).to(device)
    model.backbone.eval()
    for p in model.backbone.parameters():
        p.requires_grad = False

    eval_ds = _TensorDataset(train_images, train_labels)
    sampler = _make_balanced_sampler(train_labels)
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
    model.class_il_metric = class_il
    model.task_il_metric = task_il
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
    """
    Train the Co2L model.

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
    proj_dim : int
        The projection dimension.
    buffer_size : int
        The buffer size.
    lr : float
        The learning rate.
    momentum : float
        The momentum.
    weight_decay : float
        The weight decay.
    temperature : float
        The temperature.
    kappa : float
        The kappa.
    kappa_star : float
        The kappa star.
    lambda_ird : float
        The lambda ird.
    warmup_epochs : int
        The number of warmup epochs.
    epochs_task0 : int
        The number of epochs for task 0.
    epochs_per_task : int
        The number of epochs per task.
    eval_linear_epochs : int
        The number of epochs for linear evaluation.
    eval_linear_lr : float
        The learning rate for linear evaluation.
    eval_linear_milestones : Optional[List[int]]
        The milestones for linear evaluation.
    eval_linear_gamma : float
        The gamma for linear evaluation.
    train_loaders_eval : Optional[Dict[int, DataLoader]]
        The training loaders for evaluation.
    task_ids : Optional[List[int]]
        The task ids.
    seed : int
        The seed.
    max_replay_batch_size : Optional[int]
        The maximum replay batch size.
    initial_model : Optional[SupConNetwork]
        The initial model.
    initial_seen_task_ids : Optional[List[int]]
        The initial seen task ids.
    initial_past_model : Optional[SupConNetwork]
        The initial past model.
    initial_buffer : Optional[ReservoirReplayBuffer]
        The initial buffer.
    baseline_payload : Optional[Dict[str, object]]
        The baseline payload.
    verbose : bool
        Whether to print verbose output.
    Returns
    -------
    Tuple[SupConNetwork, Dict[str, object]]
        The trained model and the history.
    """
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

    append_baseline_row(history, baseline_payload, extra_keys={"ird_loss": 0.0})

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
                if not isinstance(batch, (tuple, list)) or len(batch) != 3:
                    raise ValueError(
                        "Co2L expects contrastive batches with (view1, view2, labels). "
                        "Build train_loaders with ContrastiveTaskDataset/TwoCropTransform."
                    )

                view1_curr, view2_curr, labels_curr = batch
                view1_curr = view1_curr.to(device, non_blocking=True)
                view2_curr = view2_curr.to(device, non_blocking=True)
                labels_curr = labels_curr.to(device, non_blocking=True)
                bsz_curr = labels_curr.size(0)

                if len(buffer) > 0:
                    replay_batch = bsz_curr
                    if max_replay_batch_size is not None:
                        replay_batch = min(replay_batch, max_replay_batch_size)

                    images_buf, labels_buf = buffer.sample(replay_batch)
                    images_buf = images_buf.to(device, non_blocking=True)
                    labels_buf = labels_buf.to(device, non_blocking=True)
                    replay_k = labels_buf.size(0)

                    view1_buf = _augment_buffer_batch(images_buf)
                    view2_buf = _augment_buffer_batch(images_buf)

                    view1_all = torch.cat([view1_curr, view1_buf], dim=0)
                    view2_all = torch.cat([view2_curr, view2_buf], dim=0)
                    labels_all = torch.cat([labels_curr, labels_buf], dim=0)
                    current_mask = torch.zeros(
                        bsz_curr + replay_k,
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
                    f"Loss = {epoch_loss:.4f} | IRD = {epoch_ird:.4f}"
                )

        past_model = deepcopy(model).to(device).eval()
        for p in past_model.parameters():
            p.requires_grad = False

        buffer_source = train_loaders_eval if train_loaders_eval is not None else train_loaders
        if train_loaders_eval is None:
            warnings.warn(
                "train_loaders_eval not provided to train_co2l; replay buffer will "
                "store augmented images.  Pass eval-transformed loaders for optimal "
                "replay quality.",
                stacklevel=2,
            )
        for batch in buffer_source[task_id]:
            images, labels = _extract_images_labels(batch)
            buffer.add(images, labels)

        seen_task_ids.append(task_id)

        buffer_images = torch.stack(buffer.images, dim=0)
        buffer_labels = torch.tensor(buffer.labels, dtype=torch.long)
        task_images, task_labels = _collect_images_labels_from_loader(buffer_source[task_id])
        eval_train_images = torch.cat([task_images, buffer_images], dim=0)
        eval_train_labels = torch.cat([task_labels, buffer_labels], dim=0)

        eval_model = _train_linear_head_balanced(
            backbone=model.backbone,
            feat_dim=feat_dim,
            train_images=eval_train_images,
            train_labels=eval_train_labels,
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
        class_il_acc = float(eval_model.class_il_metric)  
        task_il_acc = float(eval_model.task_il_metric)  

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
                f"Class-IL = {class_il_acc:.2f}% | Task-IL = {task_il_acc:.2f}%"
            )

    return model, history
