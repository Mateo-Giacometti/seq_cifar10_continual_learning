"""
Contrastive pre-training loop (Etapa 2).

Trains a backbone + projection head with SupCon loss on a single task
(typically Task 0) before any Continual Learning phase begins.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from models.backbone import ResNet18Backbone
from models.projection_head import ProjectionHead


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., NeurIPS 2020).

    Parameters
    ----------
    temperature:
        Logit scale τ.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        features:
            L2-normalised projections of shape (2N, dim) – two views
            concatenated along the batch dimension.
        labels:
            Ground-truth class labels of shape (N,).  Repeated internally to
            match the 2N feature dimension.
        """
        # Expected shape: (2N, dim) with two augmented views per sample.
        device = features.device
        n2 = features.shape[0]

        if labels.ndim != 1:
            labels = labels.view(-1)

        if labels.shape[0] <= 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Duplicate labels to match the number of views in features.
        if n2 % labels.shape[0] != 0:
            raise ValueError(
                f"SupConLoss: feature batch ({n2}) is not a multiple of labels ({labels.shape[0]})."
            )
        n_views = n2 // labels.shape[0]
        labels = labels.repeat_interleave(n_views).view(-1, 1)

        # Pairwise similarities.
        logits = torch.mm(features, features.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()  # numerical stability

        # Masks.
        logits_mask = torch.ones_like(logits, device=device)
        logits_mask.fill_diagonal_(0)
        pos_mask = (labels == labels.T).float() * logits_mask

        # Log-probabilities over non-self entries.
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        pos_per_anchor = pos_mask.sum(dim=1)
        valid = pos_per_anchor > 0

        if not torch.any(valid):
            return torch.tensor(0.0, device=device, requires_grad=True)

        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_per_anchor.clamp(min=1.0)
        loss = -mean_log_prob_pos[valid].mean()
        return loss


class ContrastiveTrainer:
    """Trains a ResNet-18 backbone with SupCon loss.

    Parameters
    ----------
    backbone:
        :class:`~models.backbone.ResNet18Backbone` instance.
    proj_head:
        :class:`~models.projection_head.ProjectionHead` instance.
    device:
        Computation device.
    temperature:
        SupCon temperature.
    """

    def __init__(
        self,
        backbone: ResNet18Backbone,
        proj_head: ProjectionHead,
        device: Optional[torch.device] = None,
        temperature: float = 0.07,
    ) -> None:
        self.backbone = backbone
        self.proj_head = proj_head
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.supcon = SupConLoss(temperature=temperature)

        self.backbone.to(self.device)
        self.proj_head.to(self.device)

        self.loss_history: List[float] = []

    # ------------------------------------------------------------------

    def train(
        self,
        train_loader: DataLoader,
        epochs: int = 200,
        lr: float = 0.05,
        weight_decay: float = 1e-4,
    ) -> List[float]:
        """Run contrastive pre-training.

        The *train_loader* must use :class:`~data.cifar10_tasks.TwoViewTransform`
        so each sample yields a ``(view1, view2)`` pair.

        Parameters
        ----------
        train_loader:
            DataLoader that returns ``(images, local_labels, global_labels)``
            where *images* is a ``(view1, view2)`` tuple.
        epochs:
            Number of training epochs.
        lr:
            Initial learning rate.
        weight_decay:
            L2 regularisation coefficient.

        Returns
        -------
        loss_history:
            List of mean epoch losses.
        """
        optimizer = SGD(
            list(self.backbone.parameters()) + list(self.proj_head.parameters()),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        self.loss_history = []

        for epoch in range(epochs):
            self.backbone.train()
            self.proj_head.train()

            epoch_loss = 0.0
            total = 0

            for batch in train_loader:
                images, local_labels, _ = batch
                local_labels = local_labels.to(self.device)

                if isinstance(images, (list, tuple)):
                    view1, view2 = images
                    view1 = view1.to(self.device)
                    view2 = view2.to(self.device)
                    imgs_cat = torch.cat([view1, view2], dim=0)  # (2N, C, H, W)
                else:
                    imgs_cat = images.to(self.device)

                optimizer.zero_grad()
                features = self.backbone(imgs_cat)
                projections = self.proj_head(features)  # (2N, proj_dim)

                loss = self.supcon(projections, local_labels)
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        "Non-finite SupCon loss detected. "
                        "Check augmentations, labels, and feature normalization."
                    )
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * len(imgs_cat)
                total += len(imgs_cat)

            scheduler.step()

            avg_loss = epoch_loss / total
            self.loss_history.append(avg_loss)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"  [ContrastiveTrainer] Epoch {epoch + 1}/{epochs} "
                    f"| SupCon Loss: {avg_loss:.4f}"
                )

        return self.loss_history
