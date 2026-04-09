from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SupConNetwork(nn.Module):
    def __init__(self, backbone: nn.Module, feat_dim: int, proj_dim: int = 128) -> None:
        super().__init__()
        self.backbone = backbone
        self.projection = ProjectionHead(feat_dim, hidden_dim=512, out_dim=proj_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(x)
        proj = F.normalize(self.projection(feat), dim=1)
        return feat, proj


class ContinualClassifier(nn.Module):
    """Backbone + linear classifier used for continual class-incremental experiments."""

    def __init__(self, backbone: nn.Module, feat_dim: int, num_classes: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)


class SupConLoss(nn.Module):
    def __init__(
        self, temperature: float = 0.07, base_temperature: float | None = None
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.base_temperature = (
            temperature if base_temperature is None else base_temperature
        )

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device_local = features.device
        batch_size, n_views, _ = features.shape

        features = F.normalize(features, dim=2)
        contrast_features = features.view(batch_size * n_views, -1)

        labels_rep = labels.repeat_interleave(n_views)  # (B*n_views,)
        labels_rep = labels_rep.view(-1, 1)
        mask = torch.eq(labels_rep, labels_rep.T).float().to(device_local)

        logits = torch.matmul(contrast_features, contrast_features.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True)[0].detach()

        logits_mask = torch.ones_like(logits) - torch.eye(
            batch_size * n_views, device=device_local
        )
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        positive_mask = mask * logits_mask
        mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1) / (
            positive_mask.sum(dim=1) + 1e-12
        )

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(n_views, batch_size).mean()
        return loss


def build_cifar_resnet18() -> Tuple[nn.Module, int]:
    return build_cifar_resnet(model_name="resnet18")


def build_cifar_resnet(model_name: str = "resnet18") -> Tuple[nn.Module, int]:
    """Build a CIFAR-adapted ResNet backbone from torchvision."""
    if model_name == "resnet18":
        backbone = models.resnet18(weights=None)
    elif model_name == "resnet34":
        backbone = models.resnet34(weights=None)
    else:
        raise ValueError("model_name must be 'resnet18' or 'resnet34'")

    backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    backbone.maxpool = nn.Identity()
    feat_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()
    return backbone, feat_dim
