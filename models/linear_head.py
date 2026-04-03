"""
Linear classification head attached on top of a frozen backbone.

Used for both:
  • Single-task evaluation after contrastive pre-training (Etapa 2).
  • Per-task heads in Task-IL evaluation.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LinearHead(nn.Module):
    """Single linear layer mapping backbone features to class logits.

    Parameters
    ----------
    in_dim:
        Backbone output dimensionality (512 for ResNet-18).
    num_classes:
        Number of output classes.
    """

    def __init__(self, in_dim: int = 512, num_classes: int = 2) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)
