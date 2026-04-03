"""
MLP Projection Head for contrastive learning (SupCon / SimCLR).

Projects backbone features to a lower-dimensional space where the
contrastive loss is applied.  Following the original papers, the head is
discarded after pre-training and not used at evaluation time.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """Two-layer MLP projection head: Linear → BN → ReLU → Linear → L2-norm.

    Parameters
    ----------
    in_dim:
        Dimensionality of the backbone output (512 for ResNet-18).
    hidden_dim:
        Width of the hidden layer (default 512, same as the backbone).
    out_dim:
        Output (projection) dimensionality (default 128).
    """

    def __init__(
        self,
        in_dim: int = 512,
        hidden_dim: int = 512,
        out_dim: int = 128,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised projections of shape (batch, out_dim)."""
        z = self.net(x)
        return nn.functional.normalize(z, dim=1)
