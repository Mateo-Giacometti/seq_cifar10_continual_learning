"""
ResNet-18 backbone for feature extraction.

The final fully-connected layer is removed so the backbone outputs a flat
feature vector of dimension 512 (the last ResNet-18 block width).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18


class ResNet18Backbone(nn.Module):
    """ResNet-18 without the classification head.

    Parameters
    ----------
    pretrained:
        If ``True`` load ImageNet-pretrained weights.  For CIFAR-10 we
        typically start from scratch (``pretrained=False``) because the
        images are 32×32 and the domain differs from ImageNet.
    cifar_stem:
        Replace the first 7×7 conv + max-pool with a 3×3 conv (no
        downsampling), which is the standard adjustment for 32×32 inputs.
    freeze:
        Freeze all parameters (useful when attaching a linear probe).
    """

    FEATURE_DIM = 512

    def __init__(
        self,
        pretrained: bool = False,
        cifar_stem: bool = True,
        freeze: bool = False,
    ) -> None:
        super().__init__()

        weights = "IMAGENET1K_V1" if pretrained else None
        base = resnet18(weights=weights)

        if cifar_stem:
            # Replace 7×7 stem with a lightweight 3×3 conv suitable for 32×32.
            base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            base.maxpool = nn.Identity()

        # Drop the final FC layer.
        self.encoder = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
            base.avgpool,
        )

        if freeze:
            self.freeze()

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a (batch, 512) feature tensor."""
        h = self.encoder(x)
        return torch.flatten(h, 1)

    # ------------------------------------------------------------------

    def freeze(self) -> None:
        """Freeze all backbone parameters."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all backbone parameters."""
        for param in self.parameters():
            param.requires_grad = True
