#!/usr/bin/env python3
"""
Download CIFAR-10 to data/cifar10.

Run from the repository root:
    python scripts/download_data.py
"""

from pathlib import Path

import torchvision.datasets as datasets


def main() -> None:
    root = Path(__file__).parent.parent / "data" / "cifar10"
    root.mkdir(parents=True, exist_ok=True)

    print(f"Downloading CIFAR-10 to: {root.resolve()}")
    datasets.CIFAR10(root=str(root), train=True, download=True)
    datasets.CIFAR10(root=str(root), train=False, download=True)
    print("Done. CIFAR-10 is ready.")


if __name__ == "__main__":
    main()
