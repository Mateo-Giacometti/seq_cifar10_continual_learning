from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def ensure_checkpoint_dir(path: str | Path) -> Path:
    """
    Ensure that the checkpoint directory exists.

    Parameters
    ----------
    path : str | Path
        The path to the checkpoint directory.

    Returns
    -------
    Path
        The checkpoint directory.
    """
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    """
    Save a checkpoint.

    Parameters
    ----------
    path : str | Path
        The path to save the checkpoint to.
    payload : dict[str, Any]
        The payload to save.

    Returns
    -------
    None
    """
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, checkpoint_path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """
    Load a checkpoint.

    Parameters
    ----------
    path : str | Path
        The path to the checkpoint.
    map_location : str | torch.device, optional
        The device to map the checkpoint to, by default "cpu".

    Returns
    -------
    dict[str, Any]
        The checkpoint.
    """
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid checkpoint format at: {checkpoint_path}")
    return payload
