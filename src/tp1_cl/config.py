from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

class ConfigNode:
    def __init__(self, data: dict[str, Any]) -> None:
        """
        Initialize the ConfigNode.

        Parameters
        ----------
        data : dict[str, Any]
            The data to initialize the ConfigNode with.
        """
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, ConfigNode(value))
            else:
                setattr(self, key, value)

    def __getitem__(self, key: str) -> Any:
        """
        Get an item from the ConfigNode.

        Parameters
        ----------
        key : str
            The key of the item to get.

        Returns
        -------
        Any
            The item.
        """
        return getattr(self, key)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the ConfigNode to a dictionary.

        Returns
        -------
        dict[str, Any]
            The dictionary.
        """
        out: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            out[key] = value.to_dict() if isinstance(value, ConfigNode) else value
        return out


def load_config(path: str | Path) -> ConfigNode:
    """
    Load a configuration from a YAML file.

    Parameters
    ----------
    path : str | Path
        The path to the YAML file.

    Returns
    -------
    ConfigNode
        The configuration.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping")

    return ConfigNode(data)
