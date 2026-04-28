from __future__ import annotations
from typing import Dict, List, Optional


def append_baseline_row(
    history: Dict[str, object],
    baseline: Optional[Dict[str, object]],
    extra_keys: Optional[Dict[str, float]] = None,
) -> None:
    """
    Append Task-0 baseline metrics to a method's history dict.

    Parameters
    ----------
    history : dict
        The history dict that accumulates per-task metrics.
    baseline : dict or None
        The baseline payload from ``task0_baseline_payload``.
        If *None* the function is a no-op.
    extra_keys : dict or None
        Additional scalar keys to insert (e.g. ``{"distill_loss": 0.0}``).
    """
    if baseline is None:
        return
    history["task_id"].append(float(baseline["task_id"]))
    history["train_loss"].append(float(baseline.get("train_loss", float("nan"))))
    history["class_il"].append(float(baseline["class_il"]))
    history["task_il"].append(float(baseline["task_il"]))
    if extra_keys is not None:
        for key, default_value in extra_keys.items():
            if key in history:
                history[key].append(default_value)
    history["taskwise_class_il_matrix"].append(list(baseline["taskwise_class_il_row"]))
    history["taskwise_task_il_matrix"].append(list(baseline["taskwise_task_il_row"]))
