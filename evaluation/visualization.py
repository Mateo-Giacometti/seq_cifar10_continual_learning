"""
Visualisation utilities for Continual Learning experiments.

Provides:
  - :func:`plot_loss_curve`  – training loss over epochs
  - :func:`plot_embeddings`  – t-SNE / UMAP of backbone embeddings
  - :func:`plot_accuracy_vs_tasks` – per-task and Class-IL accuracy curves
"""

from __future__ import annotations

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Loss curve
# ---------------------------------------------------------------------------

def plot_loss_curve(
    loss_history: List[float],
    title: str = "Training Loss",
    xlabel: str = "Epoch",
    ylabel: str = "Loss",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot a 1-D list of per-epoch losses.

    Parameters
    ----------
    loss_history:
        Sequence of average epoch losses.
    title:
        Figure title.
    save_path:
        If provided, save the figure to this path (e.g. ``"imgs/loss.png"``).

    Returns
    -------
    fig:
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(loss_history) + 1), loss_history, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Embedding visualisation (t-SNE / UMAP)
# ---------------------------------------------------------------------------

def plot_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: Optional[List[str]] = None,
    method: str = "tsne",
    title: str = "Embeddings",
    save_path: Optional[str] = None,
    random_state: int = 42,
) -> plt.Figure:
    """Reduce high-dimensional embeddings to 2-D and scatter-plot them.

    Parameters
    ----------
    embeddings:
        Array of shape (N, D) – backbone output features.
    labels:
        Integer class labels of shape (N,).
    class_names:
        Optional list of string names for each unique label.
    method:
        ``"tsne"`` or ``"umap"``.
    title:
        Figure title.
    save_path:
        Optional path to save the figure.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    fig:
        Matplotlib figure object.
    """
    if method == "tsne":
        from sklearn.manifold import TSNE

        reducer = TSNE(n_components=2, random_state=random_state, perplexity=30)
        coords = reducer.fit_transform(embeddings)
    elif method == "umap":
        try:
            import umap  # type: ignore
        except ImportError as e:
            raise ImportError("Install umap-learn to use UMAP: pip install umap-learn") from e
        reducer = umap.UMAP(n_components=2, random_state=random_state)
        coords = reducer.fit_transform(embeddings)
    else:
        raise ValueError(f"Unknown reduction method '{method}'. Use 'tsne' or 'umap'.")

    unique_labels = sorted(set(labels.tolist()))
    cmap = plt.cm.get_cmap("tab10", len(unique_labels))

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        name = class_names[lbl] if class_names is not None else str(lbl)
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=8,
            alpha=0.6,
            color=cmap(i),
            label=name,
        )

    ax.legend(markerscale=3, fontsize=8, loc="best")
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Accuracy vs tasks
# ---------------------------------------------------------------------------

def plot_accuracy_vs_tasks(
    results: Dict[str, List],
    title: str = "Accuracy vs. Number of Tasks Learned",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot Class-IL and (optionally) Task-IL accuracy after each task.

    Parameters
    ----------
    results:
        Dict as returned by :meth:`training.trainer.CLTrainer.run`.
        Expected keys: ``"class_il"`` (List[float]) and optionally
        ``"task_il"`` (List[List[float]]) for average Task-IL per step.
    title:
        Figure title.
    save_path:
        Optional path to save the figure.

    Returns
    -------
    fig:
        Matplotlib figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    n_steps = len(results["class_il"])
    x = list(range(1, n_steps + 1))

    ax.plot(x, results["class_il"], marker="o", linewidth=2, label="Class-IL")

    if "task_il" in results:
        avg_task_il = [float(np.mean(row)) for row in results["task_il"]]
        ax.plot(x, avg_task_il, marker="s", linewidth=2, linestyle="--", label="Task-IL (avg)")

    ax.set_xticks(x)
    ax.set_xlabel("Tasks Learned")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def plot_comparison_table(
    method_results: Dict[str, Dict],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Render a table comparing all four CL methods.

    Parameters
    ----------
    method_results:
        ``{ method_name: {"class_il": [...], "task_il": [...]} }``
    save_path:
        Optional save path.
    """
    methods = list(method_results.keys())
    class_il_final = [method_results[m]["class_il"][-1] for m in methods]
    task_il_final = [
        float(np.mean(method_results[m]["task_il"][-1]))
        for m in methods
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(methods))
    width = 0.35

    ax.bar(x - width / 2, task_il_final, width, label="Task-IL (avg last)")
    ax.bar(x + width / 2, class_il_final, width, label="Class-IL")

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Accuracy")
    ax.set_title("Final Accuracy Comparison")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6, axis="y")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
