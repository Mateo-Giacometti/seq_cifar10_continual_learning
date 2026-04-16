from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


_STYLE_SET = False


def set_plot_style() -> None:
    global _STYLE_SET
    if _STYLE_SET:
        return
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#FCFCFC",
            "axes.edgecolor": "#333333",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
        }
    )
    _STYLE_SET = True


def _save_figure(save_path: Optional[str | Path], fig: plt.Figure) -> None:
    if save_path is None:
        return
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")


def subsample_for_viz(
    features: torch.Tensor,
    labels: torch.Tensor,
    max_points: int = 1500,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    n = features.size(0)
    if n <= max_points:
        return features, labels
    generator = torch.Generator(device=features.device)
    generator.manual_seed(seed)
    idx = torch.randperm(n, generator=generator, device=features.device)[:max_points]
    return features[idx], labels[idx]


def reduce_to_2d(
    features: torch.Tensor, random_state: int = 42
) -> Tuple[np.ndarray, str]:
    x = features.detach().cpu().numpy()
    try:
        import umap.umap_ as umap

        reducer = umap.UMAP(n_components=2, random_state=random_state)
        return reducer.fit_transform(x), "UMAP"
    except Exception:
        pass

    try:
        from sklearn.manifold import TSNE

        perplexity = min(30, max(5, x.shape[0] // 20))
        reducer = TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
            random_state=random_state,
        )
        return reducer.fit_transform(x), "t-SNE"
    except Exception:
        pass

    centered = features - features.mean(dim=0, keepdim=True)
    _, _, v = torch.pca_lowrank(centered, q=2)
    coords = (centered @ v[:, :2]).detach().cpu().numpy()
    return coords, "PCA"


def _moving_average(values: Sequence[float], window: int) -> np.ndarray:
    if window <= 1 or len(values) == 0:
        return np.array(values, dtype=float)
    kernel = np.ones(window, dtype=float) / float(window)
    padded = np.pad(np.array(values, dtype=float), (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def plot_supcon_loss(
    losses: List[float],
    save_path: Optional[str | Path] = None,
    smooth_window: int = 5,
) -> None:
    set_plot_style()
    if len(losses) == 0:
        print("No SupCon losses to plot.")
        return

    x = np.arange(1, len(losses) + 1)
    smooth = _moving_average(losses, smooth_window)
    best_idx = int(np.argmin(losses))

    fig = plt.figure(figsize=(8, 4.6))
    plt.plot(x, losses, alpha=0.35, label="Loss (raw)", color="#4C78A8")
    plt.plot(x, smooth, label=f"Loss (MA{smooth_window})", color="#1F4E79")
    plt.scatter([x[best_idx]], [losses[best_idx]], color="#E45756", zorder=5, label="Best")
    plt.annotate(
        f"min={losses[best_idx]:.4f}",
        (x[best_idx], losses[best_idx]),
        textcoords="offset points",
        xytext=(8, 8),
    )
    plt.title("SupCon pretraining loss (Task 0)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.xlim(1, len(losses))
    plt.legend()
    _save_figure(save_path, fig)
    plt.show()


def plot_embedding_snapshots(
    snapshots: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    task_classes: List[int],
    class_names: List[str],
    seed: int,
    save_path: Optional[str | Path] = None,
) -> None:
    set_plot_style()
    stages = ["inicio", "mitad", "final"]
    available_stages = [s for s in stages if s in snapshots]
    if len(available_stages) == 0:
        print("No snapshots available for embedding visualization.")
        return

    fig, axes = plt.subplots(1, len(available_stages), figsize=(6 * len(available_stages), 5.2))
    if len(available_stages) == 1:
        axes = [axes]

    class_to_local = {c: i for i, c in enumerate(task_classes)}
    bounds: List[np.ndarray] = []

    for stage in available_stages:
        feat, _ = snapshots[stage]
        feat, _ = subsample_for_viz(feat, snapshots[stage][1], max_points=1500, seed=seed)
        coords, _ = reduce_to_2d(feat, random_state=seed)
        bounds.append(np.array([coords[:, 0].min(), coords[:, 0].max(), coords[:, 1].min(), coords[:, 1].max()]))

    all_bounds = np.array(bounds)
    x_min = float(np.min(all_bounds[:, 0]))
    x_max = float(np.max(all_bounds[:, 1]))
    y_min = float(np.min(all_bounds[:, 2]))
    y_max = float(np.max(all_bounds[:, 3]))

    for ax, stage in zip(axes, available_stages):
        feat, lab = snapshots[stage]
        feat, lab = subsample_for_viz(feat, lab, max_points=1500, seed=seed)
        coords, method = reduce_to_2d(feat, random_state=seed)
        local_labels = np.array([class_to_local[int(y)] for y in lab.detach().cpu().numpy()])

        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=local_labels,
            cmap="tab10",
            s=12,
            alpha=0.72,
        )
        ax.set_title(f"Embeddings {stage} ({method})")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    handles = []
    for i, class_name in enumerate(class_names):
        handle = plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=plt.cm.tab10(i),
            markersize=8,
            label=class_name,
        )
        handles.append(handle)
    fig.legend(handles=handles, loc="upper center", ncol=max(1, len(class_names)))

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    _save_figure(save_path, fig)
    plt.show()


def plot_linear_history(
    history: Dict[str, List[float]],
    title: str = "Linear head accuracy",
    xlabel: str = "Epoch",
    ylabel: str = "Accuracy (%)",
    save_path: Optional[str | Path] = None,
) -> None:
    set_plot_style()
    if "train_acc" not in history or "test_acc" not in history:
        print("Linear history missing train_acc/test_acc keys.")
        return

    train_acc = history["train_acc"]
    test_acc = history["test_acc"]
    x = np.arange(1, len(train_acc) + 1)
    best_idx = int(np.argmax(test_acc))

    fig = plt.figure(figsize=(8, 4.6))
    plt.plot(x, train_acc, label="Train", color="#4C78A8")
    plt.plot(x, test_acc, label="Test", color="#E45756")
    plt.fill_between(x, train_acc, test_acc, color="#72B7B2", alpha=0.15, label="Gap")
    plt.scatter([x[best_idx]], [test_acc[best_idx]], color="#E45756", zorder=5)
    plt.annotate(
        f"best={test_acc[best_idx]:.2f}%",
        (x[best_idx], test_acc[best_idx]),
        textcoords="offset points",
        xytext=(8, 8),
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(0, 100)
    plt.legend()
    _save_figure(save_path, fig)
    plt.show()


def plot_cl_metrics(
    history: Dict[str, List[float]],
    title: str = "Continual Learning metrics",
    save_path: Optional[str | Path] = None,
) -> None:
    set_plot_style()
    task_ids = [int(t) for t in history.get("task_id", [])]
    if len(task_ids) == 0:
        print("No continual-learning history to plot.")
        return

    x = [task_id + 1 for task_id in task_ids]
    class_il = history.get("class_il", [])
    task_il = history.get("task_il", [])

    fig = plt.figure(figsize=(8.4, 4.8))
    plt.plot(x, class_il, marker="o", label="Class-IL", color="#E45756")
    plt.plot(x, task_il, marker="s", label="Task-IL", color="#4C78A8")
    if len(class_il) > 0:
        plt.annotate(f"{class_il[-1]:.2f}%", (x[-1], class_il[-1]), textcoords="offset points", xytext=(6, 6))
    if len(task_il) > 0:
        plt.annotate(f"{task_il[-1]:.2f}%", (x[-1], task_il[-1]), textcoords="offset points", xytext=(6, -14))
    plt.title(title)
    plt.xlabel("Number of tasks learned")
    plt.ylabel("Accuracy (%)")
    plt.xticks(x)
    plt.ylim(0, 100)
    plt.legend()
    _save_figure(save_path, fig)
    plt.show()


def plot_methods_over_tasks(
    histories_by_method: Mapping[str, Dict[str, List[float]]],
    save_path: Optional[str | Path] = None,
) -> None:
    set_plot_style()
    if len(histories_by_method) == 0:
        print("No method histories to compare.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    ax_class, ax_task = axes

    for method, hist in histories_by_method.items():
        task_ids = [int(t) + 1 for t in hist.get("task_id", [])]
        class_il = hist.get("class_il", [])
        task_il = hist.get("task_il", [])
        if len(task_ids) == 0:
            continue
        ax_class.plot(task_ids, class_il, marker="o", label=method)
        ax_task.plot(task_ids, task_il, marker="s", label=method)

    ax_class.set_title("Class-IL over tasks")
    ax_task.set_title("Task-IL over tasks")
    for ax in axes:
        ax.set_xlabel("Number of tasks learned")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.set_xticks(sorted({i + 1 for h in histories_by_method.values() for i in range(len(h.get("task_id", [])))}))
        ax.legend()

    plt.tight_layout()
    _save_figure(save_path, fig)
    plt.show()


def compute_forgetting_matrix(taskwise_matrix: Sequence[Sequence[float]]) -> np.ndarray:
    """Compute forgetting matrix F[t, k] = A[k, k] - A[t, k].

    Input matrix A can be rectangular (n_steps, n_tasks), with NaN for
    tasks not yet seen at each step.
    """
    arr = np.array(taskwise_matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError("taskwise_matrix must be 2-dimensional")

    n_steps, n_tasks = arr.shape
    out = np.full_like(arr, np.nan, dtype=float)

    for k in range(min(n_steps, n_tasks)):
        anchor = arr[k, k]
        if np.isnan(anchor):
            continue
        for i in range(k, n_steps):
            if np.isnan(arr[i, k]):
                continue
            out[i, k] = anchor - arr[i, k]
    return out


def plot_forgetting_curves(
    histories_by_method: Mapping[str, Dict[str, List[float]]],
    metric: str = "class_il",
    save_path: Optional[str | Path] = None,
) -> None:
    set_plot_style()
    if metric not in {"class_il", "task_il"}:
        raise ValueError("metric must be 'class_il' or 'task_il'")
    if len(histories_by_method) == 0:
        print("No method histories to plot forgetting.")
        return

    key = "taskwise_class_il_matrix" if metric == "class_il" else "taskwise_task_il_matrix"
    title_metric = "Class-IL" if metric == "class_il" else "Task-IL"

    fig = plt.figure(figsize=(9.2, 4.8))
    x: Optional[np.ndarray] = None

    for method, hist in histories_by_method.items():
        matrix = hist.get(key)
        if matrix is None:
            continue
        forgetting = compute_forgetting_matrix(matrix)
        avg_forgetting = np.nanmean(forgetting, axis=1)
        if x is None:
            x = np.arange(1, len(avg_forgetting) + 1)
        plt.plot(x, avg_forgetting, marker="o", label=method)

    if x is None:
        print(f"No histories include '{key}'.")
        return

    plt.title(f"Average forgetting over tasks ({title_metric})")
    plt.xlabel("Number of tasks learned")
    plt.ylabel("Forgetting (%)")
    plt.xticks(x)
    plt.ylim(bottom=0)
    plt.legend()
    _save_figure(save_path, fig)
    plt.show()


def plot_forgetting_by_task(
    history: Dict[str, List[float]],
    method_name: str,
    metric: str = "class_il",
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot forgetting trajectories F[t, k] for each task k in one method."""
    set_plot_style()
    if metric not in {"class_il", "task_il"}:
        raise ValueError("metric must be 'class_il' or 'task_il'")

    key = "taskwise_class_il_matrix" if metric == "class_il" else "taskwise_task_il_matrix"
    matrix = history.get(key)
    if matrix is None:
        print(f"History does not include '{key}'.")
        return

    forgetting = compute_forgetting_matrix(matrix)
    n_tasks = forgetting.shape[0]
    x = np.arange(1, n_tasks + 1)

    fig = plt.figure(figsize=(9.2, 4.8))
    plotted = 0
    for task_id in range(n_tasks):
        y = forgetting[:, task_id]
        if np.all(np.isnan(y)):
            continue
        plt.plot(x, y, marker="o", label=f"Task {task_id}")
        plotted += 1

    if plotted == 0:
        print(f"No forgetting points to plot for '{method_name}' ({metric}).")
        return

    title_metric = "Class-IL" if metric == "class_il" else "Task-IL"
    plt.title(f"Forgetting by task - {method_name} ({title_metric})")
    plt.xlabel("Number of tasks learned")
    plt.ylabel("Forgetting (%)")
    plt.xticks(x)
    plt.ylim(bottom=0)
    plt.legend(ncol=2)
    _save_figure(save_path, fig)
    plt.show()
