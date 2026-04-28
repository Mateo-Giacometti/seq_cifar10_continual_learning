from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def _save_figure(save_path: Optional[str | Path], fig: plt.Figure) -> None:
    """
    Saves a figure to the specified path.

    Parameters
    ----------
    save_path : Optional[str | Path]
        The path to save the figure to.
    fig : plt.Figure
        The figure to save.
    """
    if save_path is None:
        return
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")


def subsample_for_viz(
    features: torch.Tensor,
    labels: torch.Tensor,
    max_points: int = 2000,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Subsamples features and labels for visualization.

    Parameters
    ----------
    features : torch.Tensor
        The features to subsample.
    labels : torch.Tensor
        The labels to subsample.
    max_points : int, optional
        The maximum number of points to subsample, by default 2000.
    seed : int, optional
        The seed for the random number generator, by default 42.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        The subsampled features and labels.
    """
    n = features.size(0)
    if n <= max_points:
        return features, labels
    generator = torch.Generator(device=features.device)
    generator.manual_seed(seed)
    idx = torch.randperm(n, generator=generator, device=features.device)[:max_points]
    return features[idx], labels[idx]


def reduce_to_2d(
    features: torch.Tensor,
    method: Literal["auto", "umap", "tsne", "pca"] = "auto",
    random_state: int = 42,
) -> Tuple[np.ndarray, str]:
    """
    Reduces features to 2D using dimensionality reduction.

    Parameters
    ----------
    features : torch.Tensor
        The features to reduce.
    method : Literal["auto", "umap", "tsne", "pca"], optional
        The dimensionality reduction method, by default "auto".
    random_state : int, optional
        The seed for the random number generator, by default 42.

    Returns
    -------
    Tuple[np.ndarray, str]
        The reduced features and the method used.
    """
    x = features.detach().cpu().numpy()

    if method in ["auto", "umap"]:
        try:
            import umap.umap_ as umap

            reducer = umap.UMAP(n_components=2, random_state=random_state)
            return reducer.fit_transform(x), "UMAP"
        except Exception:
            if method == "umap":
                print("UMAP no disponible, cayendo a t-SNE...")
            pass

    if method in ["auto", "tsne"] or (method == "umap"):
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
            if method == "tsne":
                print("t-SNE no disponible, cayendo a PCA...")
            pass

    centered = features - features.mean(dim=0, keepdim=True)
    _, _, v = torch.pca_lowrank(centered, q=2)
    coords = (centered @ v[:, :2]).detach().cpu().numpy()
    return coords, "PCA"


def _moving_average(values: Sequence[float], window: int) -> np.ndarray:
    """
    Computes a moving average of the given values.

    Parameters
    ----------
    values : Sequence[float]
        The values to compute the moving average of.
    window : int
        The size of the moving average window.

    Returns
    -------
    np.ndarray
        The moving average of the values.
    """
    if window <= 1 or len(values) == 0:
        return np.array(values, dtype=float)
    kernel = np.ones(window, dtype=float) / float(window)
    padded = np.pad(np.array(values, dtype=float), (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def plot_supcon_loss(
    losses: List[float], save_path: Optional[str | Path] = None, smooth_window: int = 5
) -> None:
    """
    Plots the SupCon loss over epochs.

    Parameters
    ----------
    losses : List[float]
        The SupCon losses to plot.
    save_path : Optional[str | Path], optional
        The path to save the figure to, by default None.
    smooth_window : int, optional
        The size of the moving average window, by default 5.

    Returns
    -------
    None
    """
    if len(losses) == 0:
        print("No SupCon losses to plot.")
        return

    x = np.arange(1, len(losses) + 1)

    fig = plt.figure(figsize=(12, 8))
    plt.plot(x, losses, alpha=1.0, linewidth=3.0, color="darkgreen")
    plt.xlabel(r"Epoch", fontsize=16)
    plt.ylabel(
        r"$\mathcal{L}^{\text{sup}} = \sum_{i=1}^{2N} \frac{-1}{|\mathcal{P}_i|} \sum_{p \in \mathcal{P}_i} \log \frac{\exp(z_i \cdot z_p / \tau)}{\sum_{k \neq i} \exp(z_i \cdot z_k / \tau)}$",
        fontsize=16,
    )
    plt.xlim(1, len(losses))
    plt.tick_params(axis="both", which="major", labelsize=14)
    plt.grid(alpha=0.3)
    _save_figure(save_path, fig)
    plt.title("SupCon Pre-training Loss (over Task 0)", fontsize=22, fontweight="bold")
    plt.show()


def plot_embedding_snapshots(
    snapshots: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    task_classes: List[int],
    class_names: List[str],
    seed: int,
    reduction_method: Literal["auto", "umap", "tsne", "pca"] = "auto",
    save_path: Optional[str | Path] = None,
) -> None:
    """
    Plots snapshots of embeddings in latent space using dimensionality reduction.

    Parameters
    ----------
    snapshots : Dict[str, Tuple[torch.Tensor, torch.Tensor]]
        The snapshots of embeddings to plot.
    task_classes : List[int]
        The task classes to plot.
    class_names : List[str]
        The class names to plot.
    seed : int
        The seed for the random number generator.
    reduction_method : Literal["auto", "umap", "tsne", "pca"], optional
        The dimensionality reduction method, by default "auto".
    save_path : Optional[str | Path], optional
        The path to save the figure to, by default None.

    Returns
    -------
    None
    """
    stages = ["inicio", "mitad", "final"]
    available_stages = [s for s in stages if s in snapshots]
    if len(available_stages) == 0:
        print("No snapshots available for embedding visualization.")
        return

    plot_data = {}
    bounds = []
    reduction_actual = "PCA"

    for stage in available_stages:
        feat, lab = snapshots[stage]
    
        feat, lab = subsample_for_viz(feat, lab, max_points=5000, seed=seed)
        coords, method_used = reduce_to_2d(
            feat, method=reduction_method, random_state=seed
        )
        reduction_actual = method_used

        plot_data[stage] = (coords, lab)
        bounds.append(
            [
                coords[:, 0].min(),
                coords[:, 0].max(),
                coords[:, 1].min(),
                coords[:, 1].max(),
            ]
        )

    bounds_arr = np.array(bounds)
    x_min, x_max = float(np.min(bounds_arr[:, 0])), float(np.max(bounds_arr[:, 1]))
    y_min, y_max = float(np.min(bounds_arr[:, 2])), float(np.max(bounds_arr[:, 3]))

    x_pad = (x_max - x_min) * 0.08
    y_pad = (y_max - y_min) * 0.08

    fig, axes = plt.subplots(
        len(available_stages),
        1,
        figsize=(12, 8 * len(available_stages)),
        dpi=100,
        constrained_layout=True,
    )
    if len(available_stages) == 1:
        axes = [axes]

    class_to_local = {c: i for i, c in enumerate(task_classes)}
    cmap = plt.get_cmap("tab10")

    for ax, stage in zip(axes, available_stages):
        coords, lab = plot_data[stage]
        local_labels = np.array(
            [class_to_local[int(y)] for y in lab.detach().cpu().numpy()]
        )

        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=local_labels,
            cmap=cmap,
            vmin=0,
            vmax=9,
            s=45,
            alpha=0.8,
            edgecolors="white",
            linewidths=0.3,
            zorder=2,
        )

        if stage == "inicio":
            title = "Initial Stage"
        elif stage == "mitad":
            title = "Middle of Pre-training"
        else:
            title = "End of Pre-training"

        ax.set_title(f"\n{title}", loc="center", fontsize=22, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

        ax.set_xlabel("Latent Dimension 1", fontsize=16)
        ax.set_ylabel("Latent Dimension 2", fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=14)
    
    handles = []
    for i, class_name in enumerate(class_names):
        h = plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=class_name,
            markerfacecolor=cmap(i),
            markersize=12,
        )
        handles.append(h)

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(5, len(class_names)),
        bbox_to_anchor=(0.52, -0.04),
        fontsize=18,
        frameon=True,
        shadow=True,
    )

    _save_figure(save_path, fig)
    
    fig.suptitle(
        f"Latent Space Evolution ({reduction_actual} projection)",
        fontsize=26,
        fontweight="black",
    )

    plt.show()


def plot_linear_history(
    history: Dict[str, List[float]],
    title: str = "Linear Head Accuracy",
    xlabel: str = "Epoch",
    ylabel: str = "Accuracy (%)",
    save_path: Optional[str | Path] = None,
) -> None:
    """
    Plots the linear head accuracy over epochs.

    Parameters
    ----------
    history : Dict[str, List[float]]
        The linear head accuracy history.
    title : str, optional
        The title of the plot, by default "Linear Head Accuracy".
    xlabel : str, optional
        The x-axis label, by default "Epoch".
    ylabel : str, optional
        The y-axis label, by default "Accuracy (%)".
    save_path : Optional[str | Path], optional
        The path to save the figure to, by default None.

    Returns
    -------
    None
    """
 
    if "train_acc" not in history or "test_acc" not in history:
        print("Linear history missing train_acc/test_acc keys.")
        return

    train_acc = history["train_acc"]
    test_acc = history["test_acc"]
    x = np.arange(1, len(train_acc) + 1)

    fig = plt.figure(figsize=(12, 8))
    plt.plot(x, train_acc, label="Train", color="black")
    plt.plot(x, test_acc, label="Test", color="red")
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.ylim(0, 105)
    plt.tick_params(axis="both", which="major", labelsize=14)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=14)
    _save_figure(save_path, fig)
    plt.title(title, fontsize=22, fontweight="bold")
    plt.show()


def plot_cl_metrics(
    history: Dict[str, List[float]],
    title: str = "Continual Learning metrics",
    save_path: Optional[str | Path] = None,
) -> None:
    """
    Plots the continual learning metrics over tasks.

    Parameters
    ----------
    history : Dict[str, List[float]]
        The continual learning metrics history.
    title : str, optional
        The title of the plot, by default "Continual Learning metrics".
    save_path : Optional[str | Path], optional
        The path to save the figure to, by default None.

    Returns
    -------
    None
    """
    task_ids = [int(t) for t in history.get("task_id", [])]
    if len(task_ids) == 0:
        print("No continual-learning history to plot.")
        return

    x = [task_id + 1 for task_id in task_ids]
    class_il = history.get("class_il", [])
    task_il = history.get("task_il", [])

    fig = plt.figure(figsize=(12, 8))
    plt.plot(x, class_il, marker="o", label="Class-IL", color="red")
    plt.plot(x, task_il, marker="s", label="Task-IL", color="blue")
    
    for xi, yi in zip(x, class_il):
        plt.annotate(
            f"{yi:.2f}%",
            (xi, yi),
            textcoords="offset points",
            xytext=(0, -18),  
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="red",
        )
    for xi, yi in zip(x, task_il):
        plt.annotate(
            f"{yi:.2f}%",
            (xi, yi),
            textcoords="offset points",
            xytext=(0, 10), 
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="blue",
        )
    plt.xlabel("Number of tasks learned", fontsize=16)
    plt.ylabel("Accuracy (%)", fontsize=16)
    plt.xticks(x)
    plt.tick_params(axis="both", which="major", labelsize=14)
    plt.ylim(0, 105)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=14)
    _save_figure(save_path, fig)
    plt.title(title, fontsize=22, fontweight="bold")
    plt.show()


def plot_forgetting_metrics(
    history: Dict[str, object],
    title: str = "Average Forgetting",
    definition: str = "max_history",
    save_path: Optional[str | Path] = None,
) -> None:
    """
    Plots the average forgetting over tasks for a single method.
    
    Parameters
    ----------
    history : Dict[str, object]
        The average forgetting history.
    title : str, optional
        The title of the plot, by default "Average Forgetting".
    definition : str, optional
        The definition of forgetting, by default "max_history".
    save_path : Optional[str | Path], optional
        The path to save the figure to, by default None.

    Returns
    -------
    None
    """
    task_ids = [int(t) for t in history.get("task_id", [])]
    if len(task_ids) == 0:
        print("No history to plot forgetting.")
        return

    x = [task_id + 1 for task_id in task_ids]
    
    matrix_class = history.get("taskwise_class_il_matrix")
    avg_f_class = None
    if matrix_class is not None:
        f_matrix = compute_forgetting_matrix(np.array(matrix_class), definition=definition)
        avg_f_class = compute_average_over_past(f_matrix)

    matrix_task = history.get("taskwise_task_il_matrix")
    avg_f_task = None
    if matrix_task is not None:
        f_matrix_task = compute_forgetting_matrix(np.array(matrix_task), definition=definition)
        avg_f_task = compute_average_over_past(f_matrix_task)

    if avg_f_class is None and avg_f_task is None:
        print("No taskwise matrices found in history to compute forgetting.")
        return

    fig = plt.figure(figsize=(12, 8))
    
    if avg_f_class is not None:
        plt.plot(x, avg_f_class, marker="o", label="Class-IL Forgetting", color="red", linewidth=2.5)
        
        for xi, yi in zip(x, avg_f_class):
            if not np.isnan(yi):
                plt.annotate(
                    f"{yi:.2f}%",
                    (xi, yi),
                    textcoords="offset points",
                    xytext=(0, -18),
                    ha="center",
                    fontsize=10,
                    fontweight="bold",
                    color="red",
                )

    if avg_f_task is not None:
        plt.plot(x, avg_f_task, marker="s", label="Task-IL Forgetting", color="blue", linewidth=2.5)
        
        for xi, yi in zip(x, avg_f_task):
            if not np.isnan(yi):
                plt.annotate(
                    f"{yi:.2f}%",
                    (xi, yi),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=10,
                    fontweight="bold",
                    color="blue",
                )

    plt.xlabel("Number of tasks learned", fontsize=16)
    plt.ylabel("Forgetting (%)", fontsize=16)
    plt.xticks(x)
    plt.tick_params(axis="both", which="major", labelsize=14)
    plt.grid(True, linestyle='-', alpha=0.3)
    plt.legend(fontsize=14)
    
    all_vals = []
    if avg_f_class is not None: all_vals.extend(avg_f_class[~np.isnan(avg_f_class)])
    if avg_f_task is not None: all_vals.extend(avg_f_task[~np.isnan(avg_f_task)])
    if all_vals:
        ymin = min(0, min(all_vals))
        ymax = max(all_vals) + 2
        plt.ylim(ymin, ymax)

    _save_figure(save_path, fig)
    plt.title(title, fontsize=22, fontweight="bold")
    plt.show()


def plot_bwt_metrics(
    history: Dict[str, object],
    title: str = "Average BWT",
    save_path: Optional[str | Path] = None,
) -> None:
    """
    Plots the average BWT over tasks for a single method.
    
    Parameters
    ----------
    history : Dict[str, object]
        The average BWT history.
    title : str, optional
        The title of the plot, by default "Average BWT".
    save_path : Optional[str | Path], optional
        The path to save the figure to, by default None.

    Returns
    -------
    None
    """
    task_ids = [int(t) for t in history.get("task_id", [])]
    if len(task_ids) == 0:
        print("No history to plot BWT.")
        return

    x = [task_id + 1 for task_id in task_ids]
    
    matrix_class = history.get("taskwise_class_il_matrix")
    avg_b_class = None
    if matrix_class is not None:
        b_matrix = compute_bwt_matrix(np.array(matrix_class))
        avg_b_class = compute_average_over_past(b_matrix)

    matrix_task = history.get("taskwise_task_il_matrix")
    avg_b_task = None
    if matrix_task is not None:
        b_matrix_task = compute_bwt_matrix(np.array(matrix_task))
        avg_b_task = compute_average_over_past(b_matrix_task)

    if avg_b_class is None and avg_b_task is None:
        print("No taskwise matrices found in history to compute BWT.")
        return

    fig = plt.figure(figsize=(12, 8))
    
    if avg_b_class is not None:
        plt.plot(x, avg_b_class, marker="o", label="Class-IL BWT", color="red", linewidth=2.5)
        
        for xi, yi in zip(x, avg_b_class):
            if not np.isnan(yi):
                plt.annotate(
                    f"{yi:.2f}%",
                    (xi, yi),
                    textcoords="offset points",
                    xytext=(0, -18),
                    ha="center",
                    fontsize=10,
                    fontweight="bold",
                    color="red",
                )

    if avg_b_task is not None:
        plt.plot(x, avg_b_task, marker="s", label="Task-IL BWT", color="blue", linewidth=2.5)
            
        for xi, yi in zip(x, avg_b_task):
            if not np.isnan(yi):
                plt.annotate(
                    f"{yi:.2f}%",
                    (xi, yi),
                    textcoords="offset points",
                    xytext=(0, 10),
                    ha="center",
                    fontsize=10,
                    fontweight="bold",
                    color="blue",
                )

    plt.axhline(0, color="black", linestyle="--", alpha=0.5)
    plt.xlabel("Number of tasks learned", fontsize=16)
    plt.ylabel("BWT (%)", fontsize=16)
    plt.xticks(x)
    plt.tick_params(axis="both", which="major", labelsize=14)
    plt.grid(True, linestyle='-', alpha=0.3)
    plt.legend(fontsize=14)
    
    all_vals = []
    if avg_b_class is not None: all_vals.extend(avg_b_class[~np.isnan(avg_b_class)])
    if avg_b_task is not None: all_vals.extend(avg_b_task[~np.isnan(avg_b_task)])
    if all_vals:
        ymin = min(0, min(all_vals)) - 1
        ymax = max(0, max(all_vals)) + 1
        plt.ylim(ymin, ymax)

    _save_figure(save_path, fig)
    plt.title(title, fontsize=22, fontweight="bold")
    plt.show()


def plot_methods_comparison_class_il(
    histories_by_method: Mapping[str, Dict[str, List[float]]],
    save_path: Optional[str | Path] = None,
) -> None:
    """
    Plots the Class-IL accuracy comparison across different methods.

    Parameters
    ----------
    histories_by_method : Mapping[str, Dict[str, List[float]]]
        The Class-IL accuracy history for each method.
    save_path : Optional[str | Path], optional
        The path to save the figure to, by default None.

    Returns
    -------
    None
    """
    
    if len(histories_by_method) == 0:
        print("No method histories to compare.")
        return

    markers = ["o", "s", "D", "^", "v", "*", "X", "p"]
    
    fig = plt.figure(figsize=(12, 8))
    for i, (method, hist) in enumerate(histories_by_method.items()):
        task_ids = [int(t) + 1 for t in hist.get("task_id", [])]
        class_il = hist.get("class_il", [])
        if len(task_ids) == 0:
            continue
        
        marker = markers[i % len(markers)]
        plt.plot(task_ids, class_il, marker=marker, linewidth=2.5, label=method)
        
    plt.xlabel("Number of tasks learned", fontsize=16)
    plt.ylabel("Accuracy (%)", fontsize=16) 
    plt.tick_params(axis="both", which="major", labelsize=14)
    plt.ylim(0, 105)
    
    all_tasks = {i + 1 for h in histories_by_method.values() for i in range(len(h.get("task_id", [])))}
    if all_tasks:
        plt.xticks(sorted(all_tasks))
    
    plt.legend(fontsize=14)
    plt.grid(True, linestyle='-', alpha=0.3)
    _save_figure(save_path, fig)
    plt.title("Class-IL Comparison", fontsize=22, fontweight="bold")
    plt.show()


def plot_methods_comparison_task_il(
    histories_by_method: Mapping[str, Dict[str, List[float]]],
    save_path: Optional[str | Path] = None,
) -> None:
    """
    Plots the Task-IL accuracy comparison across different methods.

    Parameters
    ----------
    histories_by_method : Mapping[str, Dict[str, List[float]]]
        The Task-IL accuracy history for each method.
    save_path : Optional[str | Path], optional
        The path to save the figure to, by default None.

    Returns
    -------
    None
    """
    
    if len(histories_by_method) == 0:
        print("No method histories to compare.")
        return

    markers = ["o", "s", "D", "^", "v", "*", "X", "p"]
    fig = plt.figure(figsize=(12, 8))
    for i, (method, hist) in enumerate(histories_by_method.items()):
        task_ids = [int(t) + 1 for t in hist.get("task_id", [])]
        task_il = hist.get("task_il", [])
        if len(task_ids) == 0:
            continue
        
        marker = markers[i % len(markers)]
        plt.plot(task_ids, task_il, marker=marker, linewidth=2.5, label=method)
        
    plt.xlabel("Number of tasks learned", fontsize=16)
    plt.ylabel("Accuracy (%)", fontsize=16)
    plt.tick_params(axis="both", which="major", labelsize=14)
    plt.ylim(0, 105)
    
    all_tasks = {i + 1 for h in histories_by_method.values() for i in range(len(h.get("task_id", [])))}
    if all_tasks:
        plt.xticks(sorted(all_tasks))
        
    plt.legend(fontsize=14)
    plt.grid(True, linestyle='-', alpha=0.3)
    _save_figure(save_path, fig)
    plt.title("Task-IL Comparison", fontsize=22, fontweight="bold")
    plt.show()


def _as_taskwise_array(taskwise_matrix: Sequence[Sequence[float]]) -> np.ndarray:
    """
    Converts a taskwise matrix to a NumPy array.
    
    Parameters
    ----------
    taskwise_matrix : Sequence[Sequence[float]]
        The taskwise matrix to convert.

    Returns
    -------
    np.ndarray
        The converted taskwise matrix.

    Raises
    -------
    ValueError
        If the taskwise matrix is not 2-dimensional.
    """
    arr = np.array(taskwise_matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError("taskwise_matrix must be 2-dimensional")
    return arr


def _first_seen_indices(arr: np.ndarray) -> List[Optional[int]]:
    """
    Computes the first seen indices for each task in a taskwise matrix.
    
    Parameters
    ----------
    arr : np.ndarray
        The taskwise matrix.

    Returns
    -------
    List[Optional[int]]
        The first seen indices for each task.
    """
    first_seen: List[Optional[int]] = []
    for k in range(arr.shape[1]):
        seen = np.where(~np.isnan(arr[:, k]))[0]
        first_seen.append(int(seen[0]) if seen.size > 0 else None)
    return first_seen


def compute_forgetting_matrix(
    taskwise_matrix: Sequence[Sequence[float]],
    definition: str = "max_history",
) -> np.ndarray:
    """
    Computes the forgetting matrix from taskwise accuracies.

    Parameters
    ----------
    taskwise_matrix : Sequence[Sequence[float]]
        The taskwise matrix to compute the forgetting matrix from.
    definition : str, optional
        The definition of forgetting to use, by default "max_history".

    Returns
    -------
    np.ndarray
        The forgetting matrix.

    Raises
    -------
    ValueError
        If the definition is not 'max_history' or 'diagonal'.
    """
    if definition not in {"max_history", "diagonal"}:
        raise ValueError("definition must be 'max_history' or 'diagonal'")

    arr = _as_taskwise_array(taskwise_matrix)
    n_steps, n_tasks = arr.shape
    out = np.full((n_steps, n_tasks), np.nan, dtype=float)
    first_seen = _first_seen_indices(arr)

    for k in range(n_tasks):
        t0 = first_seen[k]
        if t0 is None:
            continue
        anchor0 = arr[t0, k]
        if np.isnan(anchor0):
            continue

        for t in range(t0 + 1, n_steps):
            current = arr[t, k]
            if np.isnan(current):
                continue

            if definition == "diagonal":
                anchor = anchor0
            else:
                history_values = arr[t0:t, k]
                history_values = history_values[~np.isnan(history_values)]
                if history_values.size == 0:
                    continue
                anchor = float(np.max(history_values))

            out[t, k] = anchor - current

    return out


def compute_bwt_matrix(taskwise_matrix: Sequence[Sequence[float]]) -> np.ndarray:
    """
    Computes the BWT matrix B[t, k] = A[t, k] - A[t0(k), k].

    Parameters
    ----------
    taskwise_matrix : Sequence[Sequence[float]]
        The taskwise matrix to compute the BWT matrix from.

    Returns
    -------
    np.ndarray
        The BWT matrix.

    Raises
    -------
    ValueError
        If the taskwise matrix is not 2-dimensional.
    """
    arr = _as_taskwise_array(taskwise_matrix)
    n_steps, n_tasks = arr.shape
    out = np.full((n_steps, n_tasks), np.nan, dtype=float)
    first_seen = _first_seen_indices(arr)

    for k in range(n_tasks):
        t0 = first_seen[k]
        if t0 is None:
            continue
        anchor = arr[t0, k]
        if np.isnan(anchor):
            continue

        for t in range(t0 + 1, n_steps):
            current = arr[t, k]
            if np.isnan(current):
                continue
            out[t, k] = current - anchor

    return out


def compute_average_over_past(metric_matrix: Sequence[Sequence[float]]) -> np.ndarray:
    """
    Average a taskwise metric matrix over defined past-task entries per step.
    
    Parameters
    ----------
    metric_matrix : Sequence[Sequence[float]]
        The taskwise metric matrix to average over past tasks.

    Returns
    -------
    np.ndarray
        The average over past tasks.
    """
    arr = _as_taskwise_array(metric_matrix)
    avg = np.full(arr.shape[0], np.nan, dtype=float)
    for t in range(arr.shape[0]):
        valid = arr[t, ~np.isnan(arr[t])]
        if valid.size > 0:
            avg[t] = float(np.mean(valid))
    return avg


def compute_final_transfer_metrics(
    taskwise_matrix: Sequence[Sequence[float]],
    forgetting_definition: str = "max_history",
) -> Dict[str, object]:
    """
    Computes the final transfer metrics from a taskwise accuracy matrix.

    Parameters
    ----------
    taskwise_matrix : Sequence[Sequence[float]]
        The taskwise accuracy matrix.
    forgetting_definition : str, optional
        The definition of forgetting to use, by default "max_history".

    Returns
    -------
    Dict[str, object]
        The final transfer metrics.

    Raises
    -------
    ValueError
        If the taskwise matrix is not 2-dimensional.
    """
    forgetting_matrix = compute_forgetting_matrix(
        taskwise_matrix,
        definition=forgetting_definition,
    )
    bwt_matrix = compute_bwt_matrix(taskwise_matrix)

    avg_forgetting = compute_average_over_past(forgetting_matrix)
    avg_bwt = compute_average_over_past(bwt_matrix)

    final_idx = len(avg_forgetting) - 1
    final_forgetting = (
        float(avg_forgetting[final_idx]) if final_idx >= 0 else float("nan")
    )
    final_bwt = float(avg_bwt[final_idx]) if final_idx >= 0 else float("nan")

    return {
        "forgetting_matrix": forgetting_matrix,
        "bwt_matrix": bwt_matrix,
        "avg_forgetting_per_step": avg_forgetting,
        "avg_bwt_per_step": avg_bwt,
        "avg_forgetting_final": final_forgetting,
        "avg_bwt_final": final_bwt,
    }


def plot_forgetting_curves(
    histories_by_method: Mapping[str, Dict[str, object]],
    metric: str = "class_il",
    definition: str = "max_history",
    save_path: Optional[str | Path] = None,
) -> None:
    """
    Plots the forgetting curves over tasks for different methods.

    Parameters
    ----------
    histories_by_method : Mapping[str, Dict[str, object]]
        The forgetting history for each method.
    metric : str, optional
        The metric to use, by default "class_il".
    definition : str, optional
        The definition of forgetting to use, by default "max_history".
    save_path : Optional[str | Path], optional
        The path to save the figure to, by default None.

    Returns
    -------
    None
    """
    
    if metric not in {"class_il", "task_il"}:
        raise ValueError("metric must be 'class_il' or 'task_il'")
    if len(histories_by_method) == 0:
        print("No method histories to plot forgetting.")
        return

    key = (
        "taskwise_class_il_matrix"
        if metric == "class_il"
        else "taskwise_task_il_matrix"
    )
    title_metric = "Class-IL" if metric == "class_il" else "Task-IL"
    markers = ["o", "s", "D", "^", "v", "*", "X", "p"]

    fig = plt.figure(figsize=(12, 8))
    x: Optional[np.ndarray] = None
    y_values: List[np.ndarray] = []

    for i, (method, hist) in enumerate(histories_by_method.items()):
        matrix = hist.get(key)
        if matrix is None:
            continue
        forgetting = compute_forgetting_matrix(matrix, definition=definition)
        avg_forgetting = compute_average_over_past(forgetting)
        if np.all(np.isnan(avg_forgetting)):
            continue
        if x is None:
            x = np.arange(1, len(avg_forgetting) + 1)
        
        marker = markers[i % len(markers)]
        plt.plot(x, avg_forgetting, marker=marker, label=method)
        y_values.append(avg_forgetting)

    if x is None:
        print(f"No histories include '{key}'.")
        return

    label_definition = "max-history" if definition == "max_history" else "diagonal"
    plt.xlabel("Number of tasks learned", fontsize=16)
    plt.ylabel("Forgetting (%)", fontsize=16)
    plt.tick_params(axis="both", which="major", labelsize=14)
    plt.xticks(x)
    if len(y_values) > 0:
        all_vals = np.concatenate(
            [v[~np.isnan(v)] for v in y_values if np.any(~np.isnan(v))]
        )
        if all_vals.size > 0 and np.nanmin(all_vals) < 0:
            ymin = float(np.nanmin(all_vals)) - 0.5
            ymax = float(np.nanmax(all_vals)) + 0.5
            plt.ylim(ymin, ymax)
        else:
            plt.ylim(bottom=0)
    plt.legend(fontsize=14)
    plt.grid(True, linestyle='-', alpha=0.3)
    _save_figure(save_path, fig)
    plt.title(f"Average forgetting over tasks ({title_metric}, {label_definition})", fontsize=22, fontweight="bold")
    plt.show()


def plot_bwt_curves(
    histories_by_method: Mapping[str, Dict[str, object]],
    metric: str = "class_il",
    save_path: Optional[str | Path] = None,
) -> None:
    """
    Plots the BWT curves over tasks for different methods.

    Parameters
    ----------
    histories_by_method : Mapping[str, Dict[str, object]]
        The BWT history for each method.
    metric : str, optional
        The metric to use, by default "class_il".
    save_path : Optional[str | Path], optional
        The path to save the figure to, by default None.

    Returns
    -------
    None
    """
    
    if metric not in {"class_il", "task_il"}:
        raise ValueError("metric must be 'class_il' or 'task_il'")
    if len(histories_by_method) == 0:
        print("No method histories to plot BWT.")
        return

    key = (
        "taskwise_class_il_matrix"
        if metric == "class_il"
        else "taskwise_task_il_matrix"
    )
    title_metric = "Class-IL" if metric == "class_il" else "Task-IL"

    fig = plt.figure(figsize=(12, 8))
    markers = ["o", "s", "D", "^", "v", "*", "X", "p"]
    x: Optional[np.ndarray] = None

    for i, (method, hist) in enumerate(histories_by_method.items()):
        matrix = hist.get(key)
        if matrix is None:
            continue
        bwt = compute_bwt_matrix(matrix)
        avg_bwt = compute_average_over_past(bwt)
        if np.all(np.isnan(avg_bwt)):
            continue
        if x is None:
            x = np.arange(1, len(avg_bwt) + 1)
        
        marker = markers[i % len(markers)]
        plt.plot(x, avg_bwt, marker=marker, label=method)

    if x is None:
        print(f"No histories include '{key}'.")
        return

    plt.xlabel("Number of tasks learned", fontsize=16)
    plt.ylabel("BWT (%)", fontsize=16)
    plt.tick_params(axis="both", which="major", labelsize=14)
    plt.xticks(x)
    plt.legend(fontsize=14)
    plt.grid(True, linestyle='-', alpha=0.3)
    _save_figure(save_path, fig)
    plt.title(f"Average BWT over tasks ({title_metric})", fontsize=22, fontweight="bold")
    plt.show()


def plot_forgetting_by_task(
    history: Dict[str, object],
    method_name: str,
    metric: str = "class_il",
    definition: str = "max_history",
    save_path: Optional[str | Path] = None,
) -> None:
    """
    Plots the forgetting trajectories F[t, k] for each task k in one method.

    Parameters
    ----------
    history : Dict[str, object]
        The forgetting history for the method.
    method_name : str
        The name of the method.
    metric : str, optional
        The metric to use, by default "class_il".
    definition : str, optional
        The definition of forgetting to use, by default "max_history".
    save_path : Optional[str | Path], optional
        The path to save the figure to, by default None.

    Returns
    -------
    None
    """
    
    if metric not in {"class_il", "task_il"}:
        raise ValueError("metric must be 'class_il' or 'task_il'")

    key = (
        "taskwise_class_il_matrix"
        if metric == "class_il"
        else "taskwise_task_il_matrix"
    )
    matrix = history.get(key)
    if matrix is None:
        print(f"History does not include '{key}'.")
        return

    forgetting = compute_forgetting_matrix(matrix, definition=definition)
    n_steps, n_tasks = forgetting.shape
    x = np.arange(1, n_steps + 1)

    fig = plt.figure(figsize=(12, 8))
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
    label_definition = "max-history" if definition == "max_history" else "diagonal"
    plt.title(
        f"Forgetting by task - {method_name} ({title_metric}, {label_definition})"
    )
    plt.xlabel("Number of tasks learned", fontsize=16)
    plt.ylabel("Forgetting (%)", fontsize=16)
    plt.xticks(x)
    plt.tick_params(axis="both", which="major", labelsize=14)
    plt.ylim(bottom=0)
    plt.legend(ncol=2, fontsize=14)
    plt.grid(True, linestyle='-', alpha=0.3)
    _save_figure(save_path, fig)
    plt.show()


def plot_taskwise_heatmap(
    taskwise_matrix: Sequence[Sequence[float]],
    method_name: str,
    metric: str = "class_il",
    task_classes: Optional[List[List[int]]] = None,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str | Path] = None,
) -> None:
    """
    Plots a heatmap of the taskwise accuracy matrix for a given method.

    Parameters
    ----------
    taskwise_matrix : Sequence[Sequence[float]]
        The taskwise accuracy matrix.
    method_name : str
        The name of the method.
    metric : str, optional
        The metric to use, by default "class_il".
    task_classes : Optional[List[List[int]]], optional
        The classes for each task, by default None.
    class_names : Optional[List[str]], optional
        The names of the classes, by default None.
    save_path : Optional[str | Path], optional
        The path to save the figure to, by default None.

    Returns
    -------
    None
    """
    arr = _as_taskwise_array(taskwise_matrix)
    n_steps, n_tasks = arr.shape
    title_metric = "Class-IL" if metric == "class_il" else "Task-IL"

    fig, ax = plt.subplots(figsize=(max(12, n_tasks * 1.4), max(8, n_steps * 0.9)))

    masked = np.ma.array(arr, mask=np.isnan(arr))
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="lightgray")

    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    for t in range(n_steps):
        for k in range(n_tasks):
            val = arr[t, k]
            if np.isnan(val):
                continue
            text_color = "white" if val < 40 else "black"
            ax.text(
                k,
                t,
                f"{val:.1f}",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color=text_color,
            )

    step_labels = [f"After T{i}" for i in range(n_steps)]
    task_labels = [f"Task {k}" for k in range(n_tasks)]
    if task_classes is not None and class_names is not None:
        task_labels = []
        for k, classes in enumerate(task_classes):
            names = [
                class_names[c] if c < len(class_names) else f"c{c}" for c in classes
            ]
            task_labels.append(f"T{k}: {', '.join(names)}")

    ax.set_xticks(range(n_tasks))
    ax.set_xticklabels(task_labels, rotation=45, ha="right", fontsize= 12)
    ax.set_yticks(range(n_steps))
    ax.set_yticklabels(step_labels, fontsize=12)
    ax.set_xlabel("Evaluated task", fontsize=16)
    ax.set_ylabel("Training step", fontsize=16)
    fig.colorbar(im, ax=ax, label="Accuracy (%)", shrink=0.8)
    plt.tight_layout()
    _save_figure(save_path, fig)
    plt.title(f"Taskwise accuracy matrix - {method_name} ({title_metric})", fontsize=22, fontweight="bold")
    plt.show()


def plot_final_embeddings_comparison(
    embeddings_by_method: Mapping[str, Tuple[torch.Tensor, torch.Tensor]],
    class_names: Optional[List[str]] = None,
    seed: int = 42,
    reduction_method: Literal["auto", "umap", "tsne", "pca"] = "auto",
    save_path: Optional[str | Path] = None,
) -> None:
    """
    Plots the t-SNE/UMAP embeddings for each method, arranged vertically.

    Parameters
    ----------
    embeddings_by_method : Mapping[str, Tuple[torch.Tensor, torch.Tensor]]
        A dictionary mapping method names to tuples of (embeddings, labels).
    class_names : Optional[List[str]], optional
        The names of the classes, by default None.
    seed : int, optional
        The seed for reproducibility, by default 42.
    reduction_method : Literal["auto", "umap", "tsne", "pca"], optional
        The reduction method to use, by default "auto".
    save_path : Optional[str | Path], optional
        The path to save the figure to, by default None.

    Returns
    -------
    None
    """
    methods = list(embeddings_by_method.keys())
    if len(methods) == 0:
        print("No embeddings to plot.")
        return

    n = len(methods)

    plot_data = {}
    bounds = []
    reduction_actual = "PCA"
    all_labels_set: set = set()

    for method in methods:
        feat, lab = embeddings_by_method[method]
        all_labels_set.update(lab.detach().cpu().numpy().tolist())
        
        feat, lab = subsample_for_viz(feat, lab, max_points=5000, seed=seed)
        coords, method_used = reduce_to_2d(
            feat, method=reduction_method, random_state=seed
        )
        reduction_actual = method_used

        plot_data[method] = (coords, lab)
        bounds.append(
            [
                coords[:, 0].min(),
                coords[:, 0].max(),
                coords[:, 1].min(),
                coords[:, 1].max(),
            ]
        )

    unique_labels = sorted(all_labels_set)
    
    bounds_arr = np.array(bounds)
    x_min, x_max = float(np.min(bounds_arr[:, 0])), float(np.max(bounds_arr[:, 1]))
    y_min, y_max = float(np.min(bounds_arr[:, 2])), float(np.max(bounds_arr[:, 3]))

    x_pad = (x_max - x_min) * 0.08
    y_pad = (y_max - y_min) * 0.08

    fig, axes = plt.subplots(
        n,
        1,
        figsize=(12, 8 * n),
        dpi=100,
        constrained_layout=True,
    )
    if n == 1:
        axes = [axes]

    cmap = plt.get_cmap("tab10")

    for ax, method in zip(axes, methods):
        coords, lab = plot_data[method]
        lab_np = lab.detach().cpu().numpy()

        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=lab_np,
            cmap=cmap,
            vmin=0,
            vmax=9,
            s=45,
            alpha=0.8,
            edgecolors="white",
            linewidths=0.3,
            zorder=2,
        )

        ax.set_title(f"\nMethod: {method} ({reduction_actual})", loc="center", fontsize=22, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)
        ax.tick_params(axis="both", which="major", labelsize=14)

    if class_names is not None:
        handles = []
        for lbl in unique_labels:
            name = class_names[lbl] if lbl < len(class_names) else f"class {lbl}"
            h = plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=cmap(lbl / 10.0),
                markersize=10,
                label=name,
            )
            handles.append(h)

        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.52, -0.04),
            ncol=min(len(handles), 5),
            fontsize=18,
            frameon=True,
            shadow=True,
        )

    _save_figure(save_path, fig)

    fig.suptitle("Final Embeddings Comparison", fontsize=26, fontweight="black")
    plt.show()


def plot_single_method_embeddings(
    features: torch.Tensor,
    labels: torch.Tensor,
    method_name: str,
    class_names: Optional[List[str]] = None,
    seed: int = 42,
    reduction_method: Literal["auto", "umap", "tsne", "pca"] = "auto",
    save_path: Optional[str | Path] = None,
) -> None:
    """
    Plots the t-SNE/UMAP embeddings for a single method in 2D space.

    Parameters
    ----------
    features : torch.Tensor
        The features to plot.
    labels : torch.Tensor
        The labels for the features.
    method_name : str
        The name of the method.
    class_names : Optional[List[str]], optional
        The names of the classes, by default None.
    seed : int, optional
        The seed for reproducibility, by default 42.
    reduction_method : Literal["auto", "umap", "tsne", "pca"], optional
        The reduction method to use, by default "auto".
    save_path : Optional[str | Path], optional
        The path to save the figure to, by default None.

    Returns
    -------
    None
    """

    features, labels = subsample_for_viz(features, labels, max_points=5000, seed=seed)
    coords, reduction_actual = reduce_to_2d(
        features, method=reduction_method, random_state=seed
    )
    lab_np = labels.detach().cpu().numpy()
    unique_labels = sorted(np.unique(lab_np))

    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    cmap = plt.get_cmap("tab10")

    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=lab_np,
        cmap=cmap,
        vmin=0,
        vmax=9,
        s=45,
        alpha=0.8,
        edgecolors="white",
        linewidths=0.3,
        zorder=2,
    )

    ax.set_title(f"\n{reduction_actual} Latent Space", loc="center", fontsize=22, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=14)

    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    x_pad = (x_max - x_min) * 0.08
    y_pad = (y_max - y_min) * 0.08
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    if class_names is not None:
        handles = []
        for lbl in unique_labels:
            lbl_int = int(lbl)
            name = class_names[lbl_int] if lbl_int < len(class_names) else f"class {lbl_int}"
            h = plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=cmap(lbl_int / 10.0),
                markersize=10,
                label=name,
            )
            handles.append(h)

        fig.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.52, -0.04),
            ncol=min(len(handles), 5),
            fontsize=18,
            frameon=True,
            shadow=True,
        )

    _save_figure(save_path, fig)
    fig.suptitle(f"Embeddings - {method_name}", fontsize=26, fontweight="black")
    plt.show()

def plot_confusion_matrix(
    confusion: np.ndarray,
    method_name: str,
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    save_path: Optional[str | Path] = None,
) -> None:
    """
    Plot a confusion matrix as a heatmap.

    Parameters
    ----------
    confusion : np.ndarray
        Square confusion matrix (n_classes × n_classes).
    method_name : str
        Name of the CL method.
    class_names : list of str, optional
        Human-readable class names.
    normalize : bool
        If True, normalize rows to sum to 1 (show recall per class).
    """
    cm = confusion.astype(float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm = cm / row_sums * 100.0

    n = cm.shape[0]
    fig, ax = plt.subplots(figsize=(max(12, n * 0.8), max(8, n * 0.7)))

    cmap = plt.cm.Blues
    im = ax.imshow(
        cm, cmap=cmap, vmin=0, vmax=100 if normalize else None, aspect="auto"
    )

    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            fmt = f"{val:.0f}%" if normalize else f"{int(val)}"
            text_color = (
                "white" if val > (50 if normalize else cm.max() / 2) else "black"
            )
            ax.text(j, i, fmt, ha="center", va="center", fontsize=8, color=text_color)

    labels = class_names if class_names is not None else [str(i) for i in range(n)]
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=12)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel("Predicted", fontsize=16)
    ax.set_ylabel("True", fontsize=16)
    fig.colorbar(im, ax=ax, label="Recall (%)" if normalize else "Count", shrink=0.8)
    plt.tight_layout()
    _save_figure(save_path, fig)
    plt.title(f"Confusion matrix - {method_name}" + (" (normalized)" if normalize else ""), fontsize=22, fontweight="bold")
    plt.show()
