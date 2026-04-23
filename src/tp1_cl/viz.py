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
    max_points: int = 2000,
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
    features: torch.Tensor,
    method: Literal["auto", "umap", "tsne", "pca"] = "auto",
    random_state: int = 42,
) -> Tuple[np.ndarray, str]:
    x = features.detach().cpu().numpy()

    # --- UMAP ---
    if method in ["auto", "umap"]:
        try:
            import umap.umap_ as umap

            reducer = umap.UMAP(n_components=2, random_state=random_state)
            return reducer.fit_transform(x), "UMAP"
        except Exception:
            if method == "umap":
                print("⚠️ UMAP no disponible, cayendo a t-SNE...")
            pass

    # --- t-SNE ---
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
                print("⚠️ t-SNE no disponible, cayendo a PCA...")
            pass

    # --- PCA (Fallback final) ---
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
    losses: List[float], save_path: Optional[str | Path] = None, smooth_window: int = 5
) -> None:
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
    """Plot snapshots of embeddings in latent space using dimensionality reduction.

    Optimized vertical layout with consistent global limits and correct color mapping.
    """
    stages = ["inicio", "mitad", "final"]
    available_stages = [s for s in stages if s in snapshots]
    if len(available_stages) == 0:
        print("No snapshots available for embedding visualization.")
        return

    # 1. Pre-calculate projections and global bounds
    plot_data = {}
    bounds = []
    reduction_actual = "PCA"

    for stage in available_stages:
        feat, lab = snapshots[stage]
        # Subsample for speed and clarity
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

    # Calculate global limits with 8% padding
    bounds_arr = np.array(bounds)
    x_min, x_max = float(np.min(bounds_arr[:, 0])), float(np.max(bounds_arr[:, 1]))
    y_min, y_max = float(np.min(bounds_arr[:, 2])), float(np.max(bounds_arr[:, 3]))

    x_pad = (x_max - x_min) * 0.08
    y_pad = (y_max - y_min) * 0.08

    # 2. Setup figure using constrained_layout for better spacing
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

    # 3. Plot each stage
    for ax, stage in zip(axes, available_stages):
        coords, lab = plot_data[stage]
        local_labels = np.array(
            [class_to_local[int(y)] for y in lab.detach().cpu().numpy()]
        )

        # Fix vmin/vmax to ensure color index consistency with tab10
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

        ax.set_title(f"\n{title}", loc="center", fontsize=20, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

        # Minimalist spines
        ax.set_xlabel("Latent Dimension 1", fontsize=16)
        ax.set_ylabel("Latent Dimension 2", fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=14)

    # 4. Global legend at the bottom
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
    task_ids = [int(t) for t in history.get("task_id", [])]
    if len(task_ids) == 0:
        print("No continual-learning history to plot.")
        return

    x = [task_id + 1 for task_id in task_ids]
    class_il = history.get("class_il", [])
    task_il = history.get("task_il", [])

    fig = plt.figure(figsize=(12, 8))
    plt.plot(x, class_il, marker="o", label="Class-IL", color="#E45756")
    plt.plot(x, task_il, marker="s", label="Task-IL", color="#4C78A8")
    # Annotate all points
    for xi, yi in zip(x, class_il):
        plt.annotate(
            f"{yi:.2f}%",
            (xi, yi),
            textcoords="offset points",
            xytext=(0, -18),  # Below
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="#E45756",
        )
    for xi, yi in zip(x, task_il):
        plt.annotate(
            f"{yi:.2f}%",
            (xi, yi),
            textcoords="offset points",
            xytext=(0, 10),  # Above
            ha="center",
            fontsize=10,
            fontweight="bold",
            color="#4C78A8",
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


def plot_methods_comparison_class_il(
    histories_by_method: Mapping[str, Dict[str, List[float]]],
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot Class-IL accuracy comparison across different methods."""
    
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
    plt.grid(alpha=0.3)
    
    # Ensure ticks are integers
    all_tasks = {i + 1 for h in histories_by_method.values() for i in range(len(h.get("task_id", [])))}
    if all_tasks:
        plt.xticks(sorted(all_tasks))
    
    plt.legend(fontsize=14)
    _save_figure(save_path, fig)
    plt.title("Class-IL Comparison", fontsize=22, fontweight="bold")
    plt.show()


def plot_methods_comparison_task_il(
    histories_by_method: Mapping[str, Dict[str, List[float]]],
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot Task-IL accuracy comparison across different methods."""
    set_plot_style()
    if len(histories_by_method) == 0:
        print("No method histories to compare.")
        return

    markers = ["s", "o", "D", "^", "v", "*", "X", "p"]
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
    plt.grid(alpha=0.3)
    
    # Ensure ticks are integers
    all_tasks = {i + 1 for h in histories_by_method.values() for i in range(len(h.get("task_id", [])))}
    if all_tasks:
        plt.xticks(sorted(all_tasks))
        
    plt.legend(fontsize=14)
    _save_figure(save_path, fig)
    plt.title("Task-IL Comparison", fontsize=22, fontweight="bold")
    plt.show()


def _as_taskwise_array(taskwise_matrix: Sequence[Sequence[float]]) -> np.ndarray:
    arr = np.array(taskwise_matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError("taskwise_matrix must be 2-dimensional")
    return arr


def _first_seen_indices(arr: np.ndarray) -> List[Optional[int]]:
    first_seen: List[Optional[int]] = []
    for k in range(arr.shape[1]):
        seen = np.where(~np.isnan(arr[:, k]))[0]
        first_seen.append(int(seen[0]) if seen.size > 0 else None)
    return first_seen


def compute_forgetting_matrix(
    taskwise_matrix: Sequence[Sequence[float]],
    definition: str = "max_history",
) -> np.ndarray:
    """Compute forgetting matrix from taskwise accuracies.

    Let A[t, k] be the accuracy on task k after training step t.

    - definition='max_history' (recommended):
      F[t, k] = max_{l<t} A[l, k] - A[t, k]
    - definition='diagonal' (legacy):
      F[t, k] = A[t0(k), k] - A[t, k], where t0(k) is first seen step of task k

    For both, values are only defined for tasks seen before step t.
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
    """Compute BWT matrix B[t, k] = A[t, k] - A[t0(k), k].

    Values are only defined for tasks seen before step t.
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
    """Average a taskwise metric matrix over defined past-task entries per step."""
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
    """Compute aggregate forgetting/BWT metrics from a taskwise accuracy matrix."""
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
    set_plot_style()
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

    fig = plt.figure(figsize=(9.2, 4.8))
    markers = ["o", "s", "D", "^", "v", "*", "X", "p"]
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
    plt.title(f"Average forgetting over tasks ({title_metric}, {label_definition})")
    plt.xlabel("Number of tasks learned")
    plt.ylabel("Forgetting (%)")
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
    plt.legend()
    _save_figure(save_path, fig)
    plt.show()


def plot_bwt_curves(
    histories_by_method: Mapping[str, Dict[str, object]],
    metric: str = "class_il",
    save_path: Optional[str | Path] = None,
) -> None:
    set_plot_style()
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

    fig = plt.figure(figsize=(9.2, 4.8))
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

    plt.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--", alpha=0.7)
    plt.title(f"Average BWT over tasks ({title_metric})")
    plt.xlabel("Number of tasks learned")
    plt.ylabel("BWT (%)")
    plt.xticks(x)
    plt.legend()
    _save_figure(save_path, fig)
    plt.show()


def plot_forgetting_by_task(
    history: Dict[str, object],
    method_name: str,
    metric: str = "class_il",
    definition: str = "max_history",
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot forgetting trajectories F[t, k] for each task k in one method."""
    set_plot_style()
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
    label_definition = "max-history" if definition == "max_history" else "diagonal"
    plt.title(
        f"Forgetting by task - {method_name} ({title_metric}, {label_definition})"
    )
    plt.xlabel("Number of tasks learned")
    plt.ylabel("Forgetting (%)")
    plt.xticks(x)
    plt.ylim(bottom=0)
    plt.legend(ncol=2)
    _save_figure(save_path, fig)
    plt.show()


# ---------------------------------------------------------------------------
#  New visualizations
# ---------------------------------------------------------------------------


def plot_taskwise_heatmap(
    taskwise_matrix: Sequence[Sequence[float]],
    method_name: str,
    metric: str = "class_il",
    task_classes: Optional[List[List[int]]] = None,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot a heatmap of the A[t, k] accuracy matrix.

    Rows = training step (after learning task t), Columns = evaluated task k.
    """
    set_plot_style()
    arr = _as_taskwise_array(taskwise_matrix)
    n_steps, n_tasks = arr.shape
    title_metric = "Class-IL" if metric == "class_il" else "Task-IL"

    fig, ax = plt.subplots(figsize=(max(6, n_tasks * 1.4), max(4, n_steps * 0.9)))

    masked = np.ma.array(arr, mask=np.isnan(arr))
    cmap = plt.cm.RdYlGn.copy()
    cmap.set_bad(color="#E8E8E8")

    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=100, aspect="auto")

    # Annotate cells
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

    # Labels
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
    ax.set_xticklabels(task_labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(n_steps))
    ax.set_yticklabels(step_labels, fontsize=9)
    ax.set_xlabel("Evaluated task")
    ax.set_ylabel("Training step")
    ax.set_title(f"Taskwise accuracy matrix – {method_name} ({title_metric})")
    fig.colorbar(im, ax=ax, label="Accuracy (%)", shrink=0.8)

    plt.tight_layout()
    _save_figure(save_path, fig)
    plt.show()


def plot_cl_training_losses(
    histories_by_method: Mapping[str, Dict[str, object]],
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot training loss curves for each CL method side by side."""
    set_plot_style()
    methods = {
        m: h for m, h in histories_by_method.items() if len(h.get("train_loss", [])) > 0
    }
    if len(methods) == 0:
        print("No training losses to plot.")
        return

    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    axes = axes[0]

    for ax, (method, hist) in zip(axes, methods.items()):
        losses = hist["train_loss"]
        task_ids = [int(t) for t in hist.get("task_id", range(len(losses)))]
        x = range(1, len(losses) + 1)
        ax.plot(x, losses, marker="o", color="#4C78A8")

        for i, tid in enumerate(task_ids):
            ax.annotate(
                f"T{tid}",
                (i + 1, losses[i]),
                textcoords="offset points",
                xytext=(0, 8),
                fontsize=8,
                ha="center",
                color="#666666",
            )

        ax.set_title(method)
        ax.set_xlabel("Task step")
        ax.set_ylabel("Training loss")

    plt.suptitle("Training loss per CL method", fontsize=14, y=1.02)
    plt.tight_layout()
    _save_figure(save_path, fig)
    plt.show()


def plot_final_embeddings_comparison(
    embeddings_by_method: Mapping[str, Tuple[torch.Tensor, torch.Tensor]],
    class_names: Optional[List[str]] = None,
    seed: int = 42,
    reduction_method: Literal["auto", "umap", "tsne", "pca"] = "auto",
    save_path: Optional[str | Path] = None,
) -> None:
    """t-SNE/UMAP of final embeddings for each method, side by side.

    Parameters
    ----------
    embeddings_by_method : dict
        ``{method_name: (features_tensor, labels_tensor)}``.
    class_names : list of str, optional
        Human-readable class names for legend.
    """
    set_plot_style()
    methods = list(embeddings_by_method.keys())
    if len(methods) == 0:
        print("No embeddings to plot.")
        return

    n = len(methods)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5))
    if n == 1:
        axes = [axes]

    all_labels_set: set = set()
    for feat, lab in embeddings_by_method.values():
        all_labels_set.update(lab.detach().cpu().numpy().tolist())
    unique_labels = sorted(all_labels_set)

    for ax, method in zip(axes, methods):
        feat, lab = embeddings_by_method[method]
        feat, lab = subsample_for_viz(feat, lab, max_points=2000, seed=seed)
        coords, dim_method = reduce_to_2d(
            feat, method=reduction_method, random_state=seed
        )
        lab_np = lab.detach().cpu().numpy()

        scatter = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=lab_np,
            cmap="tab10",
            s=10,
            alpha=0.65,
            vmin=min(unique_labels),
            vmax=max(unique_labels),
        )
        ax.set_title(f"{method} ({dim_method})")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")

    # Legend
    if class_names is not None:
        handles = []
        for lbl in unique_labels:
            name = class_names[lbl] if lbl < len(class_names) else f"class {lbl}"
            h = plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=plt.cm.tab10(lbl / 10.0),
                markersize=7,
                label=name,
            )
            handles.append(h)
        fig.legend(
            handles=handles, loc="upper center", ncol=min(len(handles), 5), fontsize=9
        )

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    _save_figure(save_path, fig)
    plt.show()


def plot_final_accuracy_bars(
    results: Mapping[str, Dict[str, float]],
    save_path: Optional[str | Path] = None,
) -> None:
    """Bar chart comparing final Class-IL and Task-IL accuracy across methods.

    Parameters
    ----------
    results : dict
        ``{method_name: {"class_il": float, "task_il": float}}``.
    """
    set_plot_style()
    if len(results) == 0:
        print("No results to plot.")
        return

    methods = list(results.keys())
    class_il = []
    task_il = []
    for m in methods:
        c = results[m].get("class_il", 0.0)
        t = results[m].get("task_il", 0.0)
        # If it's a list/array, take the last element
        if isinstance(c, (list, tuple, np.ndarray)) and len(c) > 0:
            c = c[-1]
        if isinstance(t, (list, tuple, np.ndarray)) and len(t) > 0:
            t = t[-1]
        class_il.append(float(c))
        task_il.append(float(t))

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(methods) * 2), 5))
    bars1 = ax.bar(
        x - width / 2,
        class_il,
        width,
        label="Class-IL",
        color="#E45756",
        edgecolor="white",
        linewidth=0.8,
    )
    bars2 = ax.bar(
        x + width / 2,
        task_il,
        width,
        label="Task-IL",
        color="#4C78A8",
        edgecolor="white",
        linewidth=0.8,
    )

    # Value labels
    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{bar.get_height():.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#E45756",
        )
    for bar in bars2:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{bar.get_height():.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#4C78A8",
        )

    ax.set_xlabel("Method")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Final accuracy comparison – Class-IL vs Task-IL")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper left")

    plt.tight_layout()
    _save_figure(save_path, fig)
    plt.show()


def plot_confusion_matrix(
    confusion: np.ndarray,
    method_name: str,
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    save_path: Optional[str | Path] = None,
) -> None:
    """Plot a confusion matrix as a heatmap.

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
    set_plot_style()
    cm = confusion.astype(float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm = cm / row_sums * 100.0

    n = cm.shape[0]
    fig, ax = plt.subplots(figsize=(max(6, n * 0.8), max(5, n * 0.7)))

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
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(
        f"Confusion matrix – {method_name}" + (" (normalized)" if normalize else "")
    )
    fig.colorbar(im, ax=ax, label="Recall (%)" if normalize else "Count", shrink=0.8)

    plt.tight_layout()
    _save_figure(save_path, fig)
    plt.show()
