from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch


def subsample_for_viz(
    features: torch.Tensor,
    labels: torch.Tensor,
    max_points: int = 1500,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Subsample features and labels for visualization if there are too many points.

    Parameters
    ----------
    features : torch.Tensor
        The feature embeddings to be visualized, of shape (N, D).
    labels : torch.Tensor
        The corresponding labels for the features, of shape (N,).
    max_points : int, optional
        The maximum number of points to visualize. If the number of features exceeds this, a random
        subset will be taken. Default is 1500.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        A tuple containing the subsampled features and labels.
    """
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
    x = features.cpu().numpy()
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
    coords = (centered @ v[:, :2]).cpu().numpy()
    return coords, "PCA"


def plot_supcon_loss(losses: List[float]) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(losses) + 1), losses, marker="o")
    plt.title("SupCon pretraining loss (Task 0)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_embedding_snapshots(
    snapshots: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    task_classes: List[int],
    class_names: List[str],
    seed: int,
) -> None:
    stages = ["inicio", "mitad", "final"]
    available_stages = [s for s in stages if s in snapshots]
    if len(available_stages) == 0:
        print("No snapshots available for embedding visualization.")
        return

    fig, axes = plt.subplots(
        1, len(available_stages), figsize=(6 * len(available_stages), 5)
    )
    if len(available_stages) == 1:
        axes = [axes]

    class_to_local = {c: i for i, c in enumerate(task_classes)}

    for ax, stage in zip(axes, available_stages):
        feat, lab = snapshots[stage]
        feat, lab = subsample_for_viz(feat, lab, max_points=1500, seed=seed)
        coords, method = reduce_to_2d(feat, random_state=seed)
        local_labels = np.array([class_to_local[int(y)] for y in lab.numpy()])

        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c=local_labels,
            cmap="tab10",
            s=10,
            alpha=0.75,
        )
        ax.set_title(f"Embeddings {stage} ({method})")
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")

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
        ax.legend(handles=handles, loc="best")

    plt.tight_layout()
    plt.show()


def plot_linear_history(
    history: Dict[str, List[float]],
    title: str = "Linear head accuracy",
    xlabel: str = "Epoch",
    ylabel: str = "Accuracy (%)",
) -> None:
    plt.figure(figsize=(7, 4))
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["test_acc"], label="Test Accuracy")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def plot_cl_metrics(
    history: Dict[str, List[float]],
    title: str = "Continual Learning metrics",
) -> None:
    task_ids = [int(t) for t in history.get("task_id", [])]
    if len(task_ids) == 0:
        print("No continual-learning history to plot.")
        return

    x = [task_id + 1 for task_id in task_ids]

    plt.figure(figsize=(8, 4.5))
    plt.plot(x, history.get("class_il", []), marker="o", label="Class-IL")
    plt.plot(x, history.get("task_il", []), marker="s", label="Task-IL")
    plt.title(title)
    plt.xlabel("Number of tasks learned")
    plt.ylabel("Accuracy (%)")
    plt.xticks(x)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
