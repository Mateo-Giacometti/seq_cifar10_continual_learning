# Continual Learning on Seq-CIFAR10 with Supervised Contrastive Pre-training

This repository contains the final implementation of the **Continual Learning (CL)** project for the `I309 - Advanced Computer Vision` course (UdeSA). The project explores the intersection of **Supervised Contrastive Learning (SupCon)** and several CL strategies to mitigate catastrophic forgetting on the Sequential CIFAR-10 dataset.

## Project Overview

The pipeline implements a comprehensive CL experimental setup, starting from a pre-trained backbone and evaluating its performance across multiple tasks using both **Class-IL** and **Task-IL** metrics.

### Key Components:
- **Dataset (4.1)**: Sequential CIFAR-10 builder with support for Replay Buffers (Reservoir Sampling).
- **Pre-training (4.2)**: Supervised Contrastive pre-training on Task 0 with subsequent linear evaluation.
- **CL Methods (4.3)**:
  - **Naive Fine-tuning**: Standard training without forgetting prevention (Baseline).
  - **EWC (Elastic Weight Consolidation)**: Regularization based on Fisher Information.
  - **LwF (Learning without Forgetting)**: Knowledge distillation from previous model versions.
  - **Co2L (Contrastive Continual Learning)**: Contrastive-based CL approach.
  - **ER-ACE (Experience Replay with Asymmetric Cross-Entropy)**: Buffer-based strategy.
- **Reporting & Metrics (4.4)**:
  - Global Accuracy (Class-IL / Task-IL) evolution.
  - **Forgetting (Max-History)** and **BWT (Backward Transfer)** evolution per step.
  - Confusion Matrices and Task-wise performance heatmaps.
  - **Latent Space Analysis**: Evolution of embeddings (t-SNE/UMAP) across tasks.

## Repository Structure

```text
.
├── configs/
│   └── cifar10.yaml        # Centralized hyperparameters
├── src/
│   └── tp1_cl/             # Core package
│       ├── bootstrap.py    # Logic for Task 0 transition
│       ├── checkpoints.py  # Loading/Saving models and histories
│       ├── config.py       # YAML configuration parser
│       ├── data.py         # Seq-CIFAR10 and Replay Buffers
│       ├── models.py       # ResNet backbones and SupCon heads
│       ├── train.py        # Generic training and evaluation loops
│       ├── viz.py          # Academic-grade visualization engine
│       └── methods/        # Specific CL implementations
│           ├── co2l.py
│           ├── er_ace.py
│           ├── ewc.py
│           ├── finetuning.py
│           └── lwf.py
├── outputs/                # Generated artifacts (checkpoints, images, metrics)
├── tp1.ipynb               # Main execution notebook
└── requirements.txt        # Project dependencies
```

## Setup & Installation

It is recommended to use a virtual environment with Python 3.10+.

```bash
# Create and activate environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## How to Run

1.  **Configuration**: Modify `configs/cifar10.yaml` to adjust learning rates, batch sizes, or seeds.
2.  **Jupyter Notebook**: Open `tp1.ipynb`.
3.  **Execution Flow**:
    - Run the **Preparation** cells to initialize datasets and builders.
    - Execute **Section 4.2** to perform SupCon pre-training on Task 0.
    - Run the **CL Protocol** cells (Section 4.3) for each method. The notebook automatically handles checkpointing; if a `.pt` file exists in `outputs/checkpoints/`, it will load the results instead of retraining.
    - The final section generates a **Comparative Report** including summary tables and unified plots.

## Visualization Engine

The project includes a robust visualization module (`tp1_cl.viz`) designed for academic reporting:
- **`plot_cl_metrics`**: Evolution of global accuracies with per-point annotations.
- **`plot_forgetting_metrics` / `plot_bwt_metrics`**: Evolution of transfer metrics per step.
- **`plot_final_embeddings_comparison`**: Unified view of the latent space for all trained methods using a shared scale.
- **`plot_single_method_embeddings`**: High-detail visualization of a specific method's feature distribution.
- **`plot_taskwise_heatmap`**: Detailed breakdown of accuracy for every learned task at every step.

## Reproducibility

- **Seed Control**: A global seed is set in `configs/cifar10.yaml` to ensure deterministic behavior.
- **Hardware**: The code automatically detects and uses CUDA if available.
- **Protocol**: Uses the `from_task0_pretrained` protocol, where Task 0 provides the feature initialization for all subsequent CL experiments.
