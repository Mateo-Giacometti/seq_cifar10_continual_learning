# Continual Learning on Seq-CIFAR10 with Supervised Contrastive Pre-training

---

<table>
  <tr>
    <td width="36%" align="center" valign="middle">
      <img src=".assets/image.png" alt="University of San Andres logo" width="320" />
    </td>
    <td width="64%" valign="top">
      <h2>Project 1: Continual Learning</h2>
      <p><strong>Course:</strong> Advanced Computer Vision (I309)</p>
      <p><strong>Institution:</strong> University of San Andres, Argentina</p>
      <p><strong>Program:</strong> Artificial Intelligence Engineering</p>
      <p><strong>Academic Year:</strong> 4th year</p>
    </td>
  </tr>
</table>

---

## Project Overview

This repository contains the first assignment for the **Advanced Computer Vision** course.
The project explores the intersection of **Supervised Contrastive Learning (SupCon)** and several Continual Learning (CL) strategies to mitigate catastrophic forgetting on the Sequential CIFAR-10 dataset.

## Problem Statement

When a deep learning model is sequentially trained on new tasks, its performance on previous tasks usually degrades drastically, a phenomenon known as catastrophic forgetting. This project implements and evaluates progressively robust methods to address this issue: from naive fine-tuning to knowledge distillation and contrastive-based continual learning (Co²L).

## Methodology

1. **Dataset Preparation**: Build Seq-CIFAR-10 by dividing the 10 classes into 5 sequential tasks. Implement replay buffers (e.g., Reservoir Sampling).
2. **Pre-training**: Train the initial backbone on Task 0 using Supervised Contrastive Learning (SupCon), followed by a linear evaluation head.
3. **CL Methods Evaluation**:
   - **Naive Fine-tuning**: Standard training without forgetting prevention (Baseline).
   - **EWC (Elastic Weight Consolidation)**: Regularization based on Fisher Information.
   - **LwF (Learning without Forgetting)**: Knowledge distillation from previous model versions.
   - **Co2L (Contrastive Continual Learning)**: Contrastive-based CL approach.
   - **ER-ACE (Experience Replay with Asymmetric Cross-Entropy)**: Buffer-based strategy.
4. **Analysis & Metrics**: Evaluate Class-IL and Task-IL global accuracy, Forgetting metrics, Backward Transfer (BWT), and Latent Space evolution via t-SNE/UMAP.

## Visualizations and Reporting

The project includes an extensive visualization engine `tp1_cl.viz` tailored for academic reporting:
- **Global Accuracy**: Evolution of global accuracies with per-point annotations.
- **Transfer Metrics**: Evolution of Forgetting and Backward Transfer (BWT) metrics per step.
- **Latent Space**: Evolution of embeddings across tasks (t-SNE/UMAP).
- **Task-wise Performance**: Detailed breakdown of accuracy via heatmaps and confusion matrices.

*(Check the `outputs/` folder or the generated report for detailed visualizations)*

## Repository Structure

```text
tp1_vision_artificial_avanzada/
├── configs/
│   └── cifar10.yaml        # Centralized hyperparameters
├── src/
│   └── tp1_cl/             # Core package implementation
│       ├── bootstrap.py    # Logic for Task 0 transition
│       ├── checkpoints.py  # Loading/Saving models and histories
│       ├── config.py       # YAML configuration parser
│       ├── data.py         # Seq-CIFAR10 and Replay Buffers
│       ├── models.py       # ResNet backbones and SupCon heads
│       ├── train.py        # Generic training and evaluation loops
│       ├── viz.py          # Academic-grade visualization engine
│       └── methods/        # Specific CL implementations
├── outputs/                # Generated artifacts (checkpoints, images, metrics)
├── project/                # Assignment description and assets
├── papers/                 # Reference papers
├── report/                 # Full academic report
│   └── giacometti_martin_bernal_tp1_informe.pdf
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

## Reproducibility and Usage

1. **Configuration**: Modify `configs/cifar10.yaml` to adjust learning rates, batch sizes, or seeds.
2. **Jupyter Notebook**: Run experiments from `tp1.ipynb`.
    - Run the Preparation cells to initialize datasets.
    - Execute Section 4.2 for SupCon pre-training.
    - Run the CL Protocol cells (Section 4.3) for each method. The notebook handles checkpointing automatically.
    - Generate Comparative Report tables and plots.

## Authors

- **Mateo Giacometti**
- **Tiziano Levi Martin Bernal**

## License

This project is distributed under the MIT License. See `LICENSE` for details.
