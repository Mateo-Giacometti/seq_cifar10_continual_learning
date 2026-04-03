"""
Sequential Continual Learning training loop (Etapas 3 & 4).

:class:`CLTrainer` orchestrates:
  1. Training each method on sequential tasks.
  2. Evaluating Task-IL and Class-IL accuracy after every task.
  3. Storing metrics for later comparison / plotting.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from data.cifar10_tasks import CIFAR10Tasks
from data.replay_buffer import ReplayBuffer
from evaluation.metrics import CLMetrics
from methods.base import CLMethod


class CLTrainer:
    """Runs a CL method sequentially over all tasks and tracks metrics.

    Parameters
    ----------
    method:
        Instantiated :class:`~methods.base.CLMethod` sub-class.
    tasks:
        :class:`~data.cifar10_tasks.CIFAR10Tasks` container.
    metrics:
        :class:`~evaluation.metrics.CLMetrics` tracker.
    device:
        Computation device (defaults to method's device).
    batch_size:
        DataLoader batch size.
    num_workers:
        DataLoader worker processes.
    """

    def __init__(
        self,
        method: CLMethod,
        tasks: CIFAR10Tasks,
        metrics: CLMetrics,
        device: Optional[torch.device] = None,
        batch_size: int = 128,
        num_workers: int = 2,
    ) -> None:
        self.method = method
        self.tasks = tasks
        self.metrics = metrics
        self.device = device or method.device
        self.batch_size = batch_size
        self.num_workers = num_workers

    # ------------------------------------------------------------------

    def run(
        self,
        epochs_per_task: int = 10,
        lr: float = 1e-3,
    ) -> Dict:
        """Train sequentially over all tasks and return metric history.

        After training on task *t*, evaluates the model on tasks 0 … t.

        Returns
        -------
        results:
            Dict with keys:
              - ``"task_il"``  : 2-D list [after_task_t][eval_task_id] → accuracy
              - ``"class_il"`` : list of overall Class-IL accuracies
        """
        num_tasks = self.tasks.num_tasks
        task_il_matrix: List[List[float]] = []
        class_il_accuracies: List[float] = []

        for task_id in range(num_tasks):
            print(f"\n{'='*50}")
            print(f"Training on Task {task_id}")
            print(f"{'='*50}")

            train_loader = self._make_loader(task_id, train=True)
            self.method.train_task(task_id, train_loader, epochs=epochs_per_task, lr=lr)

            # Evaluate on all tasks seen so far.
            task_il_row: List[float] = []
            for eval_task in range(task_id + 1):
                test_loader = self._make_loader(eval_task, train=False)
                acc_task_il = self._eval_task_il(eval_task, test_loader)
                task_il_row.append(acc_task_il)

            task_il_matrix.append(task_il_row)

            # Class-IL: concatenate all seen test sets and evaluate globally.
            class_il_acc = self._eval_class_il_all(task_id)
            class_il_accuracies.append(class_il_acc)

            # Log to metrics tracker.
            self.metrics.update(
                task_il_row=task_il_row,
                class_il_acc=class_il_acc,
                after_task=task_id,
            )

            print(f"\nAfter Task {task_id}:")
            print(f"  Task-IL accuracies: {[f'{a:.3f}' for a in task_il_row]}")
            print(f"  Class-IL accuracy:  {class_il_acc:.3f}")

        return {
            "task_il": task_il_matrix,
            "class_il": class_il_accuracies,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_loader(self, task_id: int, train: bool) -> DataLoader:
        dataset = (
            self.tasks.get_train(task_id) if train else self.tasks.get_test(task_id)
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=train,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    @torch.no_grad()
    def _eval_task_il(self, task_id: int, loader: DataLoader) -> float:
        """Accuracy using the task-specific head."""
        correct = 0
        total = 0
        for batch in loader:
            images, local_labels, _ = batch
            images = images.to(self.device)
            local_labels = local_labels.to(self.device)
            logits = self.method.predict_task_il(images, task_id)
            preds = logits.argmax(dim=1)
            correct += (preds == local_labels).sum().item()
            total += images.size(0)
        return correct / total if total > 0 else 0.0

    @torch.no_grad()
    def _eval_class_il_all(self, up_to_task: int) -> float:
        """Global Class-IL accuracy over all tasks seen so far."""
        correct = 0
        total = 0
        for task_id in range(up_to_task + 1):
            loader = self._make_loader(task_id, train=False)
            for batch in loader:
                images, _, global_labels = batch
                images = images.to(self.device)
                global_labels = global_labels.to(self.device)
                logits = self.method.predict_class_il(images)
                preds = logits.argmax(dim=1)
                correct += (preds == global_labels).sum().item()
                total += images.size(0)
        return correct / total if total > 0 else 0.0
