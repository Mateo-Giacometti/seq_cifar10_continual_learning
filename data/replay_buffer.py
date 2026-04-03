"""
Replay buffer with reservoir sampling for Continual Learning.

Reservoir sampling guarantees a uniform random sample of all examples seen so
far, regardless of how many examples have been added.  This is the standard
memory-management strategy used by methods such as ER, Co²L, etc.
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


class ReplayBuffer:
    """Fixed-capacity episodic memory with reservoir sampling.

    Each stored example is a tuple ``(image_tensor, local_label, global_label,
    task_id)``.

    Parameters
    ----------
    capacity:
        Maximum number of examples held in memory.
    seed:
        Optional random seed for reproducibility.
    """

    def __init__(self, capacity: int, seed: Optional[int] = None) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        self.capacity = capacity
        self._rng = random.Random(seed)

        # Internal storage as parallel lists (avoids stacking heterogeneous types).
        self._images: List[Tensor] = []
        self._local_labels: List[int] = []
        self._global_labels: List[int] = []
        self._task_ids: List[int] = []

        # Total examples seen so far (used for reservoir probability).
        self._n_seen: int = 0

    # ------------------------------------------------------------------
    # Core reservoir sampling
    # ------------------------------------------------------------------

    def add(
        self,
        image: Tensor,
        local_label: int,
        global_label: int,
        task_id: int,
    ) -> None:
        """Add a single example to the buffer using reservoir sampling."""
        self._n_seen += 1
        if len(self._images) < self.capacity:
            # Buffer not full yet – always accept.
            self._images.append(image)
            self._local_labels.append(local_label)
            self._global_labels.append(global_label)
            self._task_ids.append(task_id)
        else:
            # Replace a random existing example with probability capacity / n_seen.
            j = self._rng.randint(0, self._n_seen - 1)
            if j < self.capacity:
                self._images[j] = image
                self._local_labels[j] = local_label
                self._global_labels[j] = global_label
                self._task_ids[j] = task_id

    def add_from_loader(
        self,
        loader: DataLoader,
        task_id: int,
        max_samples: Optional[int] = None,
    ) -> None:
        """Populate the buffer from a DataLoader.

        The loader is expected to yield batches of the form
        ``(images, local_labels, global_labels)`` as produced by
        :class:`~data.cifar10_tasks.TaskSubset`.

        Parameters
        ----------
        loader:
            DataLoader over a :class:`~data.cifar10_tasks.TaskSubset`.
        task_id:
            Task index for the stored examples.
        max_samples:
            If set, stop after processing this many examples.
        """
        n_added = 0
        for batch in loader:
            images, local_labels, global_labels = batch
            for img, ll, gl in zip(images, local_labels, global_labels):
                self.add(img, int(ll), int(gl), task_id)
                n_added += 1
                if max_samples is not None and n_added >= max_samples:
                    return

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        n: int,
        task_id: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Sample *n* examples (with replacement if n > buffer size).

        Parameters
        ----------
        n:
            Number of examples to sample.
        task_id:
            If provided, restrict sampling to examples from this task.

        Returns
        -------
        images: Tensor of shape (n, C, H, W)
        local_labels: LongTensor of shape (n,)
        global_labels: LongTensor of shape (n,)
        task_ids: LongTensor of shape (n,)
        """
        if len(self) == 0:
            raise RuntimeError("Cannot sample from an empty buffer.")

        if task_id is not None:
            indices = [i for i, t in enumerate(self._task_ids) if t == task_id]
        else:
            indices = list(range(len(self._images)))

        if len(indices) == 0:
            raise RuntimeError(f"No examples for task_id={task_id} in buffer.")

        chosen = self._rng.choices(indices, k=n)

        images = torch.stack([self._images[i] for i in chosen])
        local_labels = torch.tensor([self._local_labels[i] for i in chosen], dtype=torch.long)
        global_labels = torch.tensor([self._global_labels[i] for i in chosen], dtype=torch.long)
        task_ids = torch.tensor([self._task_ids[i] for i in chosen], dtype=torch.long)

        return images, local_labels, global_labels, task_ids

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_dataloader(
        self,
        batch_size: int = 64,
        shuffle: bool = True,
    ) -> DataLoader:
        """Return a DataLoader over all stored examples.

        Yields batches of ``(images, local_labels, global_labels, task_ids)``.
        """
        if len(self) == 0:
            raise RuntimeError("Buffer is empty.")

        images = torch.stack(self._images)
        local_labels = torch.tensor(self._local_labels, dtype=torch.long)
        global_labels = torch.tensor(self._global_labels, dtype=torch.long)
        task_ids = torch.tensor(self._task_ids, dtype=torch.long)

        dataset = TensorDataset(images, local_labels, global_labels, task_ids)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def __len__(self) -> int:
        return len(self._images)

    def __repr__(self) -> str:
        return (
            f"ReplayBuffer(capacity={self.capacity}, "
            f"stored={len(self)}, seen={self._n_seen})"
        )
