"""Causal window construction and training-window sampling.

The inference contract is fixed by the scorer: at step ``t`` a model may see
steps ``[t - context_len + 1, t]`` and nothing later. Short histories are
**left**-padded with zeros so the newest tick always lands at index ``-1``.
Training windows must be built the same way or the model sees a different input
distribution at inference time than it was trained on.
"""

from __future__ import annotations

import numpy as np

RAW_DIM = 32


def make_window(sequence: np.ndarray, step_idx: int, context_len: int) -> np.ndarray:
    """One causal window ending at ``step_idx`` (inclusive), left zero-padded."""
    start = step_idx + 1 - context_len
    if start >= 0:
        return np.asarray(sequence[start : step_idx + 1], dtype=np.float32)

    window = np.zeros((context_len, sequence.shape[1]), dtype=np.float32)
    available = sequence[: step_idx + 1]
    window[context_len - available.shape[0] :] = available
    return window


def make_windows_view(chunk: np.ndarray, context_len: int) -> np.ndarray:
    """Strided view of every causal window in a ``(S, T, C)`` chunk.

    Returns shape ``(S, T, context_len, C)``. The left pad is materialised once
    per chunk (cheap); the windows themselves are a view, so no per-window copy
    happens until you index into it.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    seq_count, _, channels = chunk.shape
    padding = np.zeros((seq_count, context_len - 1, channels), dtype=np.float32)
    padded = np.concatenate([padding, chunk], axis=1)
    return sliding_window_view(padded, context_len, axis=1).transpose(0, 1, 3, 2)


def sample_training_windows(
    raw_chunk: np.ndarray,
    target_chunk: np.ndarray,
    mask_chunk: np.ndarray,
    context_len: int,
    stride: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Stride-sample scored steps from each sequence, with a random phase.

    A stride of ``s`` yields roughly ``901 / s`` windows per sequence. The random
    offset means successive epochs see different steps, so the effective coverage
    over a run is much wider than one epoch's sample.
    """
    windows_view = make_windows_view(raw_chunk, context_len)
    seq_indices = []
    step_indices = []

    for seq_idx in range(raw_chunk.shape[0]):
        valid_steps = np.flatnonzero(mask_chunk[seq_idx])
        if valid_steps.size == 0:
            continue

        offset = int(rng.integers(0, max(stride, 1)))
        chosen = valid_steps[offset::stride]
        if chosen.size == 0 or chosen[-1] != valid_steps[-1]:
            chosen = np.concatenate([chosen, valid_steps[-1:]])

        seq_indices.append(np.full(chosen.shape[0], seq_idx, dtype=np.int32))
        step_indices.append(chosen.astype(np.int32, copy=False))

    if not seq_indices:
        return (
            np.zeros((0, context_len, raw_chunk.shape[2]), dtype=np.float32),
            np.zeros((0, target_chunk.shape[2]), dtype=np.float32),
        )

    seq_idx_arr = np.concatenate(seq_indices, axis=0)
    step_idx_arr = np.concatenate(step_indices, axis=0)
    windows = np.ascontiguousarray(windows_view[seq_idx_arr, step_idx_arr], dtype=np.float32)
    targets = np.ascontiguousarray(target_chunk[seq_idx_arr, step_idx_arr], dtype=np.float32)
    return windows, targets


class ShuffleBuffer:
    """Reservoir that decouples batch composition from chunk boundaries.

    The original training loops shuffled and batched *within* each 65-sequence
    parquet chunk. Two consequences, both bad:

    * every batch was drawn from ~65 adjacent-in-file sequences, so batches were
      far from i.i.d. samples of the corpus;
    * each chunk left a ragged trailing batch. For the TCN (520 windows, batch
      256) that was a batch of 8 every third optimizer step -- and ``mixed_loss``
      estimates a correlation *within the batch*, which is meaningless at n=8.

    Accumulate into this buffer instead and drain fixed-size batches from it.
    """

    def __init__(self, capacity: int, rng: np.random.Generator):
        self.capacity = int(capacity)
        self.rng = rng
        self._windows: list[np.ndarray] = []
        self._targets: list[np.ndarray] = []
        self._size = 0

    def add(self, windows: np.ndarray, targets: np.ndarray) -> None:
        if windows.shape[0] == 0:
            return
        self._windows.append(windows)
        self._targets.append(targets)
        self._size += windows.shape[0]

    def _consolidate(self) -> tuple[np.ndarray, np.ndarray]:
        windows = np.concatenate(self._windows, axis=0) if len(self._windows) > 1 else self._windows[0]
        targets = np.concatenate(self._targets, axis=0) if len(self._targets) > 1 else self._targets[0]
        self._windows = [windows]
        self._targets = [targets]
        return windows, targets

    def drain(self, batch_size: int, keep: int = 0):
        """Yield shuffled fixed-size batches, retaining ``keep`` samples.

        Carrying a residue across chunks is what removes the ragged trailing
        batch entirely. Call with ``keep=0`` at the end of an epoch to flush.
        """
        if self._size <= keep:
            return

        windows, targets = self._consolidate()
        order = self.rng.permutation(windows.shape[0])
        windows, targets = windows[order], targets[order]

        emit_upto = windows.shape[0] - keep
        cursor = 0
        while cursor + batch_size <= emit_upto:
            yield windows[cursor : cursor + batch_size], targets[cursor : cursor + batch_size]
            cursor += batch_size

        if keep == 0 and cursor < windows.shape[0]:
            # Final flush: emit the remainder only if it is big enough for the
            # batch-level correlation term to mean anything.
            if windows.shape[0] - cursor >= max(32, batch_size // 8):
                yield windows[cursor:], targets[cursor:]
            cursor = windows.shape[0]

        self._windows = [windows[cursor:]]
        self._targets = [targets[cursor:]]
        self._size = windows.shape[0] - cursor

    @property
    def size(self) -> int:
        return self._size


def parse_dilations(spec: str) -> list[int]:
    return [int(part.strip()) for part in spec.split(",") if part.strip()]


def receptive_field(dilations: list[int], kernel_size: int = 3) -> int:
    return 1 + (kernel_size - 1) * sum(dilations)
