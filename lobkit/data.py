"""Corpus access.

Two paths live here:

``iter_sequence_chunks``
    The original streaming reader, lifted verbatim from the training scripts
    (where four of five copies were byte-identical). Still used by
    ``scripts/materialise.py`` and by anything that must read parquet directly.

``LobCorpus``
    Memory-mapped access to the materialised ``.npy`` corpus. Every training and
    evaluation run should use this: it removes both the repeated passes over the
    1.5 GB parquet and the full extra pass each window model used to make through
    ``compute_feature_stats`` before its first epoch.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SEQUENCE_LENGTH = 1000
WARMUP_STEPS = 99

RAW_FEATURE_COLUMNS = [f"p{i}" for i in range(12)]
RAW_FEATURE_COLUMNS += [f"v{i}" for i in range(12)]
RAW_FEATURE_COLUMNS += [f"dp{i}" for i in range(4)]
RAW_FEATURE_COLUMNS += [f"dv{i}" for i in range(4)]
TARGET_COLUMNS = ["t0", "t1"]

RAW_DIM = len(RAW_FEATURE_COLUMNS)
ENGINEERED_DIM = 10
FEATURE_DIM = RAW_DIM + RAW_DIM + ENGINEERED_DIM

# Channel layout of the 32 raw features (Level-6 book plus 4 trade channels).
BID_PRICE = slice(0, 6)
ASK_PRICE = slice(6, 12)
BID_VOLUME = slice(12, 18)
ASK_VOLUME = slice(18, 24)
TRADE_PRICE = slice(24, 28)
TRADE_VOLUME = slice(28, 32)


def concat_if_needed(left: np.ndarray | None, right: np.ndarray) -> np.ndarray:
    if left is None or left.size == 0:
        return right
    return np.concatenate([left, right], axis=0)


def iter_sequence_chunks(
    parquet_path: str | Path,
    include_targets: bool,
    batch_rows: int = 65536,
    prepare_fn=None,
):
    """Yield ``(raw, targets, mask)`` in chunks of whole sequences.

    ``prepare_fn`` is applied to each raw chunk before yielding, which covers the
    one way ``solution_tlob_lite.py`` diverged from the other copies (it applies a
    signed-log compression to the volume channels).
    """
    parquet_path = Path(parquet_path)
    import pyarrow.parquet as pq

    columns = ["step_in_seq", "need_prediction", *RAW_FEATURE_COLUMNS]
    if include_targets:
        columns.extend(TARGET_COLUMNS)

    parquet_file = pq.ParquetFile(parquet_path)
    carry: dict[str, np.ndarray | None] = {column: None for column in columns}
    expected_steps = np.arange(SEQUENCE_LENGTH, dtype=np.int16)[None, :]

    for batch in parquet_file.iter_batches(batch_size=batch_rows, columns=columns):
        arrays: dict[str, np.ndarray] = {}
        for idx, column in enumerate(columns):
            dtype = np.float32
            if column == "step_in_seq":
                dtype = np.int16
            elif column == "need_prediction":
                dtype = np.int8
            arrays[column] = np.array(batch.column(idx).to_numpy(), copy=True, dtype=dtype)

        for column in columns:
            arrays[column] = concat_if_needed(carry[column], arrays[column])

        usable_rows = (arrays["step_in_seq"].shape[0] // SEQUENCE_LENGTH) * SEQUENCE_LENGTH
        for column in columns:
            carry[column] = arrays[column][usable_rows:]
            arrays[column] = arrays[column][:usable_rows]

        if usable_rows == 0:
            continue

        seq_count = usable_rows // SEQUENCE_LENGTH
        steps = arrays["step_in_seq"].reshape(seq_count, SEQUENCE_LENGTH)
        if not np.all(steps == expected_steps):
            raise ValueError(f"{parquet_path} contains malformed sequences")

        raw = np.stack(
            [arrays[column].reshape(seq_count, SEQUENCE_LENGTH) for column in RAW_FEATURE_COLUMNS],
            axis=2,
        ).astype(np.float32, copy=False)
        mask = arrays["need_prediction"].reshape(seq_count, SEQUENCE_LENGTH).astype(bool, copy=False)

        if include_targets:
            targets = np.stack(
                [arrays[column].reshape(seq_count, SEQUENCE_LENGTH) for column in TARGET_COLUMNS],
                axis=2,
            ).astype(np.float32, copy=False)
        else:
            targets = None

        if prepare_fn is not None:
            raw = prepare_fn(raw)

        yield raw, targets, mask

    leftover = carry["step_in_seq"]
    if leftover is not None and leftover.size != 0:
        raise ValueError(f"{parquet_path} ended with a partial sequence")


def engineered_features_numpy(raw_batch: np.ndarray) -> np.ndarray:
    """Raw 32 channels + first differences + 10 microstructure features -> 74 dims.

    Kept identical to the version baked into the exported ONNX graphs, so
    anything computed here matches what the models actually saw.
    """
    raw_batch = np.asarray(raw_batch, dtype=np.float32)
    delta = np.zeros_like(raw_batch)
    delta[:, 1:, :] = raw_batch[:, 1:, :] - raw_batch[:, :-1, :]

    bid_p = raw_batch[:, :, BID_PRICE]
    ask_p = raw_batch[:, :, ASK_PRICE]
    bid_v = raw_batch[:, :, BID_VOLUME]
    ask_v = raw_batch[:, :, ASK_VOLUME]
    trade_p = raw_batch[:, :, TRADE_PRICE]
    trade_v = raw_batch[:, :, TRADE_VOLUME]

    bid_p_mean = bid_p.mean(axis=2, keepdims=True, dtype=np.float32)
    ask_p_mean = ask_p.mean(axis=2, keepdims=True, dtype=np.float32)
    bid_v_sum = bid_v.sum(axis=2, keepdims=True, dtype=np.float32)
    ask_v_sum = ask_v.sum(axis=2, keepdims=True, dtype=np.float32)
    top_bid_v = bid_v[:, :, :1]
    top_ask_v = ask_v[:, :, :1]
    top_bid_p = bid_p[:, :, :1]
    top_ask_p = ask_p[:, :, :1]
    trade_p_mean = trade_p.mean(axis=2, keepdims=True, dtype=np.float32)
    trade_v_sum = trade_v.sum(axis=2, keepdims=True, dtype=np.float32)

    volume_imbalance = (bid_v_sum - ask_v_sum) / (bid_v_sum + ask_v_sum + 1e-3)
    top_volume_imbalance = (top_bid_v - top_ask_v) / (top_bid_v + top_ask_v + 1e-3)
    spread = ask_p_mean - bid_p_mean
    top_spread = top_ask_p - top_bid_p

    engineered = np.concatenate(
        [
            bid_p_mean,
            ask_p_mean,
            spread,
            top_spread,
            bid_v_sum,
            ask_v_sum,
            volume_imbalance,
            top_volume_imbalance,
            trade_p_mean,
            trade_v_sum,
        ],
        axis=2,
    )
    return np.concatenate([raw_batch, delta, engineered], axis=2).astype(np.float32, copy=False)


class LobCorpus:
    """Memory-mapped view over the materialised corpus.

    Arrays are opened with ``mmap_mode='r'``, so constructing this is free and the
    OS page cache does the work. ``raw`` is ~1.56 GB on disk for the pooled
    12,165 sequences.
    """

    def __init__(self, root: str | Path):
        root = Path(root)
        self.root = root
        self.raw = np.load(root / "all_raw.npy", mmap_mode="r")
        self.targets = np.load(root / "all_targets.npy", mmap_mode="r")
        self.mask = np.load(root / "all_mask.npy", mmap_mode="r")
        self.seq_ix = np.load(root / "all_seq_ix.npy")
        self.source = np.load(root / "all_source.npy")  # 0 = train.parquet, 1 = valid.parquet

        stats_path = root / "feature_stats.npz"
        if stats_path.exists():
            stats = np.load(stats_path)
            self.feature_mean = stats["feature_mean"]
            self.feature_std = stats["feature_std"]
        else:
            self.feature_mean = None
            self.feature_std = None

    def __len__(self) -> int:
        return int(self.raw.shape[0])

    @property
    def n_sequences(self) -> int:
        return len(self)

    def scored_steps(self) -> np.ndarray:
        """Step indices that carry a prediction (99..999)."""
        return np.arange(WARMUP_STEPS, SEQUENCE_LENGTH)

    def gather(self, sequence_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Materialise a subset of sequences into RAM as contiguous arrays."""
        sequence_indices = np.asarray(sequence_indices, dtype=np.int64)
        raw = np.ascontiguousarray(self.raw[sequence_indices])
        targets = np.ascontiguousarray(self.targets[sequence_indices])
        mask = np.ascontiguousarray(self.mask[sequence_indices])
        return raw, targets, mask

    def flat_targets(self, sequence_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Scored targets and their sequence ids, flattened in evaluation order."""
        sequence_indices = np.asarray(sequence_indices, dtype=np.int64)
        steps = self.scored_steps()
        targets = np.ascontiguousarray(self.targets[sequence_indices][:, steps, :])
        seq_ids = np.repeat(self.seq_ix[sequence_indices], steps.size)
        return targets.reshape(-1, 2), seq_ids
