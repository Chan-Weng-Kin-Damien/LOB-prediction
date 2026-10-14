"""Split construction.

Why not a single pooled K-fold over all 12,165 sequences: ``baseline.onnx`` is
the organisers' model and was almost certainly fitted on ``train.parquet``. Its
predictions on those sequences are in-sample, so any ensemble weight estimated
there systematically over-weights it. That is a real leakage channel and a
plausible partial reason the shipped blend scores *below* the baseline it is
built on.

So the corpus is partitioned by provenance first, purpose second::

    train.parquet 10,721 -+- TRAIN      9,721   our models only
                          +- INNER_VAL  1,000   epoch selection, early stopping

    valid.parquet  1,444 -+- BLEND_FIT    900   ensemble weights
                          +- HOLDOUT      544   sealed, opened once

Splits are derived from a hash of ``seq_ix``, so they are stable across machines
and reruns and do not depend on file ordering.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

TRAIN = "train"
INNER_VAL = "inner_val"
BLEND_FIT = "blend_fit"
HOLDOUT = "holdout"

SOURCE_TRAIN = 0
SOURCE_VALID = 1

# Sequences already used for exploratory measurement. They live in BLEND_FIT so
# the holdout stays untouched. These are the first 300 sequences of valid.parquet
# in file order, recorded here by index rather than id.
N_ALREADY_MEASURED = 300


def _hash_fraction(seq_ix: int, salt: str) -> float:
    """Deterministic uniform-ish value in [0, 1) for one sequence id."""
    digest = hashlib.sha256(f"{salt}:{int(seq_ix)}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


@dataclass
class SplitSpec:
    inner_val_sequences: int = 1000
    holdout_sequences: int = 544
    salt: str = "wunder-lob-v1"


def build_splits(
    seq_ix: np.ndarray,
    source: np.ndarray,
    already_measured: np.ndarray | None = None,
    spec: SplitSpec | None = None,
) -> dict[str, np.ndarray]:
    """Assign every sequence to exactly one split.

    Returns a dict of split name -> array of *positional* indices into the
    corpus arrays (not ``seq_ix`` values).
    """
    spec = spec or SplitSpec()
    seq_ix = np.asarray(seq_ix)
    source = np.asarray(source)

    train_pos = np.flatnonzero(source == SOURCE_TRAIN)
    valid_pos = np.flatnonzero(source == SOURCE_VALID)

    # --- train.parquet -> TRAIN / INNER_VAL -----------------------------------
    # Cap the inner-val carve-out so a small or truncated corpus cannot end up
    # with an empty TRAIN split (which silently produced no feature statistics).
    inner_n = min(spec.inner_val_sequences, max(0, train_pos.size // 4))
    train_scores = np.array([_hash_fraction(s, spec.salt + ":inner") for s in seq_ix[train_pos]])
    inner_cut = np.argsort(train_scores, kind="stable")[:inner_n]
    inner_mask = np.zeros(train_pos.size, dtype=bool)
    inner_mask[inner_cut] = True

    splits = {
        TRAIN: train_pos[~inner_mask],
        INNER_VAL: train_pos[inner_mask],
    }

    # --- valid.parquet -> BLEND_FIT / HOLDOUT ---------------------------------
    # Anything already looked at is pinned into BLEND_FIT first; the holdout is
    # then drawn from what remains, so it has genuinely never been evaluated.
    pinned = np.zeros(valid_pos.size, dtype=bool)
    if already_measured is not None and already_measured.size:
        pinned[np.isin(valid_pos, already_measured)] = True

    eligible = np.flatnonzero(~pinned)
    holdout_n = min(spec.holdout_sequences, eligible.size)
    eligible_scores = np.array(
        [_hash_fraction(s, spec.salt + ":holdout") for s in seq_ix[valid_pos[eligible]]]
    )
    holdout_local = eligible[np.argsort(eligible_scores, kind="stable")[:holdout_n]]

    holdout_mask = np.zeros(valid_pos.size, dtype=bool)
    holdout_mask[holdout_local] = True

    splits[HOLDOUT] = valid_pos[holdout_mask]
    splits[BLEND_FIT] = valid_pos[~holdout_mask]

    _assert_disjoint(splits, total=seq_ix.size)
    return splits


def _assert_disjoint(splits: dict[str, np.ndarray], total: int) -> None:
    seen = np.concatenate(list(splits.values()))
    if seen.size != total:
        raise ValueError(f"splits cover {seen.size} sequences, expected {total}")
    if np.unique(seen).size != total:
        raise ValueError("splits overlap: a sequence was assigned twice")


def group_kfold(seq_ix: np.ndarray, n_folds: int = 4, salt: str = "wunder-lob-v1") -> np.ndarray:
    """Fold assignment for a set of sequences, grouped by ``seq_ix``.

    Purging and embargo are deliberately absent. Both exist to remove
    *intra-series* leakage -- overlapping forward label horizons, and serial
    correlation across a split boundary in one continuous series. Here the fold
    unit is an entire independent, pre-shuffled 1000-step sequence, so no sample
    in one fold shares a tick, a label horizon, or any feature history with a
    sample in another. Grouping by sequence *is* the purge.

    What would still be unsound is splitting at tick level: windows within a
    sequence overlap by up to ``context_len - 1`` steps.
    """
    scores = np.array([_hash_fraction(s, salt + ":kfold") for s in np.asarray(seq_ix)])
    order = np.argsort(scores, kind="stable")
    folds = np.empty(scores.size, dtype=np.int32)
    folds[order] = np.arange(scores.size) % n_folds
    return folds


def save_splits(path: str | Path, splits: dict[str, np.ndarray], seq_ix: np.ndarray) -> None:
    payload = {
        name: {"n": int(idx.size), "seq_ix": [int(v) for v in seq_ix[idx]]}
        for name, idx in splits.items()
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_splits(path: str | Path, seq_ix: np.ndarray) -> dict[str, np.ndarray]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    lookup = {int(v): i for i, v in enumerate(seq_ix)}
    return {
        name: np.array([lookup[int(v)] for v in body["seq_ix"]], dtype=np.int64)
        for name, body in payload.items()
    }


def describe(splits: dict[str, np.ndarray]) -> str:
    lines = ["split          sequences   scored ticks"]
    for name in (TRAIN, INNER_VAL, BLEND_FIT, HOLDOUT):
        if name in splits:
            n = splits[name].size
            lines.append(f"{name:<14} {n:>9,} {n * 901:>14,}")
    return "\n".join(lines)
