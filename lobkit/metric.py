"""The competition metric, its differentiable training surrogates, and the
sequence-level bootstrap used to put error bars on every reported number.

The metric is a *weighted* Pearson correlation with weights ``|y_true|``, so it
concentrates on large price moves: on the validation set the top decile of
``|y|`` carries ~37% of the total weight. It is also scale-invariant, which is
why blending weights fitted by least squares are not the same thing as blending
weights that maximise the score.
"""

from __future__ import annotations

import numpy as np

CLIP_LIMIT = 6.0


def weighted_pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Competition metric for a single target column.

    Mirrors ``utils.weighted_pearson_correlation`` exactly; reimplemented here so
    ``lobkit`` stays importable without pandas/tqdm. ``tests/test_metric.py``
    asserts the two agree.
    """
    y_pred_clipped = np.clip(y_pred, -CLIP_LIMIT, CLIP_LIMIT)

    weights = np.maximum(np.abs(y_true), 1e-8)
    sum_w = weights.sum()
    if sum_w == 0:
        return 0.0

    mean_true = (y_true * weights).sum() / sum_w
    mean_pred = (y_pred_clipped * weights).sum() / sum_w

    dev_true = y_true - mean_true
    dev_pred = y_pred_clipped - mean_pred

    cov = (weights * dev_true * dev_pred).sum() / sum_w
    var_true = (weights * dev_true**2).sum() / sum_w
    var_pred = (weights * dev_pred**2).sum() / sum_w

    if var_true <= 0 or var_pred <= 0:
        return 0.0

    return float(cov / (np.sqrt(var_true) * np.sqrt(var_pred)))


def score_both_targets(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Score an ``(N, 2)`` prediction block the way the leaderboard would."""
    scores = {}
    for idx, name in enumerate(("t0", "t1")):
        scores[name] = weighted_pearson_correlation(y_true[:, idx], y_pred[:, idx])
    scores["weighted_pearson"] = float(np.mean([scores["t0"], scores["t1"]]))
    return scores


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------
# Ticks within one sequence are heavily autocorrelated, so resampling ticks
# would understate the true uncertainty by an order of magnitude. Sequences are
# independent by construction (the organisers shuffled them), which makes the
# sequence the correct resampling unit.


def _mean_wp(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return 0.5 * (
        weighted_pearson_correlation(y_true[:, 0], y_pred[:, 0])
        + weighted_pearson_correlation(y_true[:, 1], y_pred[:, 1])
    )


def _sequence_offsets(seq_ids: np.ndarray) -> list[np.ndarray]:
    """Row indices grouped by sequence, preserving order within each group."""
    order = np.argsort(seq_ids, kind="stable")
    sorted_ids = seq_ids[order]
    boundaries = np.flatnonzero(np.diff(sorted_ids)) + 1
    return np.split(order, boundaries)


def bootstrap_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    seq_ids: np.ndarray,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict:
    """Sequence-level block bootstrap for a single model's score."""
    groups = _sequence_offsets(seq_ids)
    rng = np.random.default_rng(seed)
    n_groups = len(groups)

    draws = np.empty(n_resamples)
    for i in range(n_resamples):
        picked = np.concatenate([groups[j] for j in rng.integers(0, n_groups, n_groups)])
        draws[i] = _mean_wp(y_true[picked], y_pred[picked])

    return {
        "point": _mean_wp(y_true, y_pred),
        "mean": float(draws.mean()),
        "std": float(draws.std()),
        "lo95": float(np.percentile(draws, 2.5)),
        "hi95": float(np.percentile(draws, 97.5)),
        "n_sequences": n_groups,
    }


def paired_bootstrap(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    seq_ids: np.ndarray,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict:
    """Paired bootstrap of ``score(a) - score(b)``.

    Use this, not two independent CIs, whenever comparing models. The candidates
    here correlate at 0.66-0.88, so resampling the *same* sequences for both
    cancels most of the shared variance: on 300 sequences the paired delta CI is
    about +/-0.010 where the CI on an absolute score is about +/-0.045.
    """
    groups = _sequence_offsets(seq_ids)
    rng = np.random.default_rng(seed)
    n_groups = len(groups)

    deltas = np.empty(n_resamples)
    for i in range(n_resamples):
        picked = np.concatenate([groups[j] for j in rng.integers(0, n_groups, n_groups)])
        yt = y_true[picked]
        deltas[i] = _mean_wp(yt, pred_a[picked]) - _mean_wp(yt, pred_b[picked])

    point = _mean_wp(y_true, pred_a) - _mean_wp(y_true, pred_b)
    return {
        "delta": point,
        "mean": float(deltas.mean()),
        "std": float(deltas.std()),
        "lo95": float(np.percentile(deltas, 2.5)),
        "hi95": float(np.percentile(deltas, 97.5)),
        "p_a_better": float(np.mean(deltas > 0)),
        "n_sequences": n_groups,
    }
