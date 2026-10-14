"""Corpus evaluation of exported ONNX models, plus the diagnostic breakdowns.

Predictions are always returned flattened in a fixed order -- sequences in the
order given, scored steps 99..999 ascending -- so prediction blocks from
different models are directly comparable row by row, which is what the paired
bootstrap needs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import LobCorpus, WARMUP_STEPS, SEQUENCE_LENGTH, engineered_features_numpy
from .metric import score_both_targets, weighted_pearson_correlation
from .runtime import batched_window_predict, build_session


@dataclass
class ModelSpec:
    name: str
    onnx_path: Path
    context_len: int

    @classmethod
    def from_config(cls, name: str, config_path: Path) -> "ModelSpec":
        config = json.loads(Path(config_path).read_text(encoding="utf-8"))
        return cls(
            name=name,
            onnx_path=Path(config_path).with_name(config["onnx_path"]),
            context_len=int(config["context_len"]),
        )


def predict_corpus(
    spec: ModelSpec,
    corpus: LobCorpus,
    sequence_indices: np.ndarray,
    batch_size: int = 512,
    progress_every: int = 200,
) -> np.ndarray:
    """Run one model over a set of sequences. Returns ``(n_seq * 901, 2)``."""
    session = build_session(spec.onnx_path)
    steps = np.arange(WARMUP_STEPS, SEQUENCE_LENGTH)
    sequence_indices = np.asarray(sequence_indices, dtype=np.int64)

    out = np.empty((sequence_indices.size * steps.size, 2), dtype=np.float32)
    for i, seq_pos in enumerate(sequence_indices):
        sequence = np.ascontiguousarray(corpus.raw[seq_pos], dtype=np.float32)
        out[i * steps.size : (i + 1) * steps.size] = batched_window_predict(
            session, sequence, steps, spec.context_len, batch_size=batch_size
        )
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  {spec.name}: {i + 1}/{sequence_indices.size} sequences", flush=True)
    return out


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------


def score_by_amplitude_decile(y_true: np.ndarray, y_pred: np.ndarray) -> list[dict]:
    """Where the score is earned, per target.

    The metric weights by ``|y_true|``, and on this data the top decile of ``|y|``
    carries ~37% of the total weight. A model can look fine overall and still be
    losing exactly where the score lives, so this breakdown is worth more than
    the headline number.
    """
    rows = []
    for idx, name in enumerate(("t0", "t1")):
        amplitude = np.abs(y_true[:, idx])
        edges = np.percentile(amplitude, np.arange(0, 101, 10))
        total_weight = np.maximum(amplitude, 1e-8).sum()
        for d in range(10):
            lo, hi = edges[d], edges[d + 1]
            sel = (amplitude >= lo) & (amplitude <= hi if d == 9 else amplitude < hi)
            if sel.sum() < 100:
                continue
            rows.append(
                {
                    "target": name,
                    "decile": d + 1,
                    "abs_y_lo": float(lo),
                    "abs_y_hi": float(hi),
                    "n": int(sel.sum()),
                    "weight_share": float(np.maximum(amplitude[sel], 1e-8).sum() / total_weight),
                    "wp": weighted_pearson_correlation(y_true[sel, idx], y_pred[sel, idx]),
                }
            )
    return rows


def score_by_position(y_true: np.ndarray, y_pred: np.ndarray, n_sequences: int) -> list[dict]:
    """Score as a function of how much context the model has seen.

    Flat or declining here is evidence *against* investing in longer-context
    models, which is why the t1 specialist at context 256 was deprioritised.
    """
    steps_per_seq = y_true.shape[0] // n_sequences
    yt = y_true.reshape(n_sequences, steps_per_seq, 2)
    yp = y_pred.reshape(n_sequences, steps_per_seq, 2)

    rows = []
    for lo, hi in [(0, 50), (50, 150), (150, 300), (300, 500), (500, 700), (700, steps_per_seq)]:
        a = yt[:, lo:hi].reshape(-1, 2)
        b = yp[:, lo:hi].reshape(-1, 2)
        rows.append(
            {
                "step_lo": WARMUP_STEPS + lo,
                "step_hi": WARMUP_STEPS + hi,
                "n": int(a.shape[0]),
                "t0": weighted_pearson_correlation(a[:, 0], b[:, 0]),
                "t1": weighted_pearson_correlation(a[:, 1], b[:, 1]),
            }
        )
    return rows


def score_by_regime(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    corpus: LobCorpus,
    sequence_indices: np.ndarray,
    feature: str = "spread",
    n_buckets: int = 5,
) -> list[dict]:
    """Score conditioned on a microstructure regime.

    ``feature`` is one of the engineered channels: ``spread``, ``top_spread``,
    ``volume_imbalance``, ``top_volume_imbalance``.
    """
    engineered_index = {
        "bid_p_mean": 64,
        "ask_p_mean": 65,
        "spread": 66,
        "top_spread": 67,
        "bid_v_sum": 68,
        "ask_v_sum": 69,
        "volume_imbalance": 70,
        "top_volume_imbalance": 71,
        "trade_p_mean": 72,
        "trade_v_sum": 73,
    }[feature]

    steps = np.arange(WARMUP_STEPS, SEQUENCE_LENGTH)
    values = []
    for seq_pos in np.asarray(sequence_indices, dtype=np.int64):
        raw = np.ascontiguousarray(corpus.raw[seq_pos], dtype=np.float32)[None]
        values.append(engineered_features_numpy(raw)[0, steps, engineered_index])
    values = np.concatenate(values)

    edges = np.percentile(values, np.linspace(0, 100, n_buckets + 1))
    rows = []
    for b in range(n_buckets):
        lo, hi = edges[b], edges[b + 1]
        sel = (values >= lo) & (values <= hi if b == n_buckets - 1 else values < hi)
        if sel.sum() < 500:
            continue
        rows.append(
            {
                "feature": feature,
                "bucket": b + 1,
                "lo": float(lo),
                "hi": float(hi),
                "n": int(sel.sum()),
                "t0": weighted_pearson_correlation(y_true[sel, 0], y_pred[sel, 0]),
                "t1": weighted_pearson_correlation(y_true[sel, 1], y_pred[sel, 1]),
            }
        )
    return rows


def prediction_correlations(predictions: dict[str, np.ndarray]) -> dict:
    """Pairwise correlation between model outputs, per target.

    Ensemble value comes from decorrelation, not from individual skill: a weaker
    but less-correlated member can contribute more than a stronger, redundant
    one.
    """
    names = list(predictions)
    result = {}
    for idx, target in enumerate(("t0", "t1")):
        stacked = np.stack([predictions[n][:, idx] for n in names])
        matrix = np.corrcoef(stacked)
        result[target] = {
            f"{names[i]}-{names[j]}": float(matrix[i, j])
            for i in range(len(names))
            for j in range(i + 1, len(names))
        }
    return result


def summarise(y_true: np.ndarray, predictions: dict[str, np.ndarray]) -> list[dict]:
    return [
        {"model": name, **score_both_targets(y_true, pred)} for name, pred in predictions.items()
    ]
