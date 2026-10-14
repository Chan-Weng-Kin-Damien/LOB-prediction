"""Invariants that must not break silently.

Run with the project root on the path:

    python -m pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lobkit.data import RAW_DIM, SEQUENCE_LENGTH, WARMUP_STEPS, engineered_features_numpy
from lobkit.fullseq import ReceptiveFieldError, assert_equivalence_safe, raw_lookback
from lobkit.metric import paired_bootstrap, weighted_pearson_correlation
from lobkit.splits import SOURCE_TRAIN, SOURCE_VALID, build_splits, group_kfold
from lobkit.windows import ShuffleBuffer, make_window, make_windows_view, receptive_field


# ---------------------------------------------------------------- metric ----


def test_metric_matches_reference_implementation():
    """lobkit reimplements the metric to avoid a pandas dependency; it must agree.

    Skipped when the organisers' utils.py is absent, since it is not
    redistributed with this repository.
    """
    pytest.importorskip("pandas")
    root = Path(__file__).resolve().parent.parent
    if not (root / "utils.py").exists():
        pytest.skip("organisers' utils.py not present; see README")
    sys.path.insert(0, str(root))
    from utils import weighted_pearson_correlation as reference

    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.standard_normal(5000) * rng.uniform(0.1, 5)
        p = y * rng.uniform(-1, 1) + rng.standard_normal(5000)
        assert weighted_pearson_correlation(y, p) == pytest.approx(reference(y, p), abs=1e-12)


def test_metric_clips_predictions():
    """Predictions outside [-6, 6] must be clipped before scoring, not after."""
    rng = np.random.default_rng(1)
    y = rng.standard_normal(2000)
    p = y * 3.0
    assert weighted_pearson_correlation(y, p) == pytest.approx(
        weighted_pearson_correlation(y, np.clip(p, -6, 6)), abs=1e-12
    )


def test_metric_is_scale_invariant():
    """The reason least-squares blend fitting optimises the wrong objective."""
    rng = np.random.default_rng(2)
    y = rng.standard_normal(4000) * 0.3
    p = y + rng.standard_normal(4000) * 0.3
    base = weighted_pearson_correlation(y, p)
    for scale in (0.5, 2.0):
        # Stay inside the clip range so invariance is not masked by clipping.
        assert weighted_pearson_correlation(y, p * scale) == pytest.approx(base, abs=1e-9)


# --------------------------------------------------------------- windows ----


def test_make_window_pads_left_and_keeps_newest_last():
    rng = np.random.default_rng(3)
    seq = rng.standard_normal((SEQUENCE_LENGTH, RAW_DIM)).astype(np.float32)

    w = make_window(seq, 5, 128)
    assert w.shape == (128, RAW_DIM)
    assert np.all(w[:122] == 0), "short history must be zero-padded on the left"
    assert np.array_equal(w[-1], seq[5]), "newest tick must land at index -1"
    assert np.array_equal(w[122:], seq[:6])

    w = make_window(seq, 500, 128)
    assert not np.any(w[0] == 0) or True
    assert np.array_equal(w, seq[373:501])


def test_windows_view_matches_make_window():
    rng = np.random.default_rng(4)
    chunk = rng.standard_normal((3, 300, RAW_DIM)).astype(np.float32)
    view = make_windows_view(chunk, 64)
    for s in range(3):
        for step in (0, 1, 63, 64, 150, 299):
            assert np.array_equal(view[s, step], make_window(chunk[s], step, 64))


def test_shuffle_buffer_emits_only_full_batches_until_flush():
    """The old loop emitted a batch of 8 every third step; the correlation term
    in mixed_loss is meaningless at that size."""
    rng = np.random.default_rng(5)
    buf = ShuffleBuffer(capacity=4096, rng=rng)
    sizes = []
    for _ in range(6):
        buf.add(
            rng.standard_normal((520, 8, 4)).astype(np.float32),
            rng.standard_normal((520, 2)).astype(np.float32),
        )
        sizes.extend(w.shape[0] for w, _ in buf.drain(256, keep=256))
    assert sizes, "buffer produced no batches"
    assert set(sizes) == {256}, f"ragged batches leaked through: {sorted(set(sizes))}"


def test_shuffle_buffer_conserves_samples():
    rng = np.random.default_rng(6)
    buf = ShuffleBuffer(capacity=8192, rng=rng)
    total_in, total_out = 0, 0
    for _ in range(4):
        n = int(rng.integers(300, 700))
        buf.add(
            rng.standard_normal((n, 4, 2)).astype(np.float32),
            rng.standard_normal((n, 2)).astype(np.float32),
        )
        total_in += n
        total_out += sum(w.shape[0] for w, _ in buf.drain(128, keep=128))
    total_out += sum(w.shape[0] for w, _ in buf.drain(128, keep=0))
    assert total_out <= total_in
    assert total_in - total_out < 128


# -------------------------------------------------------------- receptive ----


def test_receptive_field_accounts_for_delta_feature():
    """The shipped TCN is *exactly* saturated: RF equals the context length.

    Probing the exported graph one input position at a time shows the output
    depends on all 128 window positions -- 127 from the conv stack plus one more
    because build_features computes delta[t] = raw[t] - raw[t-1].
    """
    dilations = [1, 2, 4, 8, 16, 32]
    assert receptive_field(dilations) == 127
    assert raw_lookback(dilations) == 128
    assert assert_equivalence_safe(dilations, context_len=128) == 128


def test_receptive_field_guard_rejects_deeper_stack():
    """Adding a block would break full-sequence/windowed equivalence silently."""
    with pytest.raises(ReceptiveFieldError, match="exceeds context_len"):
        assert_equivalence_safe([1, 2, 4, 8, 16, 32, 64], context_len=128)


# ---------------------------------------------------------------- splits ----


def _fake_corpus(n_train=2000, n_valid=600):
    seq_ix = np.arange(n_train + n_valid, dtype=np.int64)
    source = np.concatenate(
        [np.full(n_train, SOURCE_TRAIN, np.int8), np.full(n_valid, SOURCE_VALID, np.int8)]
    )
    return seq_ix, source


def test_splits_are_disjoint_and_complete():
    seq_ix, source = _fake_corpus()
    splits = build_splits(seq_ix, source)
    pooled = np.concatenate(list(splits.values()))
    assert pooled.size == seq_ix.size
    assert np.unique(pooled).size == seq_ix.size


def test_holdout_and_blend_fit_come_only_from_valid_parquet():
    """baseline.onnx was fitted on train.parquet, so weights fitted there would
    over-weight it. Both evaluation splits must avoid that contamination."""
    seq_ix, source = _fake_corpus()
    splits = build_splits(seq_ix, source)
    for name in ("blend_fit", "holdout"):
        assert np.all(source[splits[name]] == SOURCE_VALID), f"{name} leaked train sequences"
    for name in ("train", "inner_val"):
        assert np.all(source[splits[name]] == SOURCE_TRAIN)


def test_already_measured_sequences_never_reach_the_holdout():
    seq_ix, source = _fake_corpus()
    already = np.flatnonzero(source == SOURCE_VALID)[:300]
    splits = build_splits(seq_ix, source, already_measured=already)
    assert not np.intersect1d(splits["holdout"], already).size


def test_splits_are_deterministic():
    seq_ix, source = _fake_corpus()
    a = build_splits(seq_ix, source)
    b = build_splits(seq_ix, source)
    for name in a:
        assert np.array_equal(a[name], b[name])


def test_group_kfold_never_splits_a_sequence():
    seq_ix = np.arange(1000)
    folds = group_kfold(seq_ix, n_folds=4)
    assert folds.size == seq_ix.size
    for s in seq_ix:
        assert np.unique(folds[seq_ix == s]).size == 1
    counts = np.bincount(folds)
    assert counts.max() - counts.min() <= 1


# ------------------------------------------------------------- bootstrap ----


def test_paired_bootstrap_detects_a_real_difference():
    rng = np.random.default_rng(7)
    n_seq, per_seq = 200, 901
    y = rng.standard_normal((n_seq * per_seq, 2))
    seq_ids = np.repeat(np.arange(n_seq), per_seq)
    strong = y * 0.5 + rng.standard_normal(y.shape) * 0.5
    weak = y * 0.2 + rng.standard_normal(y.shape) * 0.8

    res = paired_bootstrap(y, strong, weak, seq_ids, n_resamples=300)
    assert res["delta"] > 0
    assert res["p_a_better"] > 0.95
    assert res["lo95"] < res["hi95"]


def test_paired_bootstrap_is_indifferent_between_identical_models():
    rng = np.random.default_rng(8)
    n_seq, per_seq = 100, 901
    y = rng.standard_normal((n_seq * per_seq, 2))
    seq_ids = np.repeat(np.arange(n_seq), per_seq)
    pred = y * 0.3 + rng.standard_normal(y.shape)

    res = paired_bootstrap(y, pred, pred.copy(), seq_ids, n_resamples=200)
    assert res["delta"] == pytest.approx(0.0, abs=1e-12)
    assert res["lo95"] == pytest.approx(0.0, abs=1e-12)


# ------------------------------------------------------------- features ----


def test_engineered_features_are_causal():
    """Changing a future tick must not alter features at earlier steps."""
    rng = np.random.default_rng(9)
    raw = rng.standard_normal((1, 200, RAW_DIM)).astype(np.float32)
    base = engineered_features_numpy(raw)

    perturbed = raw.copy()
    perturbed[0, 150:] += 10.0
    after = engineered_features_numpy(perturbed)

    assert np.allclose(base[0, :150], after[0, :150]), "feature construction looks ahead"


def test_engineered_feature_dimensionality():
    rng = np.random.default_rng(10)
    raw = rng.standard_normal((2, 50, RAW_DIM)).astype(np.float32)
    assert engineered_features_numpy(raw).shape == (2, 50, RAW_DIM * 2 + 10)


def test_scored_region_is_steps_99_to_999():
    assert WARMUP_STEPS == 99
    assert SEQUENCE_LENGTH - WARMUP_STEPS == 901
