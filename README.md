# Limit order book prediction — Wunder Fund RNN Challenge

Short-horizon prediction of two forward price-movement indicators from limit
order book and trade data, scored by amplitude-weighted Pearson correlation.
Finished 417th of 5,062 (January–March 2026).

This repository holds two submissions. `submission_safe_3model/` is the entry
scored during the competition: a weighted blend of a provided GRU baseline, a
DeepLOB-style attention model and a causal TCN. `submission_v2/` is a later
revision in which the TCN was retrained and the blend refitted; it scores higher
on held-out data but was produced after the leaderboard closed, so it was never
scored by the organisers.

## Task

| | |
|---|---|
| Input | 32 channels per step: six bid and six ask price levels, their twelve volumes, four trade prices, four trade volumes |
| Book depth | Level 6 per side |
| Sequences | 12,165 of exactly 1,000 steps (10,721 train, 1,444 validation), mutually independent and pre-shuffled |
| Scored region | steps 99–999, so 901 predictions per sequence |
| Targets | `t0` and `t1`, two anonymised continuous indicators. No prediction horizon is disclosed, so this is regression rather than direction classification |
| Metric | Pearson correlation weighted by `abs(y_true)`, predictions clipped to [-6, 6], averaged over the two targets |

The amplitude weighting concentrates the score: the largest decile of `abs(y)`
carries about 37 per cent of the total weight.

## Results

Measured by `scripts/evaluate.py`. Saved outputs are in `artifacts/`.

On the sealed holdout of 544 sequences (490,144 scored ticks), opened once after
all models, epochs and blend weights were frozen:

| model | t0 | t1 | mean |
|---|---|---|---|
| provided GRU baseline | +0.3886 | +0.1079 | 0.2482 |
| DeepLOB-style attention | +0.3203 | +0.1304 | 0.2254 |
| TCN, as originally trained | +0.3324 | +0.0802 | 0.2063 |
| TCN, retrained, three seeds averaged | +0.4095 | +0.1217 | 0.2656 |
| competition submission (`submission_safe_3model`) | +0.3947 | +0.1233 | 0.2590 |
| revised submission (`submission_v2`) | +0.4108 | +0.1217 | 0.2663 |

Against the provided baseline the revised submission gains +0.0180, with a
95 per cent paired bootstrap interval of [+0.0075, +0.0263]. Against the
competition submission it gains +0.0073, interval [+0.0007, +0.0149].

On the 900-sequence validation split used for fitting blend weights, retraining
the TCN moved it from 0.2163 to 0.2713.

Both splits are reported because they answer different questions: the validation
split is where decisions were made, and the holdout is the only untouched
estimate.

## Method

### Full-sequence training of the TCN

The TCN's output head is a width-1 convolution, so it already emits a prediction
at every timestep. The original training loop sampled eight windows per sequence
and kept only the final position of each, discarding the rest — under 2 per cent
of the available labels.

The network is fully causal, so one forward pass over a zero-prefixed sequence
reproduces the windowed prediction at every step. Training on whole sequences
therefore supervises all 901 scored steps instead of 8, roughly 112 times more
targets per sequence, at essentially unchanged cost per epoch.

That equivalence holds only while the receptive field fits inside the context
window, and here it is exactly saturated:

```
dilations [1, 2, 4, 8, 16, 32], kernel 3   ->  1 + 2 * 63 = 127
delta features add one step of lookback    ->  +1
                                     total =  128 = context length
```

There is no slack, so a seventh dilation block would silently break the
correspondence between training and scoring. `lobkit.fullseq` makes this a hard
failure, and three gates run before any export:

```
receptive field 128 <= context 128
max |full-sequence - windowed| = 6e-08
max |torch - onnx|             = 7e-07
```

### Evaluation protocol

A sequence-level block bootstrap gives a standard deviation of about 0.026 on a
single score over 300 sequences, which is large next to the effects being
measured. Model comparisons therefore resample whole sequences and are paired,
which cancels the shared variance and tightens the interval roughly fourfold.

Splits are partitioned by provenance before purpose. The provided baseline was
almost certainly fitted on `train.parquet`, so blend weights estimated on those
sequences would be biased towards it:

```
train.parquet 10,721  ->  TRAIN      9,721   our models only
                          INNER-VAL  1,000   epoch selection
valid.parquet  1,444  ->  BLEND-FIT    900   ensemble weights, reported comparisons
                          HOLDOUT      544   sealed
```

Splits are derived from a hash of the sequence index, so they are stable across
machines.

Purging and embargo are not applied, and the reason is specific to this data.
Both exist to remove leakage from overlapping forward label horizons, or serial
correlation across a boundary, within one continuous series. Here the unit is an
entire independent, pre-shuffled sequence, so no sample in one split shares a
tick or any feature history with a sample in another: grouping by sequence is
itself the purge. Splitting at tick level would not be sound, because windows
within a sequence overlap by up to 127 steps.

The metric is a pooled correlation, so out-of-fold predictions are concatenated
and scored once rather than averaged across folds.

### Blend weights

Weights are fitted per target by maximising the competition metric directly over
the probability simplex, with shrinkage chosen by sequence-grouped
cross-validation. Fitting per target reflects that `t0` and `t1` have very
different achievable skill, not that they have different horizons.

Two results from this are negative and are kept in the repository as ablations.
First, when the base models were weak and collinear, every fitted blend scored
below the strongest single member out-of-fold. Second, the closed-form
correlation maximiser — which is the weighted generalised least squares solution,
since the metric is scale-invariant — transferred badly in that regime, because
inverting a near-singular covariance over collinear predictors amplifies noise.
It became competitive only once member skill was balanced.

In the revised submission the DeepLOB model receives zero weight on both targets
under every fitting method tried, so it is not included.

## Inference

The scorer calls `predict` one tick at a time, so latency is reported the same
way: one sequence at a time, single-threaded, `CPUExecutionProvider`, all graph
optimisations enabled.

| component | mean | p50 | p95 | p99 |
|---|---|---|---|---|
| provided GRU baseline (context 100) | 316 us | 313 | 328 | 342 |
| TCN, per seed (context 128) | 420-428 us | 416-419 | 437-446 | 450-524 |
| `submission_v2` as shipped | 1,562 us | 1,550 | 1,616 | 1,703 |
| non-prediction tick | 1.1 us | 1.1 | 1.2 | 2.3 |
| window materialisation | 10.1 us | 9.9 | 11.4 | 12.6 |

About 640 predicted ticks per second. Window construction is well under 1 per
cent of a step, so the ONNX graphs dominate and replacing the Python history
buffer with a ring buffer would recover at most about 10 us.

Averaging three seeds accounts for the entire cost increase over the competition
submission. The three seeds are fed from one shared history buffer rather than
three duplicates, which saves 28 us per tick.

This run was taken at a one-minute load average of 4.5 on eight cores, which
`scripts/benchmark.py` flags as contended. It is reported because the baseline
component read 316 us against 311 us on an idle machine, within 2 per cent, and
the per-seed figures match the same graph measured separately. Under genuine
contention the inflation is much larger: the earlier three-model ensemble
measured 1,085 us idle and 2,135 us while an evaluation job shared the host.

## Layout

```
lobkit/          shared library: metric, bootstrap, splits, windows, blending,
                 full-sequence equivalence guards, TCN definition
scripts/         materialise, train_tcn, export_tcn, evaluate, fit_blend, benchmark
tests/           19 tests: metric parity, causality, padding alignment,
                 split disjointness, receptive-field guard, bootstrap behaviour
submission_safe_3model/   the entry scored in the competition
submission_v2/            the later revision
artifacts/       evaluation reports, blend weights, benchmark output, exported ONNX
RESULTS.md       longer technical note, including what was tried and rejected
```

`lobkit` is for training and evaluation only. Both `solution.py` files are
self-contained so that a submission archive runs with nothing but NumPy and ONNX
Runtime.

## Reproducing

The datasets are competition property and are not redistributed here. Place
`train.parquet` and `valid.parquet` in `datasets/`, along with the organisers'
`utils.py`, which supplies the `DataPoint` class and the reference scorer.

```bash
python scripts/materialise.py --train-path datasets/train.parquet
python -m pytest tests/ -q
python scripts/train_tcn.py --epochs 12 --seed 1 --tag s1
python scripts/export_tcn.py --checkpoint artifacts/tcn_s1.pt
python scripts/evaluate.py --split blend_fit --save-predictions
python scripts/fit_blend.py --split blend_fit
python scripts/benchmark.py
```

Training a seed takes about three minutes on an Apple M3 via MPS; the models are
roughly 62,000 parameters, so no GPU cluster is required. Run the benchmark on an
otherwise idle machine. The sealed holdout requires `--split holdout
--i-am-done`, which is deliberately awkward.

## Provenance and caveats

- `baseline.onnx` and the single-threaded ONNX Runtime session settings come from
  the organisers' starter pack, as does the rolling-window streaming pattern the
  submissions build on.
- `submission_v2` was produced after the leaderboard closed, so the placement
  above reflects `submission_safe_3model` only.
- The int8 dynamic quantisation result described in `RESULTS.md` was measured
  outside this repository; no script or quantised model is committed, so it is
  not reproducible from here.
