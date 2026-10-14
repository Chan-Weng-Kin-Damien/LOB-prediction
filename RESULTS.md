# Short-horizon LOB prediction — technical note

Wunder Challenge 2 (WundeRNN). Predict two anonymised forward price-movement
indicators, `t0` and `t1`, from a Level-6 limit order book plus trade channels.
Scored by weighted Pearson correlation with weights `|y_true|`, predictions
clipped to `[-6, 6]`, averaged over the two targets.

All numbers below are measured, reproducible from this repo, and reported with
sequence-level bootstrap confidence intervals.

---

## 1. What the data is

| | |
|---|---|
| Sequences | 12,165 (10,721 train + 1,444 valid), each exactly 1000 steps |
| Scored region | steps 99–999 (901 per sequence); 0–98 are warm-up |
| Features | 32 channels: `p0–p5` bid prices, `p6–p11` ask prices, `v0–v5` bid volumes, `v6–v11` ask volumes, `dp0–dp3` trade prices, `dv0–dv3` trade volumes |
| Book depth | **Level 6 per side** — not top-of-book, not L10 |
| Targets | `t0`, `t1` — continuous, anonymised. **No horizon in ticks is disclosed anywhere in the data or the brief.** This is regression, not direction classification |
| Metric | mean over targets of weighted Pearson, weights `|y_true|` |

The metric's amplitude weighting matters more than it first appears: **the top
decile of `|y|` carries ~37% of the total metric weight.** Measured breakdown of
the baseline model by decile:

| `|y|` decile | 1 | 4 | 7 | 9 | 10 |
|---|---|---|---|---|---|
| weight share (t0) | 1.0% | 4.2% | 11.0% | 18.3% | **37.8%** |
| wP (t1) | +0.002 | +0.053 | +0.082 | +0.110 | **+0.195** |

The `t1` row is the useful one: skill rises monotonically with move size. The
models are not uniformly weak on `t1` — they are weak on small `t1` moves, which
the metric barely counts, and reach ~0.195 where the score actually lives.

---

## 2. The measurement problem, and why it had to be solved first

A sequence-level block bootstrap (resampling whole sequences, since the 901 ticks
inside one are heavily autocorrelated) gives a **standard deviation of 0.0256 on
a single score over 300 sequences**. The effects worth chasing are 0.002–0.01.

So comparisons are made *paired* — resampling the same sequences for both models
— which cancels the shared variance. On 300 sequences the paired delta interval
is about ±0.010 where the interval on an absolute score is ±0.045.

**Nothing below 0.005 is claimed as a result.**

### Split design

`baseline.onnx` is the organisers' model and was almost certainly fitted on
`train.parquet`. Any ensemble weight estimated on those sequences is
contaminated by its in-sample advantage — a concrete leakage channel, and a
plausible partial reason the shipped blend scored *below* the baseline it was
built on. So the corpus is partitioned by provenance first:

```
train.parquet 10,721 ─┬─ TRAIN      9,721   our models only
                      └─ INNER-VAL  1,000   epoch selection
valid.parquet  1,444 ─┬─ BLEND-FIT    900   ensemble weights, all reported comparisons
                      └─ HOLDOUT      544   sealed
```

Splits are derived from a hash of `seq_ix`, so they are stable across machines.

**On purging and embargo:** not required here, and it is worth being precise
about why. Purging exists to remove leakage from overlapping forward label
horizons within one continuous series; embargo removes serial correlation across
a split boundary. Here the fold unit is an entire independent, pre-shuffled
1000-step sequence, so no sample in one split shares a tick, a label horizon, or
any feature history with a sample in another. **Grouping by `seq_ix` is the
purge.** What would be unsound is splitting at tick level, since windows within a
sequence overlap by up to `context_len − 1` steps.

Also: the metric is a *pooled* correlation, so out-of-fold predictions are
concatenated and scored once. Averaging per-fold correlations is a different and
biased statistic.

---

## 3. Starting point

Measured on BLEND-FIT (900 sequences, 810,900 scored ticks):

| model | t0 | t1 | mean | 95% CI |
|---|---|---|---|---|
| baseline (organisers') | +0.38865 | +0.14989 | **0.26927** | [0.25100, 0.29375] |
| DeepLOB-attention | +0.33793 | +0.11908 | 0.22850 | [0.22192, 0.23546] |
| TCN | +0.32557 | +0.10700 | 0.21629 | [0.20109, 0.23429] |
| shipped 3-model blend | +0.39298 | +0.14504 | 0.26901 | [0.25383, 0.28846] |

**The shipped ensemble was statistically indistinguishable from its own strongest
component** (paired delta −0.00026, P(better) = 0.516) — and that strongest
component was the model the organisers supplied.

---

## 4. The main finding: the TCN had never trained

The TCN's head is `Conv1d(hidden, 2, kernel_size=1)` — it already emits a
prediction at *every* timestep. The original training loop then discarded all but
one:

```python
x = self.head(x)      # (B, 2, T)  — a prediction at each of 1000 steps
x = x[:, :, -1]       # (B, 2)     — 999 of them thrown away
```

Combined with `--samples-per-seq 8` and 2 epochs, training saw **8 of 901 scorable
steps per sequence** — 85,768 targets per epoch, 670 optimizer steps in total.

Because the network is fully causal, one forward pass over a zero-prefixed
sequence reproduces the windowed prediction at every step, so those discarded
outputs are free supervision: **901 targets per sequence instead of 8 — 112×
more — at essentially unchanged FLOPs** (8,758,621 per epoch on the TRAIN split).

### The equivalence is exact, and exactly saturated

Probing the exported ONNX graph one input position at a time shows the output
depends on **all 128** window positions:

```
conv stack, dilations [1,2,4,8,16,32], kernel 3   ->  1 + 2·63 = 127
build_features computes delta[t] = raw[t] − raw[t−1]  ->  +1
                                        total raw lookback = 128 = context_len
```

There is **zero slack**. A seventh dilation block would take the receptive field
to 255 and silently break the correspondence between how the model trains and how
it is scored. `lobkit.fullseq.assert_equivalence_safe` makes that a hard failure,
and a numerical gate runs before every training job:

```
gate 1/3  receptive field 128 <= context 128                  OK
gate 2/3  max|full-sequence − windowed| = 5.960e-08           OK
gate 3/3  max|torch − onnx|             = 7.153e-07           OK
```

### Result

| TCN | t0 | t1 | mean (BLEND-FIT) |
|---|---|---|---|
| original (8 targets/sequence) | +0.32557 | +0.10700 | 0.21629 |
| **full-sequence retrained** | **+0.39100** | **+0.15163** | **0.27132** |

**+0.055** from the supervision change alone. The retrained model matches the
organisers' baseline (paired delta +0.00205, P = 0.704 — a tie, but it is now
*our* model at that level rather than the supplied one).

Training cost: 15 s/epoch on an Apple M3 via MPS, ~61,922 parameters. No GPU
cluster was needed. The model peaks at **epoch 7** and then overfits hard
(ending at 0.207 after 60 epochs), so the useful schedule is ~12 epochs.

---

## 5. Ensembling only starts working once the members are competent

Blend weights fitted on BLEND-FIT with grouped 4-fold CV, comparing three
strategies out-of-fold.

**Before retraining** (baseline + DeepLOB + original TCN):

| blend | mean |
|---|---|
| baseline alone | 0.26927 |
| shipped weights | 0.26901 |
| simplex refit, OOF | 0.26579 |
| GLS closed form, OOF | 0.25512 |

Every fitted blend was *worse* than just using the strongest model. Two reasons,
both worth stating:

1. Fitting six weights on a few hundred sequences is itself an overfitting
   operation when the members are weak and collinear (baseline–DeepLOB prediction
   correlation 0.88).
2. **The shrinkage prior matters more than the shrinkage strength.** Shrinking
   toward *uniform* weights encodes "the models are equally good" — they were not
   (0.269 / 0.229 / 0.216) — so heavy shrinkage walked the blend toward the weak
   members. Shrinking toward the best single model makes the high-shrinkage limit
   "use the strongest model", and the `t1` fit then collapsed to 99.7% baseline
   on its own, i.e. the data said *don't ensemble `t1`*.

**After retraining** (baseline + DeepLOB + full-sequence TCN):

| blend | t0 | t1 | mean |
|---|---|---|---|
| baseline alone | +0.38865 | +0.14989 | 0.26927 |
| shipped weights | +0.39298 | +0.14504 | 0.26901 |
| simplex refit, OOF | +0.39702 | +0.14854 | 0.27278 |
| **GLS closed form, OOF** | +0.40391 | +0.14911 | **0.27651** |

Fitted weights are now ~0.66 baseline / 0.34 TCN on **both** targets, with
DeepLOB at zero on both.

The reversal is the interesting part. The unconstrained closed-form maximiser
(`w ∝ Σ⁻¹σ`, which is the correlation maximiser because the metric is
scale-invariant) *lost* badly before — in-sample 0.40707 → out-of-fold 0.39812 on
t0, with weights like `[2.39, −0.41, 0.38]` — because a near-singular inversion
over collinear, unequal predictors amplifies noise. Once the members are
comparable in skill it becomes the best option (P = 0.974 vs shipped).

**The lesson is not "constrain your blends". It is that the right amount of
constraint depends on the conditioning of the base models, and that has to be
measured out-of-fold rather than assumed.**

### Seed averaging

Three independently seeded 12-epoch runs, predictions averaged. Measured on
BLEND-FIT:

| | mean |
|---|---|
| best single seed | 0.27600 |
| **3-seed average** | **0.28368** |

+0.008 for 3× the inference cost — the largest reliable gain per unit of effort
in the whole exercise, and the one with a known sign. Weight averaging is not an
option here: independently initialised runs land in different basins, so the
seeds must be averaged in prediction space.

Final fitted blend, BLEND-FIT: **0.17 baseline + 0.83 TCN-average** on both
targets, scoring 0.28446 — versus 0.26901 for the shipped submission
(+0.01545, 95% CI [+0.00966, +0.02064], P = 1.000). Note the inversion: the
organisers' baseline carried 89% of the old t0 blend and now carries 17%.

---

## 6. Sealed holdout — the one unbiased number

544 sequences (490,144 scored ticks) held out from the start, evaluated once,
after the model, the epoch, the seeds and the blend weights were all frozen.

| model | t0 | t1 | mean |
|---|---|---|---|
| baseline (organisers') | +0.38861 | +0.10785 | 0.24823 |
| previous shipped blend | +0.39467 | +0.12332 | 0.25899 |
| TCN 3-seed average | +0.40953 | +0.12170 | 0.26562 |
| **submission_v2 (final)** | **+0.41084** | **+0.12173** | **0.26628** |

| comparison | delta | 95% CI | P(better) |
|---|---|---|---|
| final vs previous submission | **+0.00729** | [+0.00074, +0.01486] | 0.988 |
| final vs organisers' baseline | **+0.01805** | [+0.00745, +0.02626] | 1.000 |

**Reading the gap between splits.** Against the previous submission the gain
shrank from +0.0155 on BLEND-FIT to +0.0073 on the holdout; against the
organisers' baseline it *grew*, from +0.0152 to +0.0180. A selection overfit in
the new model would shrink both. The asymmetry instead comes from the previous
submission scoring relatively better on the holdout sequences (0.259 vs a 0.248
baseline) than on BLEND-FIT (a tie). Both shifts are within the sequence-level
noise of a 544-sequence split, so neither is evidence of selection bias. The
holdout's job was to confirm the direction of the result on untouched data, and
it did.

The honest headline is therefore **+0.007 over the previous submission and +0.018
over the organisers' baseline**, both on data never used for any decision.

### DeepLOB is dead weight

It receives weight zero on both targets under every fitting method tried, while
costing 358 µs of the 1084 µs per-tick budget. Dropping it is a **33% latency
reduction at no measured cost to score** — pending confirmation on the holdout.

---

## 7. Inference latency

Measured through the same streaming path the scorer uses -- the submission's own
`PredictionModel.predict`, one tick at a time, single intra-/inter-op thread,
`CPUExecutionProvider`, all graph optimisations -- on Apple M3, ONNX Runtime
1.19.2. This is `submission_v2` as shipped: the baseline plus three TCN seeds,
four ONNX sessions over one shared history buffer.

| component | mean | p50 | p95 | p99 |
|---|---|---|---|---|
| baseline (ctx 100) | 316 us | 313 | 328 | 342 |
| tcn_s1 (ctx 128) | 422 us | 416 | 439 | 460 |
| tcn_s2 (ctx 128) | 420 us | 417 | 437 | 450 |
| tcn_s3 (ctx 128) | 428 us | 419 | 446 | 524 |
| **submission_v2, as shipped** | **1562 us** | **1550** | **1616** | **1703** |
| non-prediction tick | 1.1 us | 1.1 | 1.2 | 2.3 |
| window materialisation | 10.1 us | 9.9 | 11.4 | 12.6 |

~640 predicted ticks/sec single-threaded; a full 1000-step sequence costs 1407 ms
end to end. Window construction is **0.65%** of a step, so the ONNX graphs
dominate and replacing the Python history buffer with a ring buffer would recover
at most ~10 us -- the measurement is what says not to bother.

Seed averaging is the whole of the cost increase over the previous submission:
three TCN sessions at ~420 us each. That is a deliberate trade, +0.008 weighted
Pearson for 3x the TCN budget, and the competition metric carries no latency
penalty. Feeding all three seeds from **one shared history buffer** rather than
three duplicate ones saves 28 us per tick (2.2%).

**Measurement hygiene:** this run was taken at load average 4.5 on 8 cores, above
the 0.3x threshold the script warns at. It is trusted because the baseline
component reads 316 us against the 311 us recorded on an idle machine -- within
2% -- and the per-seed TCN figures match the 414 us recorded for the same graph.
Under genuine contention the inflation is far larger: the old three-model
ensemble measured 1085 us idle and 2135 us while an evaluation job ran on the
same host. Cross-checking one component against a known figure is what makes a
loaded measurement usable; `scripts/benchmark.py` prints the load average and
records it in `artifacts/benchmark.json`.

---

## 8. What was tried and rejected

- **Refitting blend weights on weak base models** — worse out-of-fold than the
  strongest single model (§5). Kept in the repo as an ablation.
- **Unconstrained GLS blending on collinear members** — large in-sample gain,
  negative out-of-fold transfer (§5).
- **Uniform-prior shrinkage** — wrong prior when members differ in skill (§5).
- **Longer context for `t1`** (the `t1_specialist`, context 256). `t1` is the weak
  target, so a specialist is tempting, but score does not improve with more
  context — measured by position in sequence, the baseline scores +0.456 at steps
  99–149 and +0.395 at 799–999. Doubling the window is not the indicated fix.
- **`tlob_lite`** (MLP-Mixer, 122 features, per-sequence anchor normalisation).
  Interesting and genuinely leak-free by construction, but never exported to ONNX
  and the most expensive model to validate. Deferred.
- **`train_meta_residual`** (residual stacker MLP). A stacker over collinear base
  models cannot clear the ±0.005 noise floor. Abandoned.
- **int8 dynamic quantisation of the TCN** (`quantize_dynamic`, QUInt8 weights).
  Halves the file (288 KB -> 141 KB) and is **1.9x slower**: 413 us -> 794 us per
  window on ARM64. The model is small enough to stay cache-resident, so the
  quantise/dequantise overhead around each conv dominates any int8 gain, and
  outputs shift by 3e-02 for nothing. Measured and rejected.
- **`non_blocking=True` on MPS transfers** — produced silent NaN losses by reading
  tensors before the async copy landed. Only ever helps pinned CUDA memory.

---

## 9. Reproducing

```bash
python scripts/materialise.py --train-path datasets/train.parquet   # once, ~10s
python -m pytest tests/ -q                                          # 19 tests
python scripts/train_tcn.py --epochs 12 --seed 1 --tag s1           # ~3 min on M3
python scripts/export_tcn.py --checkpoint artifacts/tcn_s1.pt
python scripts/evaluate.py --split blend_fit --save-predictions
python scripts/fit_blend.py --split blend_fit
python scripts/benchmark.py                                         # idle machine
```

The sealed holdout requires `--split holdout --i-am-done`, deliberately awkward
so it is opened once.
