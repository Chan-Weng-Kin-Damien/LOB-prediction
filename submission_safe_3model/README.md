# Competition submission: three-model blend

The entry scored during the Wunder Fund RNN Challenge (417th of 5,062). Kept
here unchanged for reference; `../submission_v2/` is the later revision.

## Contents

| file | role |
|---|---|
| `baseline.onnx` | GRU baseline supplied in the organisers' starter pack, context 100 |
| `deeplob_attn_tuned.onnx` | CNN with inception-residual blocks, GRU and additive attention, context 128 |
| `improved_tcn.onnx` | Causal TCN, dilations 1-32, context 128 |
| `solution.py` | Streaming wrapper and blend |

## Blend

Weights are per target and sum to one within each:

| target | baseline | DeepLOB attention | TCN |
|---|---|---|---|
| t0 | 0.8915 | 0 | 0.1085 |
| t1 | 0.6582 | 0.3418 | 0 |

Context length is 128 for the DeepLOB and TCN models and 100 for the baseline.

## Measured performance

On the 544-sequence sealed holdout described in `../README.md`, this blend scores
0.2590 mean weighted Pearson correlation, against 0.2482 for the supplied
baseline alone and 0.2663 for the revised submission.

The three models run sequentially in a single thread, each maintaining its own
history buffer. Per predicted tick the blend costs about 1,085 us on an idle
Apple M3. `../submission_v2/` shares one buffer across models instead.

## Notes

- Each model keeps per-sequence state and resets when `seq_ix` changes.
- Histories shorter than the context length are zero-padded on the left, so the
  newest tick is always the last row. This affects only the first 28 scored ticks
  of a sequence for the context-128 models.
- The single-threaded ONNX Runtime session settings and the rolling-window
  pattern come from the organisers' starter pack.
- `utils.py` from the starter pack must be importable from the parent directory;
  it is not redistributed here.
