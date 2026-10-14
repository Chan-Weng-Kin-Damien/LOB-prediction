"""Wunder Challenge 2 submission: baseline GRU + 3-seed full-sequence TCN.

Self-contained by design -- no imports from ``lobkit`` -- so the zip runs under
the scorer with nothing but numpy and onnxruntime.

Blend weights were fitted on 900 validation sequences (810,900 scored ticks) by
maximising the competition metric directly over the probability simplex, with
grouped cross-validation by sequence and shrinkage toward the strongest single
model. Measured against the previous submission on the same sequences:

    previous 3-model blend   0.26901
    this submission          0.28446      +0.01545, 95% CI [+0.00966, +0.02064]

The blend is per target because t0 and t1 are different indicators with very
different achievable skill (~0.41 vs ~0.16), not because they are different
horizons -- no horizon is disclosed anywhere in the data.

Why three TCN seeds: averaging three independently seeded runs is worth +0.008
over the best single seed (0.27600 -> 0.28368), which is the largest reliable
gain available per unit of effort. It costs 3x the TCN inference budget; the
metric carries no latency penalty, so accuracy wins here.

DeepLOB-attention is deliberately absent: every weight-fitting method tried gave
it zero on both targets, while it cost 358 of the previous 1084 us per-tick
budget.
"""

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR.parent))

from utils import DataPoint

RAW_DIM = 32

# Per-target blend weights. Rows sum to 1 within each target.
BASELINE_WEIGHTS = np.array([0.1660, 0.1524], dtype=np.float32)
TCN_TOTAL_WEIGHTS = np.array([0.8340, 0.8476], dtype=np.float32)
TCN_SEEDS = ("tcn_s1.onnx", "tcn_s2.onnx", "tcn_s3.onnx")


def build_session(model_path):
    """Single-threaded CPU session, matching how the scorer runs us."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )


class WindowOnnxModel:
    """Rolling per-sequence window with left zero-padding for short histories.

    The newest tick always sits at index -1, and a shorter history is padded on
    the left rather than the right, so the model sees the same alignment at step
    99 as it does at step 999. Padding is only exercised for the first 28 scored
    ticks of each sequence (history is 100 deep at step 99, context is 128).
    """

    def __init__(self, model_path, context_len):
        self.context_len = int(context_len)
        self.current_seq_ix = None
        self.raw_history = []
        self.session = build_session(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, data_point):
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.raw_history = []

        self.raw_history.append(np.asarray(data_point.state, dtype=np.float32).copy())
        if not data_point.need_prediction:
            return None

        if len(self.raw_history) >= self.context_len:
            window = np.asarray(self.raw_history[-self.context_len :], dtype=np.float32)
        else:
            pad_len = self.context_len - len(self.raw_history)
            padding = np.zeros((pad_len, RAW_DIM), dtype=np.float32)
            window = np.concatenate(
                [padding, np.asarray(self.raw_history, dtype=np.float32)], axis=0
            )

        output = self.session.run([self.output_name], {self.input_name: window[None, :, :]})[0][0]
        return output.astype(np.float32, copy=False)


class SharedHistoryTCNs:
    """Three TCN seeds over one shared history buffer.

    The previous submission gave each model its own duplicate copy of identical
    tick history -- three appends and three buffers for the same data. One buffer
    and one window materialisation feeds all seeds here.
    """

    def __init__(self, model_paths, context_len=128):
        self.context_len = int(context_len)
        self.current_seq_ix = None
        self.raw_history = []
        self.sessions = []
        for path in model_paths:
            session = build_session(path)
            self.sessions.append(
                (session, session.get_inputs()[0].name, session.get_outputs()[0].name)
            )

    def predict(self, data_point):
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.raw_history = []

        self.raw_history.append(np.asarray(data_point.state, dtype=np.float32).copy())
        if not data_point.need_prediction:
            return None

        if len(self.raw_history) >= self.context_len:
            window = np.asarray(self.raw_history[-self.context_len :], dtype=np.float32)
        else:
            pad_len = self.context_len - len(self.raw_history)
            padding = np.zeros((pad_len, RAW_DIM), dtype=np.float32)
            window = np.concatenate(
                [padding, np.asarray(self.raw_history, dtype=np.float32)], axis=0
            )
        batch = window[None, :, :]

        total = None
        for session, input_name, output_name in self.sessions:
            out = session.run([output_name], {input_name: batch})[0][0]
            total = out if total is None else total + out
        return (total / len(self.sessions)).astype(np.float32, copy=False)


class PredictionModel:
    def __init__(self):
        self.baseline_model = WindowOnnxModel(CURRENT_DIR / "baseline.onnx", context_len=100)
        self.tcn_ensemble = SharedHistoryTCNs(
            [CURRENT_DIR / name for name in TCN_SEEDS], context_len=128
        )

    def predict(self, data_point: DataPoint):
        baseline_prediction = self.baseline_model.predict(data_point)
        tcn_prediction = self.tcn_ensemble.predict(data_point)

        if baseline_prediction is None:
            return None

        return BASELINE_WEIGHTS * baseline_prediction + TCN_TOTAL_WEIGHTS * tcn_prediction


if __name__ == "__main__":
    import json

    from utils import ScorerStepByStep

    valid_path = CURRENT_DIR.parent / "datasets" / "valid.parquet"
    scorer = ScorerStepByStep(str(valid_path))
    print(json.dumps(scorer.score(PredictionModel()), indent=2))
