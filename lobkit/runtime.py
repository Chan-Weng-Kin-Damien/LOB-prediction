"""Device selection, seeding, and the ONNX inference wrappers.

The training scripts all hardcoded ``torch.device("cpu")``, which is why a GPU
cluster sat unused. ``pick_device`` replaces those ~15 call sites.

The ONNX session settings here mirror the submission exactly (single intra- and
inter-op thread, all graph optimisations, CPU execution provider) so that any
latency figure measured through this module is the latency the scorer would see.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

RAW_DIM = 32


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def pick_device(prefer: str | None = None):
    """cuda -> mps -> cpu, unless ``prefer`` forces one."""
    import torch

    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_device(device) -> str:
    import torch

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        return f"cuda: {props.name}, {props.total_memory / 1e9:.1f} GB, {props.multi_processor_count} SMs"
    return device.type


# --------------------------------------------------------------------------
# ONNX inference
# --------------------------------------------------------------------------


def build_session(model_path: str | Path, threads: int = 1):
    """Session configured identically to the shipped submission."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = threads
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )


class WindowOnnxModel:
    """Streaming wrapper matching ``submission_safe_3model/solution.py`` byte for byte.

    Kept faithful on purpose: this is the reference the vectorised evaluator is
    checked against, so it must reproduce the submission's behaviour including
    the list-of-arrays history and the left zero-pad, not an improved version of
    it.
    """

    def __init__(self, model_path: str | Path, context_len: int):
        self.context_len = int(context_len)
        self.current_seq_ix = None
        self.raw_history: list[np.ndarray] = []
        self.session = build_session(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [item.name for item in self.session.get_outputs()]

    def reset(self) -> None:
        self.current_seq_ix = None
        self.raw_history = []

    def predict(self, seq_ix, state: np.ndarray, need_prediction: bool) -> np.ndarray | None:
        if self.current_seq_ix != seq_ix:
            self.current_seq_ix = seq_ix
            self.raw_history = []

        self.raw_history.append(np.asarray(state, dtype=np.float32).copy())
        if not need_prediction:
            return None

        if len(self.raw_history) >= self.context_len:
            window = np.asarray(self.raw_history[-self.context_len :], dtype=np.float32)
        else:
            pad_len = self.context_len - len(self.raw_history)
            padding = np.zeros((pad_len, RAW_DIM), dtype=np.float32)
            window = np.concatenate(
                [padding, np.asarray(self.raw_history, dtype=np.float32)], axis=0
            )

        output = self.session.run(self.output_names, {self.input_name: window[None, :, :]})[0][0]
        return output.astype(np.float32, copy=False)


def batched_window_predict(
    session,
    sequence: np.ndarray,
    steps: np.ndarray,
    context_len: int,
    batch_size: int = 512,
) -> np.ndarray:
    """Vectorised equivalent of replaying ``WindowOnnxModel`` over one sequence.

    Builds all windows with a strided view and runs them in batches. Identical
    outputs to the streaming path (``tests/test_equivalence.py`` asserts it), but
    hundreds of times faster, which is what makes full-corpus evaluation
    affordable.
    """
    from .windows import make_windows_view

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    view = make_windows_view(sequence[None, :, :], context_len)[0]
    windows = np.ascontiguousarray(view[steps], dtype=np.float32)

    out = np.empty((windows.shape[0], 2), dtype=np.float32)
    for start in range(0, windows.shape[0], batch_size):
        stop = min(start + batch_size, windows.shape[0])
        out[start:stop] = session.run([output_name], {input_name: windows[start:stop]})[0]
    return out
