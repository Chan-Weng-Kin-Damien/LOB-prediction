import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR.parent))

from utils import DataPoint

RAW_DIM = 32


def build_session(model_path: Path) -> tuple[ort.InferenceSession, str, list[str]]:
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(model_path), sess_options=session_options, providers=["CPUExecutionProvider"])
    input_names = [item.name for item in session.get_inputs()]
    output_names = [item.name for item in session.get_outputs()]
    return session, input_names[0], output_names


class WindowOnnxModel:
    def __init__(self, model_path: Path, context_len: int):
        self.context_len = int(context_len)
        self.current_seq_ix = None
        self.raw_history: list[np.ndarray] = []
        self.session, self.input_name, self.output_names = build_session(model_path)

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
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
            window = np.concatenate([padding, np.asarray(self.raw_history, dtype=np.float32)], axis=0)

        output = self.session.run(self.output_names, {self.input_name: window[None, :, :]})[0][0]
        return output.astype(np.float32, copy=False)


class PredictionModel:
    def __init__(self):
        deep_config = json.loads((CURRENT_DIR / "deeplob_attn_tuned.json").read_text(encoding="utf-8"))
        tcn_config = json.loads((CURRENT_DIR / "improved_tcn.json").read_text(encoding="utf-8"))

        self.baseline_model = WindowOnnxModel(CURRENT_DIR / "baseline.onnx", context_len=100)
        self.deep_model = WindowOnnxModel(CURRENT_DIR / deep_config["onnx_path"], context_len=int(deep_config["context_len"]))
        self.tcn_model = WindowOnnxModel(CURRENT_DIR / tcn_config["onnx_path"], context_len=int(tcn_config["context_len"]))

        self.baseline_weights = np.array([0.8915, 0.6582], dtype=np.float32)
        self.deep_weights = np.array([0.0, 0.3418], dtype=np.float32)
        self.tcn_weights = np.array([0.1085, 0.0], dtype=np.float32)

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        baseline_prediction = self.baseline_model.predict(data_point)
        deep_prediction = self.deep_model.predict(data_point)
        tcn_prediction = self.tcn_model.predict(data_point)

        if baseline_prediction is None:
            return None

        return (
            self.baseline_weights * baseline_prediction
            + self.deep_weights * deep_prediction
            + self.tcn_weights * tcn_prediction
        )
