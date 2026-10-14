import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort


CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR.parent))

from utils import DataPoint, ScorerStepByStep


DEFAULT_MODEL_PATH = CURRENT_DIR / "baseline.onnx"
DEFAULT_CONTEXT_LEN = 100
RAW_DIM = 32


class PredictionModel:
    def __init__(self, model_path: str = ""):
        self.current_seq_ix = None
        self.raw_history: list[np.ndarray] = []
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.ort_session = ort.InferenceSession(
            str(self.model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.raw_history = []

        self.raw_history.append(np.asarray(data_point.state, dtype=np.float32).copy())

        if not data_point.need_prediction:
            return None

        if len(self.raw_history) >= DEFAULT_CONTEXT_LEN:
            window = np.asarray(self.raw_history[-DEFAULT_CONTEXT_LEN:], dtype=np.float32)
        else:
            pad_len = DEFAULT_CONTEXT_LEN - len(self.raw_history)
            padding = np.zeros((pad_len, RAW_DIM), dtype=np.float32)
            window = np.concatenate([padding, np.asarray(self.raw_history, dtype=np.float32)], axis=0)

        output = self.ort_session.run([self.output_name], {self.input_name: window[None, :, :]})[0][0]
        return output.astype(np.float32, copy=False)


if __name__ == "__main__":
    valid_path = CURRENT_DIR.parent / "datasets" / "valid.parquet"
    model = PredictionModel()
    scorer = ScorerStepByStep(str(valid_path))
    results = scorer.score(model)
    print(json.dumps(results, indent=2))
