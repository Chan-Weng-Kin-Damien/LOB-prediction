import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR.parent))

from utils import DataPoint, ScorerStepByStep


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PredictionModel:
    """
    Ensemble submission model.

    Components:
    - Baseline GRU ONNX model.
    - DeepLOB-inspired CNN + GRU + attention ONNX model.
    - Causal TCN ONNX model.
    - Stateful GRU with augmentation ONNX model.

    Blend strategy tuned on local validation:
    - t0: 0.7575 * baseline + 0.1085 * TCN + 0.1340 * stateful-GRU
    - t1: 0.6582 * baseline + 0.3418 * DeepLOB-attention
    """

    def __init__(self):
        baseline_module = _load_module(CURRENT_DIR / "solution_baseline_onnx.py", "baseline_solution_module")
        deep_module = _load_module(CURRENT_DIR / "solution_deeplob_attn.py", "deeplob_solution_module")
        tcn_module = _load_module(CURRENT_DIR / "solution_example_improved.py", "tcn_solution_module")
        gru_module = _load_module(CURRENT_DIR / "solution_stateful_rnn_aug.py", "stateful_gru_solution_module")

        self.baseline_model = baseline_module.PredictionModel()
        self.deep_model = deep_module.PredictionModel(
            model_path=str(CURRENT_DIR / "deeplob_attn_tuned.onnx"),
            config_path=str(CURRENT_DIR / "deeplob_attn_tuned.json"),
        )
        self.tcn_model = tcn_module.PredictionModel(
            model_path=str(CURRENT_DIR / "improved_tcn.onnx"),
            config_path=str(CURRENT_DIR / "improved_tcn.json"),
        )
        self.gru_model = gru_module.PredictionModel(
            model_path=str(CURRENT_DIR / "gru_best.onnx"),
            config_path=str(CURRENT_DIR / "gru_best.json"),
        )

        self.baseline_weights = np.array([0.7575, 0.6582], dtype=np.float32)
        self.deep_weights = np.array([0.0, 0.3418], dtype=np.float32)
        self.tcn_weights = np.array([0.1085, 0.0], dtype=np.float32)
        self.gru_weights = np.array([0.1340, 0.0], dtype=np.float32)

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        baseline_prediction = self.baseline_model.predict(data_point)
        deep_prediction = self.deep_model.predict(data_point)
        tcn_prediction = self.tcn_model.predict(data_point)
        gru_prediction = self.gru_model.predict(data_point)

        if baseline_prediction is None:
            return None

        return (
            self.baseline_weights * baseline_prediction
            + self.deep_weights * deep_prediction
            + self.tcn_weights * tcn_prediction
            + self.gru_weights * gru_prediction
        )


if __name__ == "__main__":
    valid_path = CURRENT_DIR.parent / "datasets" / "valid.parquet"
    model = PredictionModel()
    scorer = ScorerStepByStep(str(valid_path))
    results = scorer.score(model)
    print(json.dumps(results, indent=2))
