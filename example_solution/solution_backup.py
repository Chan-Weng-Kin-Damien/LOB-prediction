import os
import sys
import numpy as np
import joblib

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f"{CURRENT_DIR}/..")

from utils import DataPoint, ScorerStepByStep

class PredictionModel:
    def __init__(self, model_path=""):
        self.current_seq_ix = None
        self.last_state = None
        
        # Load LightGBM models
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model0 = joblib.load(os.path.join(base_dir, "model_t0.pkl"))
        self.model1 = joblib.load(os.path.join(base_dir, "model_t1.pkl"))

    def predict(self, data_point: DataPoint) -> np.ndarray:
        # Reset state on new sequence
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.last_state = None

        state = data_point.state.copy()
        
        if not data_point.need_prediction:
            self.last_state = state
            return None

        # Data point layout based on standard dataset order:
        # p0-p11 (0-11), v0-v11 (12-23), dp0-dp3 (24-27), dv0-dv3 (28-31)
        p0, p6 = state[0], state[6]
        v0, v6 = state[12], state[18]

        # Calculate features matching the training script
        p0_diff = state[0] - self.last_state[0] if self.last_state is not None else 0.0
        v0_diff = state[12] - self.last_state[12] if self.last_state is not None else 0.0
        spread = p6 - p0
        obi = (v0 - v6) / (v0 + v6 + 1e-8)

        # Build final feature vector
        feature_vector = np.concatenate([
            state, 
            [p0_diff, v0_diff, spread, obi]
        ]).reshape(1, -1)

        # Predict
        pred_t0 = self.model0.predict(feature_vector)[0]
        pred_t1 = self.model1.predict(feature_vector)[0]

        # Update last state for next step's difference calculations
        self.last_state = state
        
        # Clip predictions to [-6, 6] as per competition rules
        return np.clip(np.array([pred_t0, pred_t1]), -6.0, 6.0)

if __name__ == "__main__":
    test_file = f"{CURRENT_DIR}/../datasets/valid.parquet"
    if os.path.exists(test_file):
        model = PredictionModel()
        scorer = ScorerStepByStep(test_file)
        
        print("Testing LightGBM Model...")
        results = scorer.score(model)
        
        print("\nResults:")
        print(f"Mean Weighted Pearson correlation: {results['weighted_pearson']:.6f}")
        for i, target in enumerate(scorer.targets):
            print(f"  {target}: {results[target]:.6f}")