# Submission: Safe 3-Model Ensemble

This is a **safe, production-ready submission** for the Wunder Challenge (LOB Prediction). It implements a **weighted ensemble** of three pre-trained ONNX models, tuned for robustness and generalization.

## 📋 Models Included

1. **Baseline GRU** (`baseline.onnx`): A simple recurrent model providing stable baseline predictions.
2. **DeepLOB Attention** (`deeplob_attn_tuned.onnx`): CNN + GRU + attention architecture, optimized for capturing complex LOB patterns.
3. **Improved TCN** (`improved_tcn.onnx`): Temporal Convolutional Network with causal convolutions for sequence modeling.

## 🧮 Ensemble Strategy

- **Blend Weights** (tuned on validation folds):
  - t0: 89.15% Baseline + 10.85% TCN
  - t1: 65.82% Baseline + 34.18% DeepLOB Attention
- **Context Length**: 128 timesteps for DeepLOB and TCN; 100 for Baseline.
- **Inference**: Models run in parallel; predictions are weighted and summed.

## 🚀 Usage

1. Ensure `utils.py` is in the parent directory (for `DataPoint` class).
2. Run `python solution.py` to test on validation data (requires `../datasets/valid.parquet`).
3. For submission, zip this folder and submit as `submission.zip`.

## 📊 Performance Notes

- Designed for **high reliability** over peak performance.
- Weights optimized to balance bias-variance on cross-validation.
- No external dependencies beyond ONNX Runtime and NumPy.

## 🛠️ Technical Details

- **ONNX Runtime**: CPU-optimized with single-threaded execution.
- **Data Handling**: Maintains per-sequence state; pads short histories with zeros.
- **Output**: Returns `np.ndarray` of shape `(2,)` for t0/t1 predictions, or `None` if no prediction needed.

This ensemble prioritizes **consistency** and **interpretability**, making it a solid choice for competition submissions.
