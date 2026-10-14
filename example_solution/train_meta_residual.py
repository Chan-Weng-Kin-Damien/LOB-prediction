import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR.parent))

from utils import DataPoint, weighted_pearson_correlation


RAW_FEATURE_COLUMNS = [f"p{i}" for i in range(12)]
RAW_FEATURE_COLUMNS += [f"v{i}" for i in range(12)]
RAW_FEATURE_COLUMNS += [f"dp{i}" for i in range(4)]
RAW_FEATURE_COLUMNS += [f"dv{i}" for i in range(4)]
TARGET_COLUMNS = ["t0", "t1"]
FEATURE_DIM = 50


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def current_engineered_features(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    bid_p = state[0:6]
    ask_p = state[6:12]
    bid_v = state[12:18]
    ask_v = state[18:24]
    trade_p = state[24:28]
    trade_v = state[28:32]

    bid_p_mean = np.mean(bid_p, dtype=np.float32)
    ask_p_mean = np.mean(ask_p, dtype=np.float32)
    bid_v_sum = np.sum(bid_v, dtype=np.float32)
    ask_v_sum = np.sum(ask_v, dtype=np.float32)
    top_bid_v = bid_v[0]
    top_ask_v = ask_v[0]
    top_bid_p = bid_p[0]
    top_ask_p = ask_p[0]
    trade_p_mean = np.mean(trade_p, dtype=np.float32)
    trade_v_sum = np.sum(trade_v, dtype=np.float32)

    return np.asarray(
        [
            bid_p_mean,
            ask_p_mean,
            ask_p_mean - bid_p_mean,
            top_ask_p - top_bid_p,
            bid_v_sum,
            ask_v_sum,
            (bid_v_sum - ask_v_sum) / (bid_v_sum + ask_v_sum + 1e-3),
            (top_bid_v - top_ask_v) / (top_bid_v + top_ask_v + 1e-3),
            trade_p_mean,
            trade_v_sum,
        ],
        dtype=np.float32,
    )


def build_meta_features(state: np.ndarray, baseline_pred: np.ndarray, deep_pred: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    baseline_pred = np.asarray(baseline_pred, dtype=np.float32)
    deep_pred = np.asarray(deep_pred, dtype=np.float32)
    diff = baseline_pred - deep_pred
    abs_diff = np.abs(diff)
    engineered = current_engineered_features(state)
    return np.concatenate([state, engineered, baseline_pred, deep_pred, diff, abs_diff], axis=0).astype(np.float32, copy=False)


class ResidualMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

    def forward(self, features: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
        correction = 1.5 * torch.tanh(self.net(features) / 1.5)
        return baseline + correction


class ResidualWrapper(nn.Module):
    def __init__(self, model: ResidualMLP):
        super().__init__()
        self.model = model

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        baseline = features[:, 42:44]
        return self.model(features, baseline)


def score_predictions(targets: np.ndarray, predictions: np.ndarray) -> dict:
    scores = {}
    for idx, name in enumerate(TARGET_COLUMNS):
        scores[name] = weighted_pearson_correlation(targets[:, idx], predictions[:, idx])
    scores["weighted_pearson"] = float(np.mean([scores[name] for name in TARGET_COLUMNS]))
    return scores


def load_models():
    baseline_mod = load_module(CURRENT_DIR / "solution.py", "baseline_solution_mod")
    deep_mod = load_module(CURRENT_DIR / "solution_deeplob_attn.py", "deep_solution_mod")
    baseline_model = baseline_mod.PredictionModel()
    deep_model = deep_mod.PredictionModel(
        model_path=str(CURRENT_DIR / "deeplob_attn_tuned.onnx"),
        config_path=str(CURRENT_DIR / "deeplob_attn_tuned.json"),
    )
    return baseline_model, deep_model


def extract_features(dataset_path: Path, output_path: Path, max_sequences: int | None) -> None:
    baseline_model, deep_model = load_models()
    parquet_file = pq.ParquetFile(dataset_path)
    columns = ["seq_ix", "step_in_seq", "need_prediction", *RAW_FEATURE_COLUMNS, *TARGET_COLUMNS]

    features = []
    baselines = []
    targets = []

    seen_sequences = set()
    stop = False

    for batch in parquet_file.iter_batches(batch_size=4096, columns=columns):
        arrays = [np.array(batch.column(i).to_numpy(), copy=True) for i in range(len(columns))]
        for row_idx in range(batch.num_rows):
            seq_ix = int(arrays[0][row_idx])
            if max_sequences is not None and seq_ix not in seen_sequences and len(seen_sequences) >= max_sequences:
                stop = True
                break
            seen_sequences.add(seq_ix)

            state = np.asarray([arrays[col_idx][row_idx] for col_idx in range(3, 35)], dtype=np.float32)
            data_point = DataPoint(
                seq_ix=seq_ix,
                step_in_seq=int(arrays[1][row_idx]),
                need_prediction=bool(arrays[2][row_idx]),
                state=state,
            )
            baseline_pred = baseline_model.predict(data_point)
            deep_pred = deep_model.predict(data_point)

            if data_point.need_prediction:
                feature_vec = build_meta_features(state, baseline_pred, deep_pred)
                features.append(feature_vec)
                baselines.append(np.asarray(baseline_pred, dtype=np.float32))
                targets.append(np.asarray([arrays[35][row_idx], arrays[36][row_idx]], dtype=np.float32))
        if stop:
            break

    features_arr = np.asarray(features, dtype=np.float32)
    baselines_arr = np.asarray(baselines, dtype=np.float32)
    targets_arr = np.asarray(targets, dtype=np.float32)
    np.savez_compressed(output_path, features=features_arr, baselines=baselines_arr, targets=targets_arr)
    print(f"Saved {features_arr.shape[0]} samples to {output_path}")


def weighted_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    weights = torch.abs(y_true) + 1e-3
    column_weights = torch.tensor([0.4, 1.6], dtype=torch.float32, device=y_true.device).view(1, 2)
    loss = weights * column_weights * torch.square(y_pred - y_true)
    return loss.mean()


def train_meta(train_npz: Path, valid_npz: Path, checkpoint_path: Path, onnx_path: Path, config_path: Path) -> None:
    train_data = np.load(train_npz)
    valid_data = np.load(valid_npz)

    x_train = torch.tensor(train_data["features"], dtype=torch.float32)
    y_train = torch.tensor(train_data["targets"], dtype=torch.float32)
    x_valid = torch.tensor(valid_data["features"], dtype=torch.float32)
    y_valid = valid_data["targets"].astype(np.float32)

    model = ResidualMLP(input_dim=x_train.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size=1024, shuffle=True, num_workers=0)

    best_score = -np.inf
    best_state = None

    for epoch in range(1, 7):
        model.train()
        running_loss = 0.0
        batch_count = 0
        for batch_features, batch_targets in loader:
            baseline = batch_features[:, 42:44]
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_features, baseline)
            loss = weighted_loss(batch_targets, predictions)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += float(loss.item())
            batch_count += 1

        model.eval()
        with torch.no_grad():
            pred_valid = model(x_valid, x_valid[:, 42:44]).cpu().numpy()
        scores = score_predictions(y_valid, pred_valid)
        print(
            f"Epoch {epoch:02d} | loss={running_loss / max(batch_count, 1):.6f} | "
            f"valid_wp={scores['weighted_pearson']:.6f} | t0={scores['t0']:.6f} | t1={scores['t1']:.6f}"
        )
        if scores["weighted_pearson"] > best_score:
            best_score = scores["weighted_pearson"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    torch.save({"state_dict": model.state_dict(), "input_dim": x_train.shape[1]}, checkpoint_path)

    wrapper = ResidualWrapper(model)
    dummy = torch.zeros(1, x_train.shape[1], dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    config_path.write_text(json.dumps({"input_dim": int(x_train.shape[1]), "onnx_path": onnx_path.name}, indent=2), encoding="utf-8")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_pred = session.run(["output"], {"input": x_valid[:64].cpu().numpy()})[0]
    with torch.no_grad():
        torch_pred = wrapper(x_valid[:64]).cpu().numpy()
    diff = float(np.max(np.abs(ort_pred - torch_pred)))
    final_scores = score_predictions(y_valid, model(x_valid, x_valid[:, 42:44]).detach().cpu().numpy())
    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Exported ONNX to {onnx_path}")
    print(f"Saved config to {config_path}")
    print(f"ONNX max abs diff: {diff:.8f}")
    print(json.dumps(final_scores, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a residual stacker over baseline + DeepLOB predictions.")
    subparsers = parser.add_subparsers(dest="command")

    extract_parser = subparsers.add_parser("extract", help="Extract meta features from a dataset.")
    extract_parser.add_argument("--dataset-path", required=True)
    extract_parser.add_argument("--output-path", required=True)
    extract_parser.add_argument("--max-sequences", type=int)

    train_parser = subparsers.add_parser("train", help="Train residual meta-model from extracted features.")
    train_parser.add_argument("--train-npz", required=True)
    train_parser.add_argument("--valid-npz", required=True)
    train_parser.add_argument("--checkpoint-path", required=True)
    train_parser.add_argument("--onnx-path", required=True)
    train_parser.add_argument("--config-path", required=True)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "extract":
        extract_features(Path(args.dataset_path), Path(args.output_path), args.max_sequences)
    elif args.command == "train":
        train_meta(
            train_npz=Path(args.train_npz),
            valid_npz=Path(args.valid_npz),
            checkpoint_path=Path(args.checkpoint_path),
            onnx_path=Path(args.onnx_path),
            config_path=Path(args.config_path),
        )
    else:
        parser.print_help()
