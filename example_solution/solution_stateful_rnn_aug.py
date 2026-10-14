import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F


CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR.parent))

from utils import DataPoint, weighted_pearson_correlation


SEQUENCE_LENGTH = 1000
WARMUP_STEPS = 99
RAW_FEATURE_COLUMNS = [f"p{i}" for i in range(12)]
RAW_FEATURE_COLUMNS += [f"v{i}" for i in range(12)]
RAW_FEATURE_COLUMNS += [f"dp{i}" for i in range(4)]
RAW_FEATURE_COLUMNS += [f"dv{i}" for i in range(4)]
TARGET_COLUMNS = ["t0", "t1"]
RAW_DIM = len(RAW_FEATURE_COLUMNS)
MODEL_INPUT_DIM = RAW_DIM * 2

DEFAULT_HIDDEN_DIM = 128
DEFAULT_ONNX_PATH = CURRENT_DIR / "stateful_gru_aug.onnx"
DEFAULT_CHECKPOINT_PATH = CURRENT_DIR / "stateful_gru_aug.pt"
DEFAULT_CONFIG_PATH = CURRENT_DIR / "stateful_gru_aug.json"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def compute_sequence_variance_median(parquet_path: str | Path, batch_rows: int = 65536) -> float:
    variances = []
    for raw, _, _ in iter_sequence_chunks(parquet_path, include_targets=False, batch_rows=batch_rows):
        seq_var = raw.astype(np.float32, copy=False).var(axis=(1, 2), dtype=np.float32)
        variances.append(seq_var)
    return float(np.median(np.concatenate(variances, axis=0)))


def transform_inputs_torch(raw: torch.Tensor) -> torch.Tensor:
    return torch.cat([torch.sigmoid(raw), F.silu(raw)], dim=2)


def augment_sequences(raw: torch.Tensor, global_median_variance: float) -> torch.Tensor:
    seq_var = torch.var(raw, dim=(1, 2), unbiased=False).clamp_min(1e-6)
    norm_scale = torch.sqrt(torch.tensor(global_median_variance, dtype=raw.dtype, device=raw.device) / seq_var)
    random_scale = 0.6 + 0.8 * torch.rand(raw.shape[0], dtype=raw.dtype, device=raw.device)
    scaled = raw * (norm_scale * random_scale).view(-1, 1, 1)
    noise = 0.05 * torch.randn_like(scaled)
    return scaled + noise


def concat_if_needed(left: np.ndarray | None, right: np.ndarray) -> np.ndarray:
    if left is None or left.size == 0:
        return right
    return np.concatenate([left, right], axis=0)


def iter_sequence_chunks(
    parquet_path: str | Path,
    include_targets: bool,
    batch_rows: int = 65536,
):
    parquet_path = Path(parquet_path)
    columns = ["step_in_seq", "need_prediction", *RAW_FEATURE_COLUMNS]
    if include_targets:
        columns.extend(TARGET_COLUMNS)

    parquet_file = pq.ParquetFile(parquet_path)
    carry: dict[str, np.ndarray | None] = {column: None for column in columns}
    expected_steps = np.arange(SEQUENCE_LENGTH, dtype=np.int16)[None, :]

    for batch in parquet_file.iter_batches(batch_size=batch_rows, columns=columns):
        arrays: dict[str, np.ndarray] = {}
        for idx, column in enumerate(columns):
            dtype = np.float32
            if column == "step_in_seq":
                dtype = np.int16
            elif column == "need_prediction":
                dtype = np.int8
            arrays[column] = np.array(batch.column(idx).to_numpy(), copy=True, dtype=dtype)

        for column in columns:
            arrays[column] = concat_if_needed(carry[column], arrays[column])

        usable_rows = (arrays["step_in_seq"].shape[0] // SEQUENCE_LENGTH) * SEQUENCE_LENGTH
        for column in columns:
            carry[column] = arrays[column][usable_rows:]
            arrays[column] = arrays[column][:usable_rows]

        if usable_rows == 0:
            continue

        seq_count = usable_rows // SEQUENCE_LENGTH
        steps = arrays["step_in_seq"].reshape(seq_count, SEQUENCE_LENGTH)
        if not np.all(steps == expected_steps):
            raise ValueError(f"{parquet_path} contains malformed sequences")

        raw = np.stack(
            [arrays[column].reshape(seq_count, SEQUENCE_LENGTH) for column in RAW_FEATURE_COLUMNS],
            axis=2,
        ).astype(np.float32, copy=False)
        mask = arrays["need_prediction"].reshape(seq_count, SEQUENCE_LENGTH).astype(bool, copy=False)

        if include_targets:
            targets = np.stack(
                [arrays[column].reshape(seq_count, SEQUENCE_LENGTH) for column in TARGET_COLUMNS],
                axis=2,
            ).astype(np.float32, copy=False)
        else:
            targets = None

        yield raw, targets, mask

    leftover = carry["step_in_seq"]
    if leftover is not None and leftover.size != 0:
        raise ValueError(f"{parquet_path} ended with a partial sequence")


def masked_weighted_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    mask: torch.Tensor,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    active_true = y_true[mask]
    active_pred = y_pred[mask]
    weights = torch.pow(torch.abs(active_true) + 1e-3, alpha)
    mse = (weights * torch.square(active_pred - active_true)).sum() / weights.sum().clamp_min(1e-8)
    corr = weighted_corr_loss(active_true, active_pred, alpha=alpha)
    return beta * mse + (1.0 - beta) * corr


def weighted_corr_loss(y_true: torch.Tensor, y_pred: torch.Tensor, alpha: float) -> torch.Tensor:
    losses = []
    for target_idx in range(y_true.shape[1]):
        true_col = y_true[:, target_idx]
        pred_col = y_pred[:, target_idx]
        weights = torch.pow(torch.abs(true_col) + 1e-3, alpha)
        sum_w = weights.sum().clamp_min(1e-8)
        mean_true = (weights * true_col).sum() / sum_w
        mean_pred = (weights * pred_col).sum() / sum_w
        dev_true = true_col - mean_true
        dev_pred = pred_col - mean_pred
        cov = (weights * dev_true * dev_pred).sum() / sum_w
        var_true = (weights * torch.square(dev_true)).sum() / sum_w
        var_pred = (weights * torch.square(dev_pred)).sum() / sum_w
        corr = cov / torch.sqrt(var_true * var_pred + 1e-8)
        losses.append(1.0 - corr)
    return torch.stack(losses).mean()


def mixed_loss(y_true: torch.Tensor, y_pred: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    weights = torch.pow(torch.abs(y_true) + 1e-3, alpha)
    mse = (weights * torch.square(y_pred - y_true)).sum() / weights.sum().clamp_min(1e-8)
    corr = weighted_corr_loss(y_true, y_pred, alpha=alpha)
    return beta * mse + (1.0 - beta) * corr


class HighwayHead(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.transform = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.residual = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, output_dim)
        nn.init.constant_(self.gate.bias, -1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x)
        transformed = F.gelu(self.transform(x_norm))
        gate = torch.sigmoid(self.gate(x_norm))
        residual = self.residual(x_norm)
        mixed = gate * transformed + (1.0 - gate) * residual
        return self.output(self.dropout(mixed))


class StatefulGRUModel(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.input_norm = nn.LayerNorm(MODEL_INPUT_DIM)
        self.gru = nn.GRU(
            input_size=MODEL_INPUT_DIM,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.head = HighwayHead(hidden_dim=hidden_dim, output_dim=len(TARGET_COLUMNS), dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self._init_chrono_bias()

    def _init_chrono_bias(self) -> None:
        with torch.no_grad():
            chrono = torch.log(torch.rand(self.hidden_dim) * (SEQUENCE_LENGTH - 1) + 1.0)
            self.gru.bias_ih_l0.zero_()
            self.gru.bias_hh_l0.zero_()
            self.gru.bias_ih_l0[self.hidden_dim : 2 * self.hidden_dim] = chrono

    def forward(self, raw: torch.Tensor, hidden: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = transform_inputs_torch(raw)
        x = self.input_norm(x)
        out, hidden = self.gru(x, hidden)
        out = self.dropout(out)
        pred = self.head(out)
        return 6.0 * torch.tanh(pred / 6.0), hidden


def evaluate_exact(
    model: StatefulGRUModel,
    parquet_path: str | Path,
    batch_size: int,
    batch_rows: int,
    device: torch.device,
) -> dict:
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for raw_chunk, target_chunk, mask_chunk in iter_sequence_chunks(parquet_path, include_targets=True, batch_rows=batch_rows):
            for start in range(0, raw_chunk.shape[0], batch_size):
                stop = min(start + batch_size, raw_chunk.shape[0])
                batch_raw = torch.tensor(raw_chunk[start:stop], dtype=torch.float32, device=device)
                batch_pred, _ = model(batch_raw)
                batch_pred_np = batch_pred.cpu().numpy()
                batch_mask = mask_chunk[start:stop]
                batch_target = target_chunk[start:stop]
                predictions.append(batch_pred_np[batch_mask])
                targets.append(batch_target[batch_mask])

    pred = np.concatenate(predictions, axis=0)
    true = np.concatenate(targets, axis=0)
    scores = {}
    for idx, name in enumerate(TARGET_COLUMNS):
        scores[name] = weighted_pearson_correlation(true[:, idx], pred[:, idx])
    scores["weighted_pearson"] = float(np.mean([scores[name] for name in TARGET_COLUMNS]))
    return scores


def save_checkpoint(
    checkpoint_path: Path,
    model: StatefulGRUModel,
    hidden_dim: int,
    dropout: float,
    global_median_variance: float,
) -> None:
    checkpoint = {
        "state_dict": model.state_dict(),
        "hidden_dim": int(hidden_dim),
        "dropout": float(dropout),
        "global_median_variance": float(global_median_variance),
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[StatefulGRUModel, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = StatefulGRUModel(
        hidden_dim=int(checkpoint["hidden_dim"]),
        dropout=float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    return model, checkpoint


def export_onnx(model: StatefulGRUModel, onnx_path: Path) -> None:
    model.eval()
    dummy_input = torch.zeros(1, 1, RAW_DIM, dtype=torch.float32)
    dummy_hidden = torch.zeros(1, 1, model.hidden_dim, dtype=torch.float32)
    torch.onnx.export(
        model.cpu(),
        (dummy_input, dummy_hidden),
        onnx_path,
        input_names=["input", "hidden_in"],
        output_names=["output", "hidden_out"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "hidden_in": {1: "batch_size"},
            "output": {0: "batch_size"},
            "hidden_out": {1: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )


def verify_onnx_export(model: StatefulGRUModel, onnx_path: Path) -> float:
    model.eval()
    sample = torch.randn(4, 1, RAW_DIM, dtype=torch.float32)
    hidden = torch.randn(1, 4, model.hidden_dim, dtype=torch.float32)
    with torch.no_grad():
        torch_output, torch_hidden = model(sample, hidden)
        torch_output = torch_output.cpu().numpy()
        torch_hidden = torch_hidden.cpu().numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_output, onnx_hidden = session.run(
        ["output", "hidden_out"],
        {"input": sample.cpu().numpy(), "hidden_in": hidden.cpu().numpy()},
    )
    return float(max(np.max(np.abs(torch_output - onnx_output)), np.max(np.abs(torch_hidden - onnx_hidden))))


def write_config(config_path: Path, onnx_path: Path, hidden_dim: int) -> None:
    config = {
        "onnx_path": onnx_path.name,
        "hidden_dim": int(hidden_dim),
        "raw_dim": RAW_DIM,
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def train_model(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    torch.set_num_threads(max(1, min(args.torch_threads, os.cpu_count() or 1)))
    device = torch.device("cpu")

    print(f"Computing sequence variance median from {args.train_path}")
    global_median_variance = compute_sequence_variance_median(args.train_path, batch_rows=args.batch_rows)
    print(f"Global sequence variance median: {global_median_variance:.6f}")

    model = StatefulGRUModel(hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_score = -np.inf
    checkpoint_path = Path(args.checkpoint_path)
    onnx_path = Path(args.onnx_path)
    config_path = Path(args.config_path)

    for epoch in range(1, args.epochs + 1):
        epoch_rng = np.random.default_rng(args.seed + epoch)
        model.train()
        running_loss = 0.0
        seen_batches = 0

        for raw_chunk, target_chunk, mask_chunk in iter_sequence_chunks(args.train_path, include_targets=True, batch_rows=args.batch_rows):
            order = epoch_rng.permutation(raw_chunk.shape[0])
            for start in range(0, raw_chunk.shape[0], args.batch_size):
                stop = min(start + args.batch_size, raw_chunk.shape[0])
                batch_idx = order[start:stop]
                batch_raw = torch.tensor(raw_chunk[batch_idx], dtype=torch.float32, device=device)
                batch_targets = torch.tensor(target_chunk[batch_idx], dtype=torch.float32, device=device)
                batch_mask = torch.tensor(mask_chunk[batch_idx], dtype=torch.bool, device=device)
                batch_raw = augment_sequences(batch_raw, global_median_variance=global_median_variance)

                optimizer.zero_grad(set_to_none=True)
                predictions, _ = model(batch_raw)
                loss = masked_weighted_metrics(batch_targets, predictions, batch_mask, alpha=args.alpha, beta=args.beta)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                running_loss += float(loss.item())
                seen_batches += 1

        avg_loss = running_loss / max(seen_batches, 1)
        print(f"Epoch {epoch:02d} | train_loss={avg_loss:.6f}")

        scores = evaluate_exact(
            model=model,
            parquet_path=args.valid_path,
            batch_size=args.eval_batch_size,
            batch_rows=args.batch_rows,
            device=device,
        )
        print(
            f"Epoch {epoch:02d} | valid_wp={scores['weighted_pearson']:.6f} | "
            f"t0={scores['t0']:.6f} | t1={scores['t1']:.6f}"
        )

        if scores["weighted_pearson"] > best_score:
            best_score = scores["weighted_pearson"]
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                global_median_variance=global_median_variance,
            )

    best_model, checkpoint = load_checkpoint(checkpoint_path, device=device)
    export_onnx(best_model, onnx_path=onnx_path)
    write_config(config_path=config_path, onnx_path=onnx_path, hidden_dim=int(checkpoint["hidden_dim"]))
    max_abs_diff = verify_onnx_export(best_model, onnx_path=onnx_path)
    final_scores = evaluate_exact(
        model=best_model,
        parquet_path=args.valid_path,
        batch_size=args.eval_batch_size,
        batch_rows=args.batch_rows,
        device=device,
    )

    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Exported ONNX to {onnx_path}")
    print(f"Saved config to {config_path}")
    print(f"ONNX max abs diff: {max_abs_diff:.8f}")
    print(json.dumps(final_scores, indent=2))


def validate_checkpoint(args: argparse.Namespace) -> None:
    device = torch.device("cpu")
    model, checkpoint = load_checkpoint(Path(args.checkpoint_path), device=device)
    scores = evaluate_exact(
        model=model,
        parquet_path=args.valid_path,
        batch_size=args.eval_batch_size,
        batch_rows=args.batch_rows,
        device=device,
    )
    print(json.dumps(scores, indent=2))


def export_checkpoint(args: argparse.Namespace) -> None:
    device = torch.device("cpu")
    model, checkpoint = load_checkpoint(Path(args.checkpoint_path), device=device)
    onnx_path = Path(args.onnx_path)
    config_path = Path(args.config_path)

    export_onnx(model, onnx_path=onnx_path)
    write_config(config_path=config_path, onnx_path=onnx_path, hidden_dim=int(checkpoint["hidden_dim"]))
    max_abs_diff = verify_onnx_export(model, onnx_path=onnx_path)

    print(f"Exported ONNX to {onnx_path}")
    print(f"Saved config to {config_path}")
    print(f"ONNX max abs diff: {max_abs_diff:.8f}")


class PredictionModel:
    def __init__(self, model_path: str = "", config_path: str = ""):
        self.current_seq_ix = None
        self.hidden_state: np.ndarray | None = None

        self.model_path = Path(model_path) if model_path else DEFAULT_ONNX_PATH
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self.hidden_dim = DEFAULT_HIDDEN_DIM

        if self.config_path.exists():
            config = json.loads(self.config_path.read_text(encoding="utf-8"))
            self.hidden_dim = int(config["hidden_dim"])
            if not model_path:
                self.model_path = self.config_path.with_name(config.get("onnx_path", self.model_path.name))

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
        self.hidden_input_name = self.ort_session.get_inputs()[1].name
        self.output_name = self.ort_session.get_outputs()[0].name
        self.hidden_output_name = self.ort_session.get_outputs()[1].name

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.hidden_state = np.zeros((1, 1, self.hidden_dim), dtype=np.float32)

        if not data_point.need_prediction:
            _, self.hidden_state = self.ort_session.run(
                [self.output_name, self.hidden_output_name],
                {
                    self.input_name: np.asarray(data_point.state, dtype=np.float32)[None, None, :],
                    self.hidden_input_name: self.hidden_state,
                },
            )
            return None

        output, self.hidden_state = self.ort_session.run(
            [self.output_name, self.hidden_output_name],
            {
                self.input_name: np.asarray(data_point.state, dtype=np.float32)[None, None, :],
                self.hidden_input_name: self.hidden_state,
            },
        )
        return output[0, 0].astype(np.float32, copy=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stateful GRU with augmentation, validation, and ONNX export.")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train on train.parquet, validate on valid.parquet, export ONNX.")
    train_parser.add_argument("--train-path", default=str(CURRENT_DIR.parent / "datasets" / "train.parquet"))
    train_parser.add_argument("--valid-path", default=str(CURRENT_DIR.parent / "datasets" / "valid.parquet"))
    train_parser.add_argument("--checkpoint-path", default=str(DEFAULT_CHECKPOINT_PATH))
    train_parser.add_argument("--onnx-path", default=str(DEFAULT_ONNX_PATH))
    train_parser.add_argument("--config-path", default=str(DEFAULT_CONFIG_PATH))
    train_parser.add_argument("--epochs", type=int, default=6)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--eval-batch-size", type=int, default=64)
    train_parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    train_parser.add_argument("--dropout", type=float, default=0.10)
    train_parser.add_argument("--lr", type=float, default=3e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-2)
    train_parser.add_argument("--alpha", type=float, default=0.75)
    train_parser.add_argument("--beta", type=float, default=0.65)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--batch-rows", type=int, default=65536)
    train_parser.add_argument("--torch-threads", type=int, default=8)

    validate_parser = subparsers.add_parser("validate", help="Validate a saved checkpoint on valid.parquet.")
    validate_parser.add_argument("--valid-path", default=str(CURRENT_DIR.parent / "datasets" / "valid.parquet"))
    validate_parser.add_argument("--checkpoint-path", default=str(DEFAULT_CHECKPOINT_PATH))
    validate_parser.add_argument("--eval-batch-size", type=int, default=512)
    validate_parser.add_argument("--batch-rows", type=int, default=65536)

    export_parser = subparsers.add_parser("export", help="Export a saved checkpoint to ONNX.")
    export_parser.add_argument("--checkpoint-path", default=str(DEFAULT_CHECKPOINT_PATH))
    export_parser.add_argument("--onnx-path", default=str(DEFAULT_ONNX_PATH))
    export_parser.add_argument("--config-path", default=str(DEFAULT_CONFIG_PATH))

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        train_model(args)
    elif args.command == "validate":
        validate_checkpoint(args)
    elif args.command == "export":
        export_checkpoint(args)
    else:
        if DEFAULT_ONNX_PATH.exists():
            model = PredictionModel()
            print(f"Loaded {model.model_path}")
        else:
            parser.print_help()
