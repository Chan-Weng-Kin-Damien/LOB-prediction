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
RAW_FEATURE_COLUMNS = [f"p{i}" for i in range(12)]
RAW_FEATURE_COLUMNS += [f"v{i}" for i in range(12)]
RAW_FEATURE_COLUMNS += [f"dp{i}" for i in range(4)]
RAW_FEATURE_COLUMNS += [f"dv{i}" for i in range(4)]
TARGET_COLUMNS = ["t0", "t1"]
ENGINEERED_DIM = 10
RAW_DIM = len(RAW_FEATURE_COLUMNS)
FEATURE_DIM = RAW_DIM + RAW_DIM + ENGINEERED_DIM

DEFAULT_CONTEXT_LEN = 128
DEFAULT_DILATIONS = [1, 2, 4, 8, 16, 32]
DEFAULT_ONNX_PATH = CURRENT_DIR / "deeplob_attn.onnx"
DEFAULT_CHECKPOINT_PATH = CURRENT_DIR / "deeplob_attn.pt"
DEFAULT_CONFIG_PATH = CURRENT_DIR / "deeplob_attn.json"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_dilations(spec: str) -> list[int]:
    return [int(part.strip()) for part in spec.split(",") if part.strip()]


def receptive_field(dilations: list[int], kernel_size: int = 3) -> int:
    return 1 + (kernel_size - 1) * sum(dilations)


def engineered_features_numpy(raw_batch: np.ndarray) -> np.ndarray:
    raw_batch = np.asarray(raw_batch, dtype=np.float32)
    delta = np.zeros_like(raw_batch)
    delta[:, 1:, :] = raw_batch[:, 1:, :] - raw_batch[:, :-1, :]

    bid_p = raw_batch[:, :, 0:6]
    ask_p = raw_batch[:, :, 6:12]
    bid_v = raw_batch[:, :, 12:18]
    ask_v = raw_batch[:, :, 18:24]
    trade_p = raw_batch[:, :, 24:28]
    trade_v = raw_batch[:, :, 28:32]

    bid_p_mean = bid_p.mean(axis=2, keepdims=True, dtype=np.float32)
    ask_p_mean = ask_p.mean(axis=2, keepdims=True, dtype=np.float32)
    bid_v_sum = bid_v.sum(axis=2, keepdims=True, dtype=np.float32)
    ask_v_sum = ask_v.sum(axis=2, keepdims=True, dtype=np.float32)
    top_bid_v = bid_v[:, :, :1]
    top_ask_v = ask_v[:, :, :1]
    top_bid_p = bid_p[:, :, :1]
    top_ask_p = ask_p[:, :, :1]
    trade_p_mean = trade_p.mean(axis=2, keepdims=True, dtype=np.float32)
    trade_v_sum = trade_v.sum(axis=2, keepdims=True, dtype=np.float32)

    volume_imbalance = (bid_v_sum - ask_v_sum) / (bid_v_sum + ask_v_sum + 1e-3)
    top_volume_imbalance = (top_bid_v - top_ask_v) / (top_bid_v + top_ask_v + 1e-3)
    spread = ask_p_mean - bid_p_mean
    top_spread = top_ask_p - top_bid_p

    engineered = np.concatenate(
        [
            bid_p_mean,
            ask_p_mean,
            spread,
            top_spread,
            bid_v_sum,
            ask_v_sum,
            volume_imbalance,
            top_volume_imbalance,
            trade_p_mean,
            trade_v_sum,
        ],
        axis=2,
    )
    return np.concatenate([raw_batch, delta, engineered], axis=2).astype(np.float32, copy=False)


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


def compute_feature_stats(parquet_path: str | Path, batch_rows: int = 65536) -> tuple[np.ndarray, np.ndarray]:
    total_sum = np.zeros(FEATURE_DIM, dtype=np.float64)
    total_sq_sum = np.zeros(FEATURE_DIM, dtype=np.float64)
    total_count = 0

    for raw, _, _ in iter_sequence_chunks(parquet_path, include_targets=False, batch_rows=batch_rows):
        features = engineered_features_numpy(raw)
        flat = features.reshape(-1, FEATURE_DIM).astype(np.float64, copy=False)
        total_sum += flat.sum(axis=0)
        total_sq_sum += np.square(flat).sum(axis=0)
        total_count += flat.shape[0]

    mean = (total_sum / max(total_count, 1)).astype(np.float32)
    variance = np.maximum(total_sq_sum / max(total_count, 1) - np.square(mean.astype(np.float64)), 1e-6)
    std = np.sqrt(variance).astype(np.float32)
    return mean, std


def make_window(raw_sequence: np.ndarray, end_idx: int, context_len: int) -> np.ndarray:
    window = np.zeros((context_len, RAW_DIM), dtype=np.float32)
    start = max(0, end_idx - context_len + 1)
    tail = raw_sequence[start : end_idx + 1]
    window[-tail.shape[0] :] = tail
    return window


def make_windows_view(raw_chunk: np.ndarray, context_len: int) -> np.ndarray:
    pad = np.zeros((raw_chunk.shape[0], context_len - 1, RAW_DIM), dtype=np.float32)
    padded = np.concatenate([pad, raw_chunk], axis=1)
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=context_len, axis=1)
    return np.moveaxis(windows, -1, 2)


def sample_training_windows(
    raw_chunk: np.ndarray,
    target_chunk: np.ndarray,
    mask_chunk: np.ndarray,
    context_len: int,
    stride: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    windows_view = make_windows_view(raw_chunk, context_len)
    seq_indices = []
    step_indices = []

    for seq_idx in range(raw_chunk.shape[0]):
        valid_steps = np.flatnonzero(mask_chunk[seq_idx])
        if valid_steps.size == 0:
            continue

        offset = int(rng.integers(0, max(stride, 1)))
        chosen = valid_steps[offset::stride]
        if chosen.size == 0 or chosen[-1] != valid_steps[-1]:
            chosen = np.concatenate([chosen, valid_steps[-1:]])

        seq_indices.append(np.full(chosen.shape[0], seq_idx, dtype=np.int32))
        step_indices.append(chosen.astype(np.int32, copy=False))

    if not seq_indices:
        return (
            np.zeros((0, context_len, RAW_DIM), dtype=np.float32),
            np.zeros((0, len(TARGET_COLUMNS)), dtype=np.float32),
        )

    seq_idx_arr = np.concatenate(seq_indices, axis=0)
    step_idx_arr = np.concatenate(step_indices, axis=0)
    windows = np.ascontiguousarray(windows_view[seq_idx_arr, step_idx_arr], dtype=np.float32)
    targets = np.ascontiguousarray(target_chunk[seq_idx_arr, step_idx_arr], dtype=np.float32)
    return windows, targets


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


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        self.left_padding = kernel_size - 1
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.left_padding, 0)))


class InceptionResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float):
        super().__init__()
        branch_channels = out_channels // 4
        self.branch1 = nn.Conv1d(in_channels, branch_channels, kernel_size=1)
        self.branch3 = CausalConv1d(in_channels, branch_channels, kernel_size=3)
        self.branch5 = CausalConv1d(in_channels, branch_channels, kernel_size=5)
        self.branch7 = CausalConv1d(in_channels, branch_channels, kernel_size=7)
        self.mix = nn.Conv1d(branch_channels * 4, out_channels, kernel_size=1)
        self.norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)
        branches = [
            self.branch1(x),
            self.branch3(x),
            self.branch5(x),
            self.branch7(x),
        ]
        x = torch.cat(branches, dim=1)
        x = F.gelu(self.mix(x))
        x = self.dropout(self.norm(x))
        return F.gelu(x + residual)


class DeepLOBWindowModel(nn.Module):
    def __init__(
        self,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        conv_dim: int,
        gru_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.register_buffer("feature_mean", torch.tensor(feature_mean, dtype=torch.float32).view(1, 1, -1))
        self.register_buffer("feature_std", torch.tensor(feature_std, dtype=torch.float32).view(1, 1, -1))
        self.stem = nn.Conv1d(FEATURE_DIM, conv_dim, kernel_size=1)
        self.block1 = InceptionResidualBlock(conv_dim, conv_dim, dropout=dropout)
        self.block2 = InceptionResidualBlock(conv_dim, conv_dim, dropout=dropout)
        self.gru = nn.GRU(
            input_size=conv_dim,
            hidden_size=gru_dim,
            num_layers=1,
            batch_first=True,
        )
        self.attn_proj = nn.Linear(gru_dim, gru_dim)
        self.attn_score = nn.Linear(gru_dim, 1)
        self.head = nn.Sequential(
            nn.Linear(conv_dim + gru_dim * 2, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, len(TARGET_COLUMNS)),
        )

    def build_features(self, raw: torch.Tensor) -> torch.Tensor:
        delta = torch.zeros_like(raw)
        delta[:, 1:, :] = raw[:, 1:, :] - raw[:, :-1, :]

        bid_p = raw[:, :, 0:6]
        ask_p = raw[:, :, 6:12]
        bid_v = raw[:, :, 12:18]
        ask_v = raw[:, :, 18:24]
        trade_p = raw[:, :, 24:28]
        trade_v = raw[:, :, 28:32]

        bid_p_mean = bid_p.mean(dim=2, keepdim=True)
        ask_p_mean = ask_p.mean(dim=2, keepdim=True)
        bid_v_sum = bid_v.sum(dim=2, keepdim=True)
        ask_v_sum = ask_v.sum(dim=2, keepdim=True)
        top_bid_v = bid_v[:, :, :1]
        top_ask_v = ask_v[:, :, :1]
        top_bid_p = bid_p[:, :, :1]
        top_ask_p = ask_p[:, :, :1]
        trade_p_mean = trade_p.mean(dim=2, keepdim=True)
        trade_v_sum = trade_v.sum(dim=2, keepdim=True)

        volume_imbalance = (bid_v_sum - ask_v_sum) / (bid_v_sum + ask_v_sum + 1e-3)
        top_volume_imbalance = (top_bid_v - top_ask_v) / (top_bid_v + top_ask_v + 1e-3)
        spread = ask_p_mean - bid_p_mean
        top_spread = top_ask_p - top_bid_p

        engineered = torch.cat(
            [
                bid_p_mean,
                ask_p_mean,
                spread,
                top_spread,
                bid_v_sum,
                ask_v_sum,
                volume_imbalance,
                top_volume_imbalance,
                trade_p_mean,
                trade_v_sum,
            ],
            dim=2,
        )
        features = torch.cat([raw, delta, engineered], dim=2)
        return (features - self.feature_mean) / self.feature_std

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        x = self.build_features(raw)
        x = x.transpose(1, 2)
        x = F.gelu(self.stem(x))
        x = self.block1(x)
        x = self.block2(x)

        conv_last = x[:, :, -1]
        x = x.transpose(1, 2)
        gru_out, _ = self.gru(x)
        last_hidden = gru_out[:, -1, :]
        recent = gru_out[:, -32:, :]
        attn_logits = self.attn_score(torch.tanh(self.attn_proj(recent))).squeeze(-1)
        attn_weights = torch.softmax(attn_logits, dim=1)
        attn_context = torch.sum(attn_weights.unsqueeze(-1) * recent, dim=1)

        fused = torch.cat([conv_last, last_hidden, attn_context], dim=1)
        output = self.head(fused)
        return 6.0 * torch.tanh(output / 6.0)


def evaluate_exact(
    model: DeepLOBWindowModel,
    parquet_path: str | Path,
    context_len: int,
    batch_size: int,
    batch_rows: int,
    device: torch.device,
) -> dict:
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for raw_chunk, target_chunk, mask_chunk in iter_sequence_chunks(parquet_path, include_targets=True, batch_rows=batch_rows):
            windows_view = make_windows_view(raw_chunk, context_len)
            for seq_idx in range(raw_chunk.shape[0]):
                valid_steps = np.flatnonzero(mask_chunk[seq_idx])
                if valid_steps.size == 0:
                    continue

                seq_targets = target_chunk[seq_idx, valid_steps]
                seq_windows = np.ascontiguousarray(windows_view[seq_idx, valid_steps], dtype=np.float32)

                for start in range(0, valid_steps.size, batch_size):
                    stop = min(start + batch_size, valid_steps.size)
                    batch_windows = torch.tensor(seq_windows[start:stop], dtype=torch.float32, device=device)
                    batch_predictions = model(batch_windows).cpu().numpy()
                    predictions.append(batch_predictions)
                    targets.append(seq_targets[start:stop])

    pred = np.concatenate(predictions, axis=0)
    true = np.concatenate(targets, axis=0)
    scores = {}
    for idx, name in enumerate(TARGET_COLUMNS):
        scores[name] = weighted_pearson_correlation(true[:, idx], pred[:, idx])
    scores["weighted_pearson"] = float(np.mean([scores[name] for name in TARGET_COLUMNS]))
    return scores


def save_checkpoint(
    checkpoint_path: Path,
    model: DeepLOBWindowModel,
    conv_dim: int,
    gru_dim: int,
    dropout: float,
    context_len: int,
) -> None:
    checkpoint = {
        "state_dict": model.state_dict(),
        "feature_mean": model.feature_mean.detach().cpu().reshape(-1).tolist(),
        "feature_std": model.feature_std.detach().cpu().reshape(-1).tolist(),
        "conv_dim": int(conv_dim),
        "gru_dim": int(gru_dim),
        "dropout": float(dropout),
        "context_len": int(context_len),
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[DeepLOBWindowModel, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = DeepLOBWindowModel(
        feature_mean=np.asarray(checkpoint["feature_mean"], dtype=np.float32),
        feature_std=np.asarray(checkpoint["feature_std"], dtype=np.float32),
        conv_dim=int(checkpoint["conv_dim"]),
        gru_dim=int(checkpoint["gru_dim"]),
        dropout=float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    return model, checkpoint


def export_onnx(model: DeepLOBWindowModel, context_len: int, onnx_path: Path) -> None:
    model.eval()
    dummy_input = torch.zeros(1, context_len, RAW_DIM, dtype=torch.float32)
    torch.onnx.export(
        model.cpu(),
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )


def verify_onnx_export(model: DeepLOBWindowModel, context_len: int, onnx_path: Path) -> float:
    model.eval()
    sample = torch.randn(4, context_len, RAW_DIM, dtype=torch.float32)
    with torch.no_grad():
        torch_output = model(sample).cpu().numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_output = session.run(["output"], {"input": sample.cpu().numpy()})[0]
    return float(np.max(np.abs(torch_output - onnx_output)))


def write_config(config_path: Path, onnx_path: Path, context_len: int) -> None:
    config = {
        "onnx_path": onnx_path.name,
        "context_len": int(context_len),
        "raw_dim": RAW_DIM,
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def train_model(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    torch.set_num_threads(max(1, min(args.torch_threads, os.cpu_count() or 1)))
    device = torch.device("cpu")

    if args.resume_checkpoint and Path(args.resume_checkpoint).exists():
        print(f"Resuming from {args.resume_checkpoint}")
        model, checkpoint = load_checkpoint(Path(args.resume_checkpoint), device=device)
        args.context_len = int(checkpoint["context_len"])
        args.conv_dim = int(checkpoint["conv_dim"])
        args.gru_dim = int(checkpoint["gru_dim"])
        args.dropout = float(checkpoint["dropout"])
    else:
        print(f"Computing feature stats from {args.train_path}")
        feature_mean, feature_std = compute_feature_stats(args.train_path, batch_rows=args.batch_rows)
        model = DeepLOBWindowModel(
            feature_mean=feature_mean,
            feature_std=feature_std,
            conv_dim=args.conv_dim,
            gru_dim=args.gru_dim,
            dropout=args.dropout,
        ).to(device)

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
            windows, targets = sample_training_windows(
                raw_chunk=raw_chunk,
                target_chunk=target_chunk,
                mask_chunk=mask_chunk,
                context_len=args.context_len,
                stride=args.train_stride,
                rng=epoch_rng,
            )

            if windows.shape[0] == 0:
                continue

            order = epoch_rng.permutation(windows.shape[0])
            for start in range(0, windows.shape[0], args.batch_size):
                stop = min(start + args.batch_size, windows.shape[0])
                batch_idx = order[start:stop]
                batch_windows = torch.tensor(windows[batch_idx], dtype=torch.float32, device=device)
                batch_targets = torch.tensor(targets[batch_idx], dtype=torch.float32, device=device)

                optimizer.zero_grad(set_to_none=True)
                predictions = model(batch_windows)
                loss = mixed_loss(batch_targets, predictions, alpha=args.alpha, beta=args.beta)
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
            context_len=args.context_len,
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
                conv_dim=args.conv_dim,
                gru_dim=args.gru_dim,
                dropout=args.dropout,
                context_len=args.context_len,
            )

    best_model, checkpoint = load_checkpoint(checkpoint_path, device=device)
    export_onnx(best_model, context_len=int(checkpoint["context_len"]), onnx_path=onnx_path)
    write_config(config_path=config_path, onnx_path=onnx_path, context_len=int(checkpoint["context_len"]))
    max_abs_diff = verify_onnx_export(best_model, context_len=int(checkpoint["context_len"]), onnx_path=onnx_path)
    final_scores = evaluate_exact(
        model=best_model,
        parquet_path=args.valid_path,
        context_len=int(checkpoint["context_len"]),
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
        context_len=int(checkpoint["context_len"]),
        batch_size=args.eval_batch_size,
        batch_rows=args.batch_rows,
        device=device,
    )
    print(json.dumps(scores, indent=2))


def export_checkpoint(args: argparse.Namespace) -> None:
    device = torch.device("cpu")
    model, checkpoint = load_checkpoint(Path(args.checkpoint_path), device=device)
    context_len = int(checkpoint["context_len"])
    onnx_path = Path(args.onnx_path)
    config_path = Path(args.config_path)

    export_onnx(model, context_len=context_len, onnx_path=onnx_path)
    write_config(config_path=config_path, onnx_path=onnx_path, context_len=context_len)
    max_abs_diff = verify_onnx_export(model, context_len=context_len, onnx_path=onnx_path)

    print(f"Exported ONNX to {onnx_path}")
    print(f"Saved config to {config_path}")
    print(f"ONNX max abs diff: {max_abs_diff:.8f}")


class PredictionModel:
    def __init__(self, model_path: str = "", config_path: str = ""):
        self.current_seq_ix = None
        self.raw_history: list[np.ndarray] = []

        self.model_path = Path(model_path) if model_path else DEFAULT_ONNX_PATH
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self.context_len = DEFAULT_CONTEXT_LEN

        if self.config_path.exists():
            config = json.loads(self.config_path.read_text(encoding="utf-8"))
            self.context_len = int(config["context_len"])
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
        self.output_name = self.ort_session.get_outputs()[0].name

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

        output = self.ort_session.run([self.output_name], {self.input_name: window[None, :, :]})[0][0]
        return output.astype(np.float32, copy=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Improved LOB solution with training, validation, and ONNX export.")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train on train.parquet, validate on valid.parquet, export ONNX.")
    train_parser.add_argument("--train-path", default=str(CURRENT_DIR.parent / "datasets" / "train.parquet"))
    train_parser.add_argument("--valid-path", default=str(CURRENT_DIR.parent / "datasets" / "valid.parquet"))
    train_parser.add_argument("--checkpoint-path", default=str(DEFAULT_CHECKPOINT_PATH))
    train_parser.add_argument("--onnx-path", default=str(DEFAULT_ONNX_PATH))
    train_parser.add_argument("--config-path", default=str(DEFAULT_CONFIG_PATH))
    train_parser.add_argument("--resume-checkpoint")
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--batch-size", type=int, default=256)
    train_parser.add_argument("--eval-batch-size", type=int, default=512)
    train_parser.add_argument("--train-stride", type=int, default=24)
    train_parser.add_argument("--context-len", type=int, default=DEFAULT_CONTEXT_LEN)
    train_parser.add_argument("--conv-dim", type=int, default=64)
    train_parser.add_argument("--gru-dim", type=int, default=64)
    train_parser.add_argument("--dropout", type=float, default=0.05)
    train_parser.add_argument("--lr", type=float, default=3e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--alpha", type=float, default=0.75)
    train_parser.add_argument("--beta", type=float, default=0.7)
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
