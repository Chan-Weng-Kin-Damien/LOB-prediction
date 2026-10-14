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
ENGINEERED_DIM = 26
FEATURE_DIM = RAW_DIM * 3 + ENGINEERED_DIM
VOLUME_SLICE = np.r_[12:24, 28:32]

DEFAULT_CONTEXT_LEN = 192
DEFAULT_ONNX_PATH = CURRENT_DIR / "tlob_lite.onnx"
DEFAULT_CHECKPOINT_PATH = CURRENT_DIR / "tlob_lite.pt"
DEFAULT_CONFIG_PATH = CURRENT_DIR / "tlob_lite.json"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def signed_log_numpy(values: np.ndarray) -> np.ndarray:
    return np.sign(values) * np.log1p(np.abs(values))


def prepare_raw_numpy(raw_batch: np.ndarray) -> np.ndarray:
    prepared = np.asarray(raw_batch, dtype=np.float32).copy()
    prepared[:, :, VOLUME_SLICE] = signed_log_numpy(prepared[:, :, VOLUME_SLICE])
    return prepared


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
        prepared = prepare_raw_numpy(raw)
        mask = arrays["need_prediction"].reshape(seq_count, SEQUENCE_LENGTH).astype(bool, copy=False)

        if include_targets:
            targets = np.stack(
                [arrays[column].reshape(seq_count, SEQUENCE_LENGTH) for column in TARGET_COLUMNS],
                axis=2,
            ).astype(np.float32, copy=False)
        else:
            targets = None

        yield prepared, targets, mask

    leftover = carry["step_in_seq"]
    if leftover is not None and leftover.size != 0:
        raise ValueError(f"{parquet_path} ended with a partial sequence")


def compute_anchor_stats(prepared_chunk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    anchor = prepared_chunk[:, :WARMUP_STEPS, :]
    mean = anchor.mean(axis=1, dtype=np.float32)
    std = np.maximum(anchor.std(axis=1, dtype=np.float32), 1e-3)
    return mean.astype(np.float32, copy=False), std.astype(np.float32, copy=False)


def make_windows_view(prepared_chunk: np.ndarray, context_len: int) -> np.ndarray:
    pad = np.zeros((prepared_chunk.shape[0], context_len - 1, RAW_DIM), dtype=np.float32)
    padded = np.concatenate([pad, prepared_chunk], axis=1)
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=context_len, axis=1)
    return np.moveaxis(windows, -1, 2)


def sample_training_windows(
    prepared_chunk: np.ndarray,
    target_chunk: np.ndarray,
    mask_chunk: np.ndarray,
    anchor_mean_chunk: np.ndarray,
    anchor_std_chunk: np.ndarray,
    context_len: int,
    stride: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    windows_view = make_windows_view(prepared_chunk, context_len)
    seq_indices = []
    step_indices = []

    for seq_idx in range(prepared_chunk.shape[0]):
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
        empty_windows = np.zeros((0, context_len, RAW_DIM), dtype=np.float32)
        empty_anchors = np.zeros((0, RAW_DIM), dtype=np.float32)
        empty_targets = np.zeros((0, len(TARGET_COLUMNS)), dtype=np.float32)
        return empty_windows, empty_anchors, empty_anchors.copy(), empty_targets

    seq_idx_arr = np.concatenate(seq_indices, axis=0)
    step_idx_arr = np.concatenate(step_indices, axis=0)
    windows = np.ascontiguousarray(windows_view[seq_idx_arr, step_idx_arr], dtype=np.float32)
    anchor_mean = np.ascontiguousarray(anchor_mean_chunk[seq_idx_arr], dtype=np.float32)
    anchor_std = np.ascontiguousarray(anchor_std_chunk[seq_idx_arr], dtype=np.float32)
    targets = np.ascontiguousarray(target_chunk[seq_idx_arr, step_idx_arr], dtype=np.float32)
    return windows, anchor_mean, anchor_std, targets


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


class TemporalConvStem(nn.Module):
    def __init__(self, model_dim: int, dropout: float):
        super().__init__()
        self.depthwise = nn.Conv1d(model_dim, model_dim, kernel_size=5, padding=2, groups=model_dim)
        self.pointwise = nn.Conv1d(model_dim, model_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.gelu(self.depthwise(x))
        x = self.dropout(self.pointwise(x))
        return x + residual


class MixerBlock(nn.Module):
    def __init__(self, context_len: int, model_dim: int, token_hidden: int, channel_hidden: int, dropout: float):
        super().__init__()
        self.token_norm = nn.LayerNorm(model_dim)
        self.token_fc1 = nn.Linear(context_len, token_hidden)
        self.token_fc2 = nn.Linear(token_hidden, context_len)
        self.channel_norm = nn.LayerNorm(model_dim)
        self.channel_fc1 = nn.Linear(model_dim, channel_hidden)
        self.channel_fc2 = nn.Linear(channel_hidden, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        token_mix = self.token_norm(x).transpose(1, 2)
        token_mix = self.token_fc2(self.dropout(F.gelu(self.token_fc1(token_mix))))
        x = x + token_mix.transpose(1, 2)

        channel_mix = self.channel_norm(x)
        channel_mix = self.channel_fc2(self.dropout(F.gelu(self.channel_fc1(channel_mix))))
        return x + channel_mix


class TLOBLiteModel(nn.Module):
    def __init__(
        self,
        context_len: int,
        model_dim: int,
        num_blocks: int,
        token_hidden: int,
        channel_hidden: int,
        dropout: float,
    ):
        super().__init__()
        self.context_len = int(context_len)
        self.input_proj = nn.Linear(FEATURE_DIM, model_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.positional_embedding = nn.Parameter(torch.zeros(1, context_len, model_dim))
        self.stem = TemporalConvStem(model_dim=model_dim, dropout=dropout)
        self.blocks = nn.ModuleList(
            [
                MixerBlock(
                    context_len=context_len,
                    model_dim=model_dim,
                    token_hidden=token_hidden,
                    channel_hidden=channel_hidden,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_norm = nn.LayerNorm(model_dim)
        self.attn_proj = nn.Linear(model_dim, model_dim)
        self.attn_score = nn.Linear(model_dim, 1)
        self.shared_head = nn.Sequential(
            nn.Linear(model_dim * 3, 192),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(192, 96),
            nn.GELU(),
        )
        self.head_t0 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(96, 48),
            nn.GELU(),
            nn.Linear(48, 1),
        )
        self.head_t1 = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(96, 48),
            nn.GELU(),
            nn.Linear(48, 1),
        )

    def build_features(
        self,
        prepared_window: torch.Tensor,
        anchor_mean: torch.Tensor,
        anchor_std: torch.Tensor,
    ) -> torch.Tensor:
        norm = (prepared_window - anchor_mean.unsqueeze(1)) / anchor_std.unsqueeze(1)

        delta1 = torch.zeros_like(norm)
        delta1[:, 1:, :] = norm[:, 1:, :] - norm[:, :-1, :]

        delta4 = torch.zeros_like(norm)
        delta4[:, 4:, :] = norm[:, 4:, :] - norm[:, :-4, :]

        bid_p = norm[:, :, 0:6]
        ask_p = norm[:, :, 6:12]
        bid_v = norm[:, :, 12:18]
        ask_v = norm[:, :, 18:24]
        trade_p = norm[:, :, 24:28]
        trade_v = norm[:, :, 28:32]

        spread_levels = ask_p - bid_p

        bid_depth_top1 = bid_v[:, :, :1].sum(dim=2, keepdim=True)
        bid_depth_top3 = bid_v[:, :, :3].sum(dim=2, keepdim=True)
        bid_depth_top6 = bid_v.sum(dim=2, keepdim=True)
        ask_depth_top1 = ask_v[:, :, :1].sum(dim=2, keepdim=True)
        ask_depth_top3 = ask_v[:, :, :3].sum(dim=2, keepdim=True)
        ask_depth_top6 = ask_v.sum(dim=2, keepdim=True)

        imbalance_top1 = (bid_depth_top1 - ask_depth_top1) / (bid_depth_top1 + ask_depth_top1 + 1e-3)
        imbalance_top3 = (bid_depth_top3 - ask_depth_top3) / (bid_depth_top3 + ask_depth_top3 + 1e-3)
        imbalance_top6 = (bid_depth_top6 - ask_depth_top6) / (bid_depth_top6 + ask_depth_top6 + 1e-3)
        depth_pressure = bid_depth_top6 - ask_depth_top6

        ofi_base_top1 = bid_depth_top1 - ask_depth_top1
        ofi_base_top3 = bid_depth_top3 - ask_depth_top3
        ofi_base_top6 = bid_depth_top6 - ask_depth_top6
        ofi_top1 = torch.zeros_like(ofi_base_top1)
        ofi_top3 = torch.zeros_like(ofi_base_top3)
        ofi_top6 = torch.zeros_like(ofi_base_top6)
        ofi_top1[:, 1:, :] = ofi_base_top1[:, 1:, :] - ofi_base_top1[:, :-1, :]
        ofi_top3[:, 1:, :] = ofi_base_top3[:, 1:, :] - ofi_base_top3[:, :-1, :]
        ofi_top6[:, 1:, :] = ofi_base_top6[:, 1:, :] - ofi_base_top6[:, :-1, :]

        trade_sum = trade_v.sum(dim=2, keepdim=True)
        trade_notional = (trade_p * trade_v).sum(dim=2, keepdim=True)
        trade_mean = trade_p.mean(dim=2, keepdim=True)
        bid_price_slope = (bid_p[:, :, 1:] - bid_p[:, :, :-1]).mean(dim=2, keepdim=True)
        ask_price_slope = (ask_p[:, :, 1:] - ask_p[:, :, :-1]).mean(dim=2, keepdim=True)
        bid_volume_slope = (bid_v[:, :, 1:] - bid_v[:, :, :-1]).mean(dim=2, keepdim=True)
        ask_volume_slope = (ask_v[:, :, 1:] - ask_v[:, :, :-1]).mean(dim=2, keepdim=True)

        engineered = torch.cat(
            [
                spread_levels,
                bid_depth_top1,
                bid_depth_top3,
                bid_depth_top6,
                ask_depth_top1,
                ask_depth_top3,
                ask_depth_top6,
                imbalance_top1,
                imbalance_top3,
                imbalance_top6,
                ofi_top1,
                ofi_top3,
                ofi_top6,
                trade_sum,
                trade_notional,
                trade_mean,
                depth_pressure,
                bid_price_slope,
                ask_price_slope,
                bid_volume_slope,
                ask_volume_slope,
            ],
            dim=2,
        )
        return torch.cat([norm, delta1, delta4, engineered], dim=2)

    def forward(self, prepared_window: torch.Tensor, anchor_mean: torch.Tensor, anchor_std: torch.Tensor) -> torch.Tensor:
        x = self.build_features(prepared_window, anchor_mean, anchor_std)
        x = self.input_proj(x)
        x = self.input_dropout(x + self.positional_embedding)
        x = self.stem(x.transpose(1, 2)).transpose(1, 2)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        recent_len = min(48, self.context_len)
        recent = x[:, -recent_len:, :]
        last_hidden = x[:, -1, :]
        mean_recent = recent.mean(dim=1)
        attn_logits = self.attn_score(torch.tanh(self.attn_proj(recent))).squeeze(-1)
        attn_weights = torch.softmax(attn_logits, dim=1)
        attn_context = torch.sum(attn_weights.unsqueeze(-1) * recent, dim=1)

        shared = self.shared_head(torch.cat([last_hidden, mean_recent, attn_context], dim=1))
        outputs = torch.cat([self.head_t0(shared), self.head_t1(shared)], dim=1)
        return 6.0 * torch.tanh(outputs / 6.0)


def evaluate_exact(
    model: TLOBLiteModel,
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
        for prepared_chunk, target_chunk, mask_chunk in iter_sequence_chunks(parquet_path, include_targets=True, batch_rows=batch_rows):
            anchor_mean_chunk, anchor_std_chunk = compute_anchor_stats(prepared_chunk)
            windows_view = make_windows_view(prepared_chunk, context_len)
            for seq_idx in range(prepared_chunk.shape[0]):
                valid_steps = np.flatnonzero(mask_chunk[seq_idx])
                if valid_steps.size == 0:
                    continue

                seq_targets = target_chunk[seq_idx, valid_steps]
                seq_windows = np.ascontiguousarray(windows_view[seq_idx, valid_steps], dtype=np.float32)
                seq_anchor_mean = np.repeat(anchor_mean_chunk[seq_idx : seq_idx + 1], valid_steps.size, axis=0)
                seq_anchor_std = np.repeat(anchor_std_chunk[seq_idx : seq_idx + 1], valid_steps.size, axis=0)

                for start in range(0, valid_steps.size, batch_size):
                    stop = min(start + batch_size, valid_steps.size)
                    batch_windows = torch.tensor(seq_windows[start:stop], dtype=torch.float32, device=device)
                    batch_anchor_mean = torch.tensor(seq_anchor_mean[start:stop], dtype=torch.float32, device=device)
                    batch_anchor_std = torch.tensor(seq_anchor_std[start:stop], dtype=torch.float32, device=device)
                    batch_predictions = model(batch_windows, batch_anchor_mean, batch_anchor_std).cpu().numpy()
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
    model: TLOBLiteModel,
    model_dim: int,
    num_blocks: int,
    token_hidden: int,
    channel_hidden: int,
    dropout: float,
    context_len: int,
) -> None:
    checkpoint = {
        "state_dict": model.state_dict(),
        "model_dim": int(model_dim),
        "num_blocks": int(num_blocks),
        "token_hidden": int(token_hidden),
        "channel_hidden": int(channel_hidden),
        "dropout": float(dropout),
        "context_len": int(context_len),
    }
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[TLOBLiteModel, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = TLOBLiteModel(
        context_len=int(checkpoint["context_len"]),
        model_dim=int(checkpoint["model_dim"]),
        num_blocks=int(checkpoint["num_blocks"]),
        token_hidden=int(checkpoint["token_hidden"]),
        channel_hidden=int(checkpoint["channel_hidden"]),
        dropout=float(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    return model, checkpoint


def export_onnx(model: TLOBLiteModel, context_len: int, onnx_path: Path) -> None:
    model.eval()
    dummy_window = torch.zeros(1, context_len, RAW_DIM, dtype=torch.float32)
    dummy_anchor_mean = torch.zeros(1, RAW_DIM, dtype=torch.float32)
    dummy_anchor_std = torch.ones(1, RAW_DIM, dtype=torch.float32)
    torch.onnx.export(
        model.cpu(),
        (dummy_window, dummy_anchor_mean, dummy_anchor_std),
        onnx_path,
        input_names=["prepared_window", "anchor_mean", "anchor_std"],
        output_names=["output"],
        dynamic_axes={
            "prepared_window": {0: "batch_size"},
            "anchor_mean": {0: "batch_size"},
            "anchor_std": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )


def verify_onnx_export(model: TLOBLiteModel, context_len: int, onnx_path: Path) -> float:
    model.eval()
    sample_window = torch.randn(4, context_len, RAW_DIM, dtype=torch.float32)
    sample_anchor_mean = torch.randn(4, RAW_DIM, dtype=torch.float32)
    sample_anchor_std = torch.rand(4, RAW_DIM, dtype=torch.float32) + 0.5
    with torch.no_grad():
        torch_output = model(sample_window, sample_anchor_mean, sample_anchor_std).cpu().numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_output = session.run(
        ["output"],
        {
            "prepared_window": sample_window.cpu().numpy(),
            "anchor_mean": sample_anchor_mean.cpu().numpy(),
            "anchor_std": sample_anchor_std.cpu().numpy(),
        },
    )[0]
    return float(np.max(np.abs(torch_output - onnx_output)))


def write_config(config_path: Path, onnx_path: Path, context_len: int) -> None:
    config = {
        "onnx_path": onnx_path.name,
        "context_len": int(context_len),
        "raw_dim": RAW_DIM,
        "warmup_steps": WARMUP_STEPS,
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")


def train_model(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    torch.set_num_threads(max(1, min(args.torch_threads, os.cpu_count() or 1)))
    device = torch.device("cpu")

    model = TLOBLiteModel(
        context_len=args.context_len,
        model_dim=args.model_dim,
        num_blocks=args.num_blocks,
        token_hidden=args.token_hidden,
        channel_hidden=args.channel_hidden,
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

        for prepared_chunk, target_chunk, mask_chunk in iter_sequence_chunks(args.train_path, include_targets=True, batch_rows=args.batch_rows):
            anchor_mean_chunk, anchor_std_chunk = compute_anchor_stats(prepared_chunk)
            windows, anchor_mean, anchor_std, targets = sample_training_windows(
                prepared_chunk=prepared_chunk,
                target_chunk=target_chunk,
                mask_chunk=mask_chunk,
                anchor_mean_chunk=anchor_mean_chunk,
                anchor_std_chunk=anchor_std_chunk,
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
                batch_anchor_mean = torch.tensor(anchor_mean[batch_idx], dtype=torch.float32, device=device)
                batch_anchor_std = torch.tensor(anchor_std[batch_idx], dtype=torch.float32, device=device)
                batch_targets = torch.tensor(targets[batch_idx], dtype=torch.float32, device=device)

                optimizer.zero_grad(set_to_none=True)
                predictions = model(batch_windows, batch_anchor_mean, batch_anchor_std)
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
                model_dim=args.model_dim,
                num_blocks=args.num_blocks,
                token_hidden=args.token_hidden,
                channel_hidden=args.channel_hidden,
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
        self.prepared_history: list[np.ndarray] = []
        self.anchor_mean: np.ndarray | None = None
        self.anchor_std: np.ndarray | None = None

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
        self.window_input_name = self.ort_session.get_inputs()[0].name
        self.anchor_mean_input_name = self.ort_session.get_inputs()[1].name
        self.anchor_std_input_name = self.ort_session.get_inputs()[2].name
        self.output_name = self.ort_session.get_outputs()[0].name

    def _prepare_state(self, state: np.ndarray) -> np.ndarray:
        prepared = np.asarray(state, dtype=np.float32).copy()
        prepared[VOLUME_SLICE] = signed_log_numpy(prepared[VOLUME_SLICE])
        return prepared

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.prepared_history = []
            self.anchor_mean = None
            self.anchor_std = None

        self.prepared_history.append(self._prepare_state(data_point.state))
        if self.anchor_mean is None and len(self.prepared_history) >= WARMUP_STEPS:
            anchor = np.asarray(self.prepared_history[:WARMUP_STEPS], dtype=np.float32)
            self.anchor_mean = anchor.mean(axis=0, dtype=np.float32)
            self.anchor_std = np.maximum(anchor.std(axis=0, dtype=np.float32), 1e-3)

        if not data_point.need_prediction:
            return None

        if self.anchor_mean is None or self.anchor_std is None:
            raise ValueError("Anchor statistics are unavailable when a prediction is required.")

        if len(self.prepared_history) >= self.context_len:
            window = np.asarray(self.prepared_history[-self.context_len :], dtype=np.float32)
        else:
            pad_len = self.context_len - len(self.prepared_history)
            padding = np.zeros((pad_len, RAW_DIM), dtype=np.float32)
            window = np.concatenate([padding, np.asarray(self.prepared_history, dtype=np.float32)], axis=0)

        output = self.ort_session.run(
            [self.output_name],
            {
                self.window_input_name: window[None, :, :],
                self.anchor_mean_input_name: self.anchor_mean[None, :],
                self.anchor_std_input_name: self.anchor_std[None, :],
            },
        )[0][0]
        return output.astype(np.float32, copy=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TLOB-lite LOB solution with training, validation, and ONNX export.")
    subparsers = parser.add_subparsers(dest="command")

    train_parser = subparsers.add_parser("train", help="Train on train.parquet, validate on valid.parquet, export ONNX.")
    train_parser.add_argument("--train-path", default=str(CURRENT_DIR.parent / "datasets" / "train.parquet"))
    train_parser.add_argument("--valid-path", default=str(CURRENT_DIR.parent / "datasets" / "valid.parquet"))
    train_parser.add_argument("--checkpoint-path", default=str(DEFAULT_CHECKPOINT_PATH))
    train_parser.add_argument("--onnx-path", default=str(DEFAULT_ONNX_PATH))
    train_parser.add_argument("--config-path", default=str(DEFAULT_CONFIG_PATH))
    train_parser.add_argument("--epochs", type=int, default=5)
    train_parser.add_argument("--batch-size", type=int, default=256)
    train_parser.add_argument("--eval-batch-size", type=int, default=512)
    train_parser.add_argument("--train-stride", type=int, default=20)
    train_parser.add_argument("--context-len", type=int, default=DEFAULT_CONTEXT_LEN)
    train_parser.add_argument("--model-dim", type=int, default=96)
    train_parser.add_argument("--num-blocks", type=int, default=4)
    train_parser.add_argument("--token-hidden", type=int, default=128)
    train_parser.add_argument("--channel-hidden", type=int, default=192)
    train_parser.add_argument("--dropout", type=float, default=0.08)
    train_parser.add_argument("--lr", type=float, default=3e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
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
