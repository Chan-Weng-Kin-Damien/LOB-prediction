"""The TCN, in both its windowed and full-sequence forms.

Weights are shared between the two: the network is fully causal, so the
prediction at step ``t`` from a full-sequence pass equals the prediction the
windowed model makes from the 128-step window ending at ``t``. Training uses the
full-sequence form (901 supervised targets per sequence instead of 1); the
submission uses the windowed form, exported to ONNX with the same weights.

The architecture is unchanged from ``solution_example_improved.py`` -- same
layers, same ``build_features``, same output squashing -- so exported graphs stay
drop-in compatible with the existing submission.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

RAW_DIM = 32
ENGINEERED_DIM = 10
FEATURE_DIM = RAW_DIM * 2 + ENGINEERED_DIM  # 74
N_TARGETS = 2
DEFAULT_DILATIONS = [1, 2, 4, 8, 16, 32]


class CausalConvBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, kernel_size: int = 3, dropout: float = 0.05):
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel_size, dilation=dilation)
        self.mix = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.pad(x, (self.left_padding, 0))
        x = F.gelu(self.conv(x))
        x = self.dropout(x)
        x = self.mix(x)
        return F.gelu(x + residual)


class TCN(nn.Module):
    """``full_sequence=False`` reproduces the original ``WindowTCN`` exactly."""

    def __init__(
        self,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        hidden_dim: int = 48,
        dilations: list[int] | None = None,
        dropout: float = 0.05,
        full_sequence: bool = False,
    ):
        super().__init__()
        dilations = list(dilations or DEFAULT_DILATIONS)
        self.dilations = dilations
        self.full_sequence = full_sequence

        self.register_buffer(
            "feature_mean", torch.tensor(feature_mean, dtype=torch.float32).view(1, 1, -1)
        )
        self.register_buffer(
            "feature_std", torch.tensor(feature_std, dtype=torch.float32).view(1, 1, -1)
        )
        self.input_proj = nn.Conv1d(FEATURE_DIM, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            [CausalConvBlock(hidden_dim, dilation=d, dropout=dropout) for d in dilations]
        )
        self.head = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, N_TARGETS, kernel_size=1),
        )

    def build_features(self, raw: torch.Tensor) -> torch.Tensor:
        """32 raw + 32 first-differences + 10 microstructure = 74 channels.

        ``delta[0] = 0`` by construction. That convention is load-bearing: it is
        what makes the zero-padded windowed path and the zero-prefixed
        full-sequence path agree at the padding boundary.
        """
        delta = torch.zeros_like(raw)
        delta[:, 1:, :] = raw[:, 1:, :] - raw[:, :-1, :]

        bid_p, ask_p = raw[:, :, 0:6], raw[:, :, 6:12]
        bid_v, ask_v = raw[:, :, 12:18], raw[:, :, 18:24]
        trade_p, trade_v = raw[:, :, 24:28], raw[:, :, 28:32]

        bid_p_mean = bid_p.mean(dim=2, keepdim=True)
        ask_p_mean = ask_p.mean(dim=2, keepdim=True)
        bid_v_sum = bid_v.sum(dim=2, keepdim=True)
        ask_v_sum = ask_v.sum(dim=2, keepdim=True)
        top_bid_v, top_ask_v = bid_v[:, :, :1], ask_v[:, :, :1]
        top_bid_p, top_ask_p = bid_p[:, :, :1], ask_p[:, :, :1]

        engineered = torch.cat(
            [
                bid_p_mean,
                ask_p_mean,
                ask_p_mean - bid_p_mean,
                top_ask_p - top_bid_p,
                bid_v_sum,
                ask_v_sum,
                (bid_v_sum - ask_v_sum) / (bid_v_sum + ask_v_sum + 1e-3),
                (top_bid_v - top_ask_v) / (top_bid_v + top_ask_v + 1e-3),
                trade_p.mean(dim=2, keepdim=True),
                trade_v.sum(dim=2, keepdim=True),
            ],
            dim=2,
        )
        features = torch.cat([raw, delta, engineered], dim=2)
        return (features - self.feature_mean) / self.feature_std

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        x = self.build_features(raw).transpose(1, 2)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.head(x)

        if self.full_sequence:
            x = x.transpose(1, 2)  # (B, T, 2)
        else:
            x = x[:, :, -1]  # (B, 2), the original windowed contract
        return 6.0 * torch.tanh(x / 6.0)

    def windowed(self) -> "TCN":
        """A view of the same weights in windowed mode, for ONNX export."""
        self.full_sequence = False
        return self


def build_from_checkpoint(checkpoint: dict, full_sequence: bool = False) -> TCN:
    model = TCN(
        feature_mean=np.asarray(checkpoint["feature_mean"], dtype=np.float32),
        feature_std=np.asarray(checkpoint["feature_std"], dtype=np.float32),
        hidden_dim=int(checkpoint["hidden_dim"]),
        dilations=list(checkpoint["dilations"]),
        dropout=float(checkpoint.get("dropout", 0.05)),
        full_sequence=full_sequence,
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model
