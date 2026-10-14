"""Differentiable training losses.

Lifted verbatim from the five training scripts, where ``weighted_corr_loss`` and
``mixed_loss`` were byte-identical copies. Kept in its own module so ``metric.py``
stays importable without torch.
"""

from __future__ import annotations

import torch


def weighted_corr_loss(y_true: torch.Tensor, y_pred: torch.Tensor, alpha: float) -> torch.Tensor:
    """``1 - weighted Pearson``, per target column, averaged.

    ``alpha`` controls how hard the loss leans on large-amplitude ticks:
    weights are ``(|y| + 1e-3) ** alpha``. The competition metric is effectively
    ``alpha = 1``, so values below 1 trade metric alignment for gradient
    stability on the bulk of small moves.
    """
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


def mixed_loss(
    y_true: torch.Tensor, y_pred: torch.Tensor, alpha: float, beta: float
) -> torch.Tensor:
    """``beta * weighted-MSE + (1 - beta) * (1 - weighted-Pearson)``.

    The correlation term is what the metric actually rewards; the MSE term keeps
    the output scale anchored, which matters because predictions are clipped to
    [-6, 6] before scoring.

    Note the correlation term is estimated *within the batch*, so it is badly
    biased for small batches. The TCN's chunk-local sampler produced trailing
    batches of 8 (one optimizer step in three); ``lobkit.windows.shuffled_batches``
    exists to avoid exactly that.
    """
    weights = torch.pow(torch.abs(y_true) + 1e-3, alpha)
    mse = (weights * torch.square(y_pred - y_true)).sum() / weights.sum().clamp_min(1e-8)
    corr = weighted_corr_loss(y_true, y_pred, alpha=alpha)
    return beta * mse + (1.0 - beta) * corr
