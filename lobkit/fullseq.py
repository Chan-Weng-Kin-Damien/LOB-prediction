"""Full-sequence training for causal window models.

The TCN's head is ``Conv1d(hidden, 2, kernel_size=1)``: it already emits a
prediction at *every* timestep, and the original training loop then threw all but
one away (``x = x[:, :, -1]``). Because the network is fully causal, one forward
pass over a padded 1000-step sequence reproduces the windowed prediction at every
step exactly -- so the discarded outputs are 901 free supervised targets per
sequence instead of 1, at essentially unchanged cost.

That equivalence is only valid while the receptive field fits inside the context
window, and it is currently **exactly saturated**, not comfortably under. Probing
the exported graph one input position at a time shows the output depends on all
128 window positions:

    conv stack, dilations [1,2,4,8,16,32], kernel 3 -> 1 + 2*63 = 127
    build_features computes delta[t] = raw[t] - raw[t-1]  -> +1
    total raw lookback                                    = 128 = context_len

So there is zero slack. Adding a seventh dilation block would take the receptive
field to 255, silently breaking the correspondence between how the model trains
and how it is scored. ``assert_equivalence_safe`` makes that a hard failure
rather than a silent one.

This does **not** generalise to the DeepLOB or t1-specialist models: both contain
an ``nn.GRU`` whose memory is unbounded, so a full-sequence pass genuinely
differs from windowed inference.
"""

from __future__ import annotations

import numpy as np

from .windows import receptive_field


class ReceptiveFieldError(RuntimeError):
    pass


def raw_lookback(dilations: list[int], kernel_size: int = 3, uses_delta_features: bool = True) -> int:
    """Steps of *raw input* the output at one position depends on."""
    return receptive_field(dilations, kernel_size) + (1 if uses_delta_features else 0)


def assert_equivalence_safe(
    dilations: list[int],
    context_len: int,
    kernel_size: int = 3,
    uses_delta_features: bool = True,
) -> int:
    """Guard the invariant that makes full-sequence training equivalent."""
    lookback = raw_lookback(dilations, kernel_size, uses_delta_features)
    if lookback > context_len:
        raise ReceptiveFieldError(
            f"raw lookback {lookback} exceeds context_len {context_len}: full-sequence "
            f"training would no longer match windowed inference. "
            f"(conv receptive field {receptive_field(dilations, kernel_size)}"
            f"{' + 1 for the delta feature' if uses_delta_features else ''}). "
            f"Either shorten the dilation stack or raise context_len."
        )
    return lookback


def pad_sequences(raw: "np.ndarray", context_len: int):
    """Left zero-pad whole sequences so position ``context_len-1+t`` predicts step ``t``.

    Matches the inference convention exactly: the window path pads short
    histories with zeros on the left, and the delta feature at the pad boundary
    evaluates to ``raw[0] - 0`` in both paths.
    """
    import torch

    if isinstance(raw, np.ndarray):
        raw = torch.from_numpy(raw)
    pad = torch.zeros(
        (raw.shape[0], context_len - 1, raw.shape[2]), dtype=raw.dtype, device=raw.device
    )
    return torch.cat([pad, raw], dim=1)


def masked_sequence_loss(
    predictions,
    targets,
    mask,
    alpha: float,
    beta: float,
):
    """``mixed_loss`` over every scored step in a batch of sequences.

    ``predictions`` and ``targets`` are ``(B, T, 2)``, ``mask`` is ``(B, T)``.

    The correlation term in ``mixed_loss`` is estimated *within the batch*. Under
    the old windowed loop a batch held 256 samples (sometimes 8), which is why
    ``beta`` was pinned at 0.7 to lean on MSE instead. Here a batch of 128
    sequences carries ~115,000 scored ticks, so the batch correlation is a
    near-noiseless estimate of the population value and ``beta = 0`` -- training
    purely on the metric -- becomes viable.
    """
    from .losses import mixed_loss

    flat_pred = predictions[mask]
    flat_true = targets[mask]
    return mixed_loss(flat_true, flat_pred, alpha=alpha, beta=beta)


def check_numerical_equivalence(
    model,
    sequence: "np.ndarray",
    steps: "np.ndarray",
    context_len: int,
    tolerance: float = 1e-4,
) -> float:
    """Compare full-sequence outputs against per-window outputs on real data.

    The config guard proves the equivalence in principle; this proves the
    implementation actually realises it. Run before launching any long job -- if
    it fails, every downstream number is fiction.

    Returns the max absolute difference.
    """
    import torch

    from .windows import make_window

    model.eval()
    with torch.no_grad():
        padded = pad_sequences(np.asarray(sequence, dtype=np.float32)[None], context_len)
        full = model(padded)
        if full.dim() == 3 and full.shape[1] != padded.shape[1]:
            full = full.transpose(1, 2)
        full_at_steps = full[0, np.asarray(steps) + context_len - 1].cpu().numpy()

        windowed = np.stack(
            [
                model(torch.from_numpy(make_window(sequence, int(s), context_len))[None])[0]
                .cpu()
                .numpy()
                for s in steps
            ]
        )

    max_diff = float(np.max(np.abs(full_at_steps - windowed)))
    if max_diff > tolerance:
        raise ReceptiveFieldError(
            f"full-sequence and windowed predictions differ by {max_diff:.3e} "
            f"(tolerance {tolerance:.1e}). Full-sequence training is NOT equivalent "
            f"for this model -- do not train it this way."
        )
    return max_diff
