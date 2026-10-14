#!/usr/bin/env python
"""Train the TCN on whole sequences, then export it in windowed form.

The original loop sampled 8 windows per sequence and trained for 2 epochs: 85,768
supervised targets per epoch and 670 optimizer steps in total. Because the
network is fully causal and its head already emits a prediction at every
timestep, one forward pass over a padded sequence supervises all 901 scored steps
instead of one -- ~112x more signal for essentially the same FLOPs.

That equivalence is checked twice before any training happens: once from the
config (receptive field must fit in the context window) and once numerically
(full-sequence outputs must match per-window outputs on real data). If either
fails the run aborts, because everything downstream would otherwise be measured
against a model that trains differently from how it is scored.

    python scripts/train_tcn.py --epochs 60
    python scripts/train_tcn.py --epochs 60 --beta 0.0 --seed 1 --tag corr
    python scripts/train_tcn.py --resume artifacts/tcn_fullseq.pt --epochs 20
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lobkit.data import LobCorpus, SEQUENCE_LENGTH, WARMUP_STEPS  # noqa: E402
from lobkit.fullseq import assert_equivalence_safe, check_numerical_equivalence  # noqa: E402
from lobkit.losses import mixed_loss  # noqa: E402
from lobkit.metric import weighted_pearson_correlation  # noqa: E402
from lobkit.runtime import pick_device, describe_device, set_seed  # noqa: E402
from lobkit.splits import INNER_VAL, TRAIN, load_splits  # noqa: E402
from lobkit.tcn import DEFAULT_DILATIONS, TCN  # noqa: E402
from lobkit.windows import parse_dilations  # noqa: E402

DEFAULT_CORPUS = Path.home() / ".cache" / "lob-prediction" / "corpus"


def cosine_lr(step: int, total: int, base_lr: float, warmup_frac: float = 0.05) -> float:
    warmup = max(1, int(total * warmup_frac))
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def evaluate(model, raw, targets, device, context_len, batch_size=64) -> dict:
    """Full-sequence scoring of the scored region. ~100x cheaper than the
    windowed evaluator it replaces, and exactly equivalent."""
    model.eval()
    preds, trues = [], []
    for start in range(0, raw.shape[0], batch_size):
        chunk = raw[start : start + batch_size].to(device)
        pad = torch.zeros(
            (chunk.shape[0], context_len - 1, chunk.shape[2]), dtype=chunk.dtype, device=device
        )
        out = model(torch.cat([pad, chunk], dim=1))[:, context_len - 1 :, :]
        preds.append(out[:, WARMUP_STEPS:, :].float().cpu().numpy())
        trues.append(targets[start : start + batch_size, WARMUP_STEPS:, :].numpy())

    p = np.concatenate(preds).reshape(-1, 2).astype(np.float64)
    t = np.concatenate(trues).reshape(-1, 2).astype(np.float64)
    scores = {n: weighted_pearson_correlation(t[:, i], p[:, i]) for i, n in enumerate(("t0", "t1"))}
    scores["weighted_pearson"] = float(np.mean([scores["t0"], scores["t1"]]))
    return scores


def save_checkpoint(path: Path, model, optimizer, epoch: int, best: float, args, stats) -> None:
    """Atomic, and complete enough to actually resume.

    The original checkpoints stored only weights, so resuming restarted Adam's
    moments from zero and reset ``best_score`` to -inf -- which meant the first
    post-resume epoch could silently overwrite a better checkpoint.
    """
    payload = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_score": float(best),
        "hidden_dim": int(args.hidden_dim),
        "dilations": list(parse_dilations(args.dilations)),
        "dropout": float(args.dropout),
        "context_len": int(args.context_len),
        "feature_mean": stats[0].tolist(),
        "feature_std": stats[1].tolist(),
        "args": {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool))},
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parent.parent
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--out", default=str(root / "artifacts"))
    parser.add_argument("--tag", default="fullseq")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument(
        "--beta",
        type=float,
        default=0.3,
        help="Weight on the MSE term. The old windowed loop needed 0.7 because a "
        "256-sample batch gives a hopelessly noisy correlation estimate; a "
        "full-sequence batch carries ~58k scored ticks, so leaning on the "
        "correlation term (low beta) is now viable and better matches the metric.",
    )
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--dilations", default="1,2,4,8,16,32")
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--skip-gate", action="store_true", help="Not recommended.")
    args = parser.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"tcn_{args.tag}.pt"

    dilations = parse_dilations(args.dilations)
    lookback = assert_equivalence_safe(dilations, args.context_len)
    print(f"device: {describe_device(device)}")
    print(
        f"receptive field over raw input: {lookback} steps "
        f"(conv {lookback - 1} + 1 for the delta feature) <= context {args.context_len}  OK"
    )

    corpus = LobCorpus(args.corpus)
    splits = load_splits(Path(args.corpus) / "splits.json", corpus.seq_ix)
    train_idx, val_idx = splits[TRAIN], splits[INNER_VAL]
    stats = (corpus.feature_mean, corpus.feature_std)

    print(f"train {train_idx.size:,} sequences | inner-val {val_idx.size:,} sequences")
    print(
        f"supervised targets per epoch: {train_idx.size * (SEQUENCE_LENGTH - WARMUP_STEPS):,} "
        f"(the windowed loop used 85,768 per epoch: 8 per sequence)"
    )

    model = TCN(
        feature_mean=stats[0],
        feature_std=stats[1],
        hidden_dim=args.hidden_dim,
        dilations=dilations,
        dropout=args.dropout,
        full_sequence=True,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,}")

    # --- the gate --------------------------------------------------------------
    if not args.skip_gate:
        probe = np.ascontiguousarray(corpus.raw[int(val_idx[0])], dtype=np.float32)
        steps = np.array([99, 100, 126, 127, 200, 500, 999])
        model.full_sequence = True
        cpu_model = TCN(
            feature_mean=stats[0], feature_std=stats[1], hidden_dim=args.hidden_dim,
            dilations=dilations, dropout=0.0, full_sequence=True,
        )
        cpu_model.load_state_dict(model.state_dict())

        class _Dual(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                self.m.full_sequence = x.shape[1] != args.context_len
                return self.m(x)

        diff = check_numerical_equivalence(_Dual(cpu_model), probe, steps, args.context_len)
        print(f"equivalence gate: max|full-sequence - windowed| = {diff:.3e}  PASS")

    # --- data ------------------------------------------------------------------
    train_raw = torch.from_numpy(np.ascontiguousarray(corpus.raw[train_idx]))
    train_tgt = torch.from_numpy(np.ascontiguousarray(corpus.targets[train_idx]))
    val_raw = torch.from_numpy(np.ascontiguousarray(corpus.raw[val_idx]))
    val_tgt = torch.from_numpy(np.ascontiguousarray(corpus.targets[val_idx]))
    print(f"resident training tensor: {train_raw.nbytes / 1e9:.2f} GB")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch, best = 0, -np.inf

    if args.resume and Path(args.resume).exists():
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["state_dict"])
        optimizer.load_state_dict(ck["optimizer"])
        start_epoch, best = int(ck["epoch"]), float(ck["best_score"])
        print(f"resumed from epoch {start_epoch} with best {best:+.5f} (optimizer state restored)")

    steps_per_epoch = math.ceil(train_idx.size / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    rng = np.random.default_rng(args.seed)
    history = []
    global_step = start_epoch * steps_per_epoch
    pad_len = args.context_len - 1

    print(f"\n{'epoch':>5} {'loss':>10} {'t0':>9} {'t1':>9} {'mean':>9} {'lr':>9} {'sec':>7}")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        model.full_sequence = True
        order = rng.permutation(train_idx.size)  # global shuffle, not chunk-local
        running, seen, skipped, t_start = 0.0, 0, 0, time.perf_counter()

        for start in range(0, order.size, args.batch_size):
            batch = np.sort(order[start : start + args.batch_size])
            # NOTE: no non_blocking=True. On MPS the async copy can be read
            # before it lands, which silently yields garbage and a NaN loss.
            # It only buys anything for pinned CUDA memory in any case.
            raw = train_raw[batch].to(device)
            tgt = train_tgt[batch].to(device)

            pad = torch.zeros((raw.shape[0], pad_len, raw.shape[2]), dtype=raw.dtype, device=device)
            out = model(torch.cat([pad, raw], dim=1))[:, pad_len:, :]

            pred = out[:, WARMUP_STEPS:, :].reshape(-1, 2)
            true = tgt[:, WARMUP_STEPS:, :].reshape(-1, 2)

            lr = cosine_lr(global_step, total_steps, args.lr)
            for group in optimizer.param_groups:
                group["lr"] = lr

            loss = mixed_loss(true, pred, alpha=args.alpha, beta=args.beta)

            # A single non-finite batch would otherwise poison every weight via
            # the optimizer step, turning the whole run into NaN with no
            # indication of where it started. Skip the batch and say so.
            if not torch.isfinite(loss):
                skipped += 1
                if skipped <= 5:
                    print(
                        f"      step {global_step}: non-finite loss, batch skipped "
                        f"(size {raw.shape[0]}, |pred|max {float(pred.abs().max()):.3f})"
                    )
                if skipped > 0.1 * steps_per_epoch:
                    print("      too many non-finite batches; aborting", file=sys.stderr)
                    return 1
                global_step += 1
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if not torch.isfinite(grad_norm):
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                continue
            optimizer.step()

            running += float(loss.item())
            seen += 1
            global_step += 1

        elapsed = time.perf_counter() - t_start
        if skipped:
            print(f"      {skipped}/{steps_per_epoch} batches skipped as non-finite")
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            scores = evaluate(model, val_raw, val_tgt, device, args.context_len)
            model.full_sequence = True
            print(
                f"{epoch + 1:>5} {running / max(seen, 1):>10.5f} {scores['t0']:>+9.5f} "
                f"{scores['t1']:>+9.5f} {scores['weighted_pearson']:>+9.5f} {lr:>9.2e} {elapsed:>7.1f}"
            )
            history.append({"epoch": epoch + 1, "loss": running / max(seen, 1), **scores})
            if scores["weighted_pearson"] > best:
                best = scores["weighted_pearson"]
                save_checkpoint(ckpt_path, model, optimizer, epoch + 1, best, args, stats)
        else:
            print(f"{epoch + 1:>5} {running / max(seen, 1):>10.5f} {'':>9} {'':>9} {'':>9} {lr:>9.2e} {elapsed:>7.1f}")

    print(f"\nbest inner-val weighted Pearson: {best:+.5f}")
    print(f"checkpoint: {ckpt_path}")
    (out_dir / f"tcn_{args.tag}_history.json").write_text(
        json.dumps({"history": history, "best": best, "args": vars(args)}, indent=2, default=str),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
