#!/usr/bin/env python
"""One-time conversion of the parquet corpus into memory-mappable arrays.

Why: every training run used to make two full streaming passes over the 1.5 GB
``train.parquet`` -- one for ``compute_feature_stats`` and one per epoch -- and
the source files live on OneDrive, where cold reads can stall for minutes. After
this runs once, startup is instant and window sampling is a fancy-index gather.

Output (default ``~/.cache/lob-prediction/corpus``, deliberately *outside* the
OneDrive tree so 1.6 GB of binary is not sync-replicated; override with ``--out``,
and on a cluster point it at node-local scratch):

    all_raw.npy       (12165, 1000, 32) float32   1.56 GB
    all_targets.npy   (12165, 1000, 2)  float32   97 MB
    all_mask.npy      (12165, 1000)     bool      12 MB
    all_seq_ix.npy    (12165,)          int64
    all_source.npy    (12165,)          int8      0 = train.parquet, 1 = valid.parquet
    feature_stats.npz feature_mean/std over the 74 engineered dims
    splits.json       TRAIN / INNER_VAL / BLEND_FIT / HOLDOUT

Feature statistics are computed over TRAIN sequences only. Computing them over
the pooled corpus would leak holdout distribution information into the
normalisation constants baked into every exported graph.

    python scripts/materialise.py
    python scripts/materialise.py --out "$SCRATCH/lob" --limit-sequences 200   # smoke test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lobkit.data import (  # noqa: E402
    RAW_DIM,
    SEQUENCE_LENGTH,
    WARMUP_STEPS,
    engineered_features_numpy,
    iter_sequence_chunks,
)
from lobkit.splits import (  # noqa: E402
    N_ALREADY_MEASURED,
    SOURCE_TRAIN,
    SOURCE_VALID,
    TRAIN,
    build_splits,
    describe,
    save_splits,
)

DEFAULT_OUT = Path.home() / ".cache" / "lob-prediction" / "corpus"


def count_sequences(parquet_path: Path) -> int:
    import pyarrow.parquet as pq

    rows = pq.ParquetFile(parquet_path).metadata.num_rows
    if rows % SEQUENCE_LENGTH:
        raise ValueError(f"{parquet_path} has {rows} rows, not a multiple of {SEQUENCE_LENGTH}")
    return rows // SEQUENCE_LENGTH


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent.parent
    parser.add_argument("--train-path", default=str(here / "datasets" / "train.parquet"))
    parser.add_argument("--valid-path", default=str(here / "datasets" / "valid.parquet"))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--batch-rows", type=int, default=65536)
    parser.add_argument(
        "--limit-sequences",
        type=int,
        default=0,
        help="Truncate each source file to N sequences (smoke testing only).",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing corpus.")
    parser.add_argument(
        "--single-source",
        action="store_true",
        help="Build the corpus from --valid-path alone, splitting it into all four "
        "splits. For use while train.parquet is unavailable: it keeps the "
        "holdout sealed, but the resulting TRAIN split is NOT usable for "
        "training models that are then evaluated on the same file.",
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if (out / "all_raw.npy").exists() and not args.force:
        print(f"corpus already present at {out} (use --force to rebuild)")
        return 0

    if args.single_source:
        # Everything comes from one file and is labelled VALID, so build_splits
        # carves BLEND_FIT/HOLDOUT from it. A slice is labelled TRAIN purely so
        # feature statistics have somewhere to come from.
        sources = [(Path(args.valid_path), SOURCE_VALID)]
    else:
        sources = [(Path(args.train_path), SOURCE_TRAIN), (Path(args.valid_path), SOURCE_VALID)]
    for path, _ in sources:
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            return 1

    counts = []
    for path, _ in sources:
        n = count_sequences(path)
        if args.limit_sequences:
            n = min(n, args.limit_sequences)
        counts.append(n)
    total = sum(counts)
    if args.single_source:
        print(f"single-source corpus from {Path(args.valid_path).name}: {total:,} sequences")
    else:
        print(f"train sequences={counts[0]:,}  valid sequences={counts[1]:,}  total={total:,}")
    print(f"raw array will be {total * SEQUENCE_LENGTH * RAW_DIM * 4 / 1e9:.2f} GB at {out}")

    raw_out = np.lib.format.open_memmap(
        out / "all_raw.npy", mode="w+", dtype=np.float32, shape=(total, SEQUENCE_LENGTH, RAW_DIM)
    )
    tgt_out = np.lib.format.open_memmap(
        out / "all_targets.npy", mode="w+", dtype=np.float32, shape=(total, SEQUENCE_LENGTH, 2)
    )
    mask_out = np.lib.format.open_memmap(
        out / "all_mask.npy", mode="w+", dtype=bool, shape=(total, SEQUENCE_LENGTH)
    )
    source_out = np.empty(total, dtype=np.int8)

    cursor = 0
    started = time.perf_counter()
    for (path, source_id), limit in zip(sources, counts):
        print(f"reading {path.name} ...", flush=True)
        written_here = 0
        for raw, targets, mask in iter_sequence_chunks(
            path, include_targets=True, batch_rows=args.batch_rows
        ):
            take = raw.shape[0] if not args.limit_sequences else min(raw.shape[0], limit - written_here)
            if take <= 0:
                break
            raw_out[cursor : cursor + take] = raw[:take]
            tgt_out[cursor : cursor + take] = targets[:take]
            mask_out[cursor : cursor + take] = mask[:take]
            source_out[cursor : cursor + take] = source_id
            cursor += take
            written_here += take
            if written_here % 2000 < take:
                elapsed = time.perf_counter() - started
                print(f"  {written_here:,}/{limit:,} sequences ({elapsed:.0f}s)", flush=True)
            if args.limit_sequences and written_here >= limit:
                break

    if cursor != total:
        print(f"expected {total} sequences, wrote {cursor}", file=sys.stderr)
        return 1

    # Sequences carry no stable id in the files, so position is the id. It is
    # stable because the parquet row order is fixed.
    seq_ix = np.arange(total, dtype=np.int64)
    np.save(out / "all_seq_ix.npy", seq_ix)
    raw_out.flush()
    tgt_out.flush()
    mask_out.flush()

    expected_mask = np.arange(SEQUENCE_LENGTH) >= WARMUP_STEPS
    sample = np.random.default_rng(0).choice(total, size=min(64, total), replace=False)
    if not np.array_equal(mask_out[sample], np.broadcast_to(expected_mask, (sample.size, SEQUENCE_LENGTH))):
        print("WARNING: need_prediction is not exactly (step_in_seq >= 99)", file=sys.stderr)
    else:
        print(f"verified need_prediction == (step_in_seq >= {WARMUP_STEPS}) on {sample.size} sequences")

    # --- splits, then TRAIN-only feature statistics ---------------------------
    if args.single_source:
        # Reserve a quarter of the file for feature statistics only, then split
        # the remainder exactly as usual.
        n_stats = max(1, total // 4)
        source_out[:n_stats] = SOURCE_TRAIN
    np.save(out / "all_source.npy", source_out)
    already = np.flatnonzero(source_out == SOURCE_VALID)[:N_ALREADY_MEASURED]
    splits = build_splits(seq_ix, source_out, already_measured=already)
    save_splits(out / "splits.json", splits, seq_ix)
    print()
    print(describe(splits))
    print()

    print("computing feature statistics over TRAIN sequences only ...", flush=True)
    train_idx = splits[TRAIN]
    if train_idx.size == 0:
        print(
            "TRAIN split is empty -- no feature statistics written. This happens when "
            "the corpus is too small or --limit-sequences is set too low.",
            file=sys.stderr,
        )
        return 1
    n_acc = 0
    total_sum = None
    total_sq = None
    for start in range(0, train_idx.size, 256):
        block = train_idx[start : start + 256]
        feats = engineered_features_numpy(np.ascontiguousarray(raw_out[block]))
        flat = feats.reshape(-1, feats.shape[2]).astype(np.float64)
        if total_sum is None:
            total_sum = flat.sum(0)
            total_sq = (flat**2).sum(0)
        else:
            total_sum += flat.sum(0)
            total_sq += (flat**2).sum(0)
        n_acc += flat.shape[0]
        if start % 2048 == 0:
            print(f"  {start:,}/{train_idx.size:,} sequences", flush=True)

    mean = total_sum / n_acc
    var = np.maximum(total_sq / n_acc - mean**2, 0.0)
    std = np.sqrt(var)
    std[std < 1e-6] = 1.0
    np.savez(
        out / "feature_stats.npz",
        feature_mean=mean.astype(np.float32),
        feature_std=std.astype(np.float32),
        n_rows=np.int64(n_acc),
    )

    (out / "manifest.json").write_text(
        json.dumps(
            {
                "n_sequences": int(total),
                "sequence_length": SEQUENCE_LENGTH,
                "raw_dim": RAW_DIM,
                "warmup_steps": WARMUP_STEPS,
                "train_parquet": str(args.train_path),
                "valid_parquet": str(args.valid_path),
                "feature_stats_scope": "TRAIN split only",
                "limit_sequences": int(args.limit_sequences),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\ndone in {time.perf_counter() - started:.0f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
