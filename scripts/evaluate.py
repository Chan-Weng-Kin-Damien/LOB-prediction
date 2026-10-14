#!/usr/bin/env python
"""Evaluate ONNX models on a named split, with bootstrap confidence intervals.

Every number this prints carries an error bar, because the differences being
chased here are small relative to the noise: a sequence-level bootstrap over 300
sequences gives a standard deviation of ~0.026 on an absolute score. Comparisons
between models use a *paired* bootstrap, which cancels the shared variance and
tightens the interval by roughly 4x.

    python scripts/evaluate.py --split blend_fit
    python scripts/evaluate.py --split blend_fit --limit 300 --save-predictions

The holdout is deliberately awkward to reach: pass ``--split holdout --i-am-done``.
It should be opened once, at the end, and reported whatever it says.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lobkit.data import LobCorpus  # noqa: E402
from lobkit.evaluate import (  # noqa: E402
    ModelSpec,
    predict_corpus,
    prediction_correlations,
    score_by_amplitude_decile,
    score_by_position,
    summarise,
)
from lobkit.metric import bootstrap_score, paired_bootstrap, score_both_targets  # noqa: E402
from lobkit.splits import HOLDOUT, load_splits  # noqa: E402

DEFAULT_CORPUS = Path.home() / ".cache" / "lob-prediction" / "corpus"

# The shipped blend, kept here so every report shows what is actually deployed
# next to whatever is being proposed.
SHIPPED_WEIGHTS = {
    "t0": {"baseline": 0.8915, "deeplob": 0.0, "tcn": 0.1085},
    "t1": {"baseline": 0.6582, "deeplob": 0.3418, "tcn": 0.0},
}


def default_model_specs(package_root: Path) -> dict[str, ModelSpec]:
    sub = package_root / "submission_safe_3model"
    return {
        "baseline": ModelSpec("baseline", sub / "baseline.onnx", 100),
        "deeplob": ModelSpec.from_config("deeplob", sub / "deeplob_attn_tuned.json"),
        "tcn": ModelSpec.from_config("tcn", sub / "improved_tcn.json"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parent.parent
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    parser.add_argument("--split", default="blend_fit")
    parser.add_argument("--limit", type=int, default=0, help="Use only the first N sequences.")
    parser.add_argument("--models", default="baseline,deeplob,tcn")
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        help="Add a model not in the submission: name=path/to/config.json "
        "(repeatable). Use to score a freshly trained candidate against the "
        "shipped ones on identical sequences.",
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--out", default=str(root / "artifacts"))
    parser.add_argument(
        "--i-am-done",
        action="store_true",
        help="Required to evaluate the sealed holdout. Opening it more than once "
        "invalidates it as an unbiased estimate.",
    )
    args = parser.parse_args()

    if args.split == HOLDOUT and not args.i_am_done:
        print(
            "Refusing to open the sealed holdout without --i-am-done.\n"
            "The holdout exists to give one unbiased number at the end of the work. "
            "If you evaluate it now and then keep tuning, it stops measuring what "
            "you need it to measure. Use --split blend_fit while iterating.",
            file=sys.stderr,
        )
        return 2

    corpus = LobCorpus(args.corpus)
    splits = load_splits(Path(args.corpus) / "splits.json", corpus.seq_ix)
    if args.split not in splits:
        print(f"unknown split {args.split!r}; have {sorted(splits)}", file=sys.stderr)
        return 1

    indices = splits[args.split]
    if args.limit:
        indices = indices[: args.limit]

    specs = default_model_specs(root)
    for entry in args.extra:
        if "=" not in entry:
            print(f"--extra expects name=path.json, got {entry!r}", file=sys.stderr)
            return 1
        name, cfg = entry.split("=", 1)
        specs[name.strip()] = ModelSpec.from_config(name.strip(), Path(cfg.strip()))
    wanted = [name.strip() for name in args.models.split(",") if name.strip()]
    missing = [n for n in wanted if n not in specs]
    if missing:
        print(f"unknown models: {missing}", file=sys.stderr)
        return 1

    y_true, seq_ids = corpus.flat_targets(indices)
    print(f"split={args.split}  sequences={indices.size:,}  scored ticks={y_true.shape[0]:,}\n")

    predictions = {}
    for name in wanted:
        started = time.perf_counter()
        predictions[name] = predict_corpus(specs[name], corpus, indices).astype(np.float64)
        print(f"  {name}: {time.perf_counter() - started:.1f}s")

    if all(m in predictions for m in ("baseline", "deeplob", "tcn")):
        predictions["shipped_blend"] = np.stack(
            [
                sum(SHIPPED_WEIGHTS[t][m] * predictions[m][:, i] for m in SHIPPED_WEIGHTS[t])
                for i, t in enumerate(("t0", "t1"))
            ],
            axis=1,
        )

    print("\n=== weighted Pearson ===")
    print(f"{'model':<16} {'t0':>9} {'t1':>9} {'mean':>9}   {'95% CI (mean)':>22}")
    for row in summarise(y_true, predictions):
        boot = bootstrap_score(
            y_true, predictions[row["model"]], seq_ids, n_resamples=args.bootstrap
        )
        print(
            f"{row['model']:<16} {row['t0']:>+9.5f} {row['t1']:>+9.5f} "
            f"{row['weighted_pearson']:>+9.5f}   [{boot['lo95']:+.5f}, {boot['hi95']:+.5f}]"
        )

    reference = "baseline" if "baseline" in predictions else wanted[0]
    print(f"\n=== paired comparison vs {reference} ===")
    print(f"{'model':<16} {'delta':>9}   {'95% CI':>22}  {'P(better)':>10}")
    for name, pred in predictions.items():
        if name == reference:
            continue
        cmp = paired_bootstrap(
            y_true, pred, predictions[reference], seq_ids, n_resamples=args.bootstrap
        )
        print(
            f"{name:<16} {cmp['delta']:>+9.5f}   "
            f"[{cmp['lo95']:+.5f}, {cmp['hi95']:+.5f}]  {cmp['p_a_better']:>10.3f}"
        )

    print("\n=== prediction correlation (ensemble diversity) ===")
    base_only = {k: v for k, v in predictions.items() if k in wanted}
    for target, pairs in prediction_correlations(base_only).items():
        rendered = "  ".join(f"{k}={v:.3f}" for k, v in pairs.items())
        print(f"  {target}: {rendered}")

    print("\n=== score by |y| decile (metric weight is concentrated here) ===")
    print(f"{'target':<7} {'decile':>7} {'n':>9} {'weight share':>13} {'wp':>9}")
    for row in score_by_amplitude_decile(y_true, predictions[reference]):
        print(
            f"{row['target']:<7} {row['decile']:>7} {row['n']:>9,} "
            f"{row['weight_share']:>12.1%} {row['wp']:>+9.4f}"
        )

    print("\n=== score by position in sequence (does more context help?) ===")
    print(f"{'steps':<14} {'t0':>9} {'t1':>9}")
    for row in score_by_position(y_true, predictions[reference], indices.size):
        print(f"{row['step_lo']:>4}-{row['step_hi']:<9} {row['t0']:>+9.4f} {row['t1']:>+9.4f}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "split": args.split,
        "n_sequences": int(indices.size),
        "n_ticks": int(y_true.shape[0]),
        "scores": {name: score_both_targets(y_true, p) for name, p in predictions.items()},
        "correlations": prediction_correlations(base_only),
    }
    (out_dir / f"eval_{args.split}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out_dir / f'eval_{args.split}.json'}")

    if args.save_predictions:
        path = out_dir / f"preds_{args.split}.npz"
        np.savez_compressed(
            path, y_true=y_true, seq_ids=seq_ids, **{k: v for k, v in predictions.items()}
        )
        print(f"wrote {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
