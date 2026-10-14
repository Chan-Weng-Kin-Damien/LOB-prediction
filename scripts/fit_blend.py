#!/usr/bin/env python
"""Refit ensemble weights on saved out-of-fold predictions.

Consumes ``artifacts/preds_<split>.npz`` written by ``scripts/evaluate.py``.
Costs no GPU time.

Three candidate blends are compared under grouped CV so the comparison is
out-of-sample:

  shipped      the hardcoded literals currently in the submission
  simplex      weights on the probability simplex, maximising the metric itself
  gls          the closed-form correlation maximiser, included to be refuted

    python scripts/fit_blend.py --split blend_fit
    python scripts/fit_blend.py --split blend_fit --write-solution
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lobkit.blend import _blend, apply_blend, fit_blend, fit_gls_weights, fit_simplex_weights  # noqa: E402
from lobkit.metric import paired_bootstrap, score_both_targets, weighted_pearson_correlation  # noqa: E402

SHIPPED = {
    "t0": {"baseline": 0.8915, "deeplob": 0.0, "tcn": 0.1085},
    "t1": {"baseline": 0.6582, "deeplob": 0.3418, "tcn": 0.0},
}


def oof_predictions(y, preds, seq_ids, fitter, n_folds=4, seed=0):
    """Grouped-CV out-of-fold blend, so the reported score is honest."""
    unique = np.unique(seq_ids)
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(unique, rng.permutation(unique.size) % n_folds))
    fold = np.array([fold_of[s] for s in seq_ids])

    out = np.empty_like(y)
    for f in range(n_folds):
        tr, te = fold != f, fold == f
        for i in range(2):
            w = fitter(y[tr, i], preds[tr, i, :])
            out[te, i] = _blend(preds[te, i, :], w)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parent.parent
    parser.add_argument("--split", default="blend_fit")
    parser.add_argument("--artifacts", default=str(root / "artifacts"))
    parser.add_argument("--models", default="baseline,deeplob,tcn")
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--write-solution", action="store_true", help="Patch the submission weights.")
    args = parser.parse_args()

    path = Path(args.artifacts) / f"preds_{args.split}.npz"
    if not path.exists():
        print(f"missing {path}; run scripts/evaluate.py --split {args.split} --save-predictions first", file=sys.stderr)
        return 1

    data = np.load(path)
    names = [n.strip() for n in args.models.split(",") if n.strip()]
    y = data["y_true"].astype(np.float64)
    seq_ids = data["seq_ids"]
    preds = {n: data[n].astype(np.float64) for n in names}
    stacked = np.stack([preds[n] for n in names], axis=2)  # (N, 2, M)

    print(f"split={args.split}  models={names}  ticks={y.shape[0]:,}  sequences={np.unique(seq_ids).size:,}\n")

    # The shipped reference is always the three models actually deployed, even
    # when a candidate is being evaluated in place of one of them.
    shipped_names = [n for n in ("baseline", "deeplob", "tcn") if n in data]
    shipped = np.stack(
        [
            sum(SHIPPED[t][n] * data[n].astype(np.float64)[:, i] for n in shipped_names)
            for i, t in enumerate(("t0", "t1"))
        ],
        axis=1,
    )

    print("=== fitting simplex weights with CV-selected shrinkage ===")
    blend = fit_blend(y, preds, seq_ids, n_folds=args.folds)
    for target in ("t0", "t1"):
        rendered = "  ".join(f"{k}={v:.4f}" for k, v in blend["weights"][target].items())
        print(f"  {target}: {rendered}   (ridge={blend['ridge'][target]:g})")
    fitted = apply_blend(preds, blend)

    print("\n=== honest out-of-fold comparison ===")
    oof_simplex = oof_predictions(
        y, stacked, seq_ids, lambda yy, pp: fit_simplex_weights(yy, pp, ridge=0.01), args.folds
    )
    oof_gls = oof_predictions(
        y, stacked, seq_ids, lambda yy, pp: fit_gls_weights(yy, pp, ridge=1e-6), args.folds
    )

    candidates = {
        "baseline only": preds["baseline"],
        "shipped weights": shipped,
        "simplex (OOF)": oof_simplex,
        "gls closed form (OOF)": oof_gls,
        "simplex (in-sample)": fitted,
    }

    print(f"{'blend':<24} {'t0':>9} {'t1':>9} {'mean':>9}")
    for label, pred in candidates.items():
        s = score_both_targets(y, pred)
        print(f"{label:<24} {s['t0']:>+9.5f} {s['t1']:>+9.5f} {s['weighted_pearson']:>+9.5f}")

    print(f"\n=== paired bootstrap vs shipped weights ===")
    print(f"{'blend':<24} {'delta':>9}   {'95% CI':>22}  {'P(better)':>10}")
    for label in ("baseline only", "simplex (OOF)", "gls closed form (OOF)"):
        cmp = paired_bootstrap(y, candidates[label], shipped, seq_ids, n_resamples=args.bootstrap)
        print(
            f"{label:<24} {cmp['delta']:>+9.5f}   "
            f"[{cmp['lo95']:+.5f}, {cmp['hi95']:+.5f}]  {cmp['p_a_better']:>10.3f}"
        )

    print("\n=== why the closed form is not used ===")
    for i, target in enumerate(("t0", "t1")):
        w = fit_gls_weights(y[:, i], stacked[:, i, :], ridge=1e-6)
        insample = weighted_pearson_correlation(y[:, i], _blend(stacked[:, i, :], w))
        outsample = weighted_pearson_correlation(y[:, i], oof_gls[:, i])
        rendered = np.array2string(w, precision=3, floatmode="fixed")
        print(
            f"  {target}: weights {rendered}  in-sample {insample:+.5f} -> "
            f"out-of-fold {outsample:+.5f}  (drop {insample - outsample:+.5f})"
        )
    print(
        "  Correlation is scale-invariant, so the maximiser is the weighted-GLS\n"
        "  solution w = inv(Sigma) @ sigma. With base models correlated at ~0.88\n"
        "  that inversion is near-singular and amplifies noise; the simplex\n"
        "  constraint is doing real regularisation work."
    )

    out = Path(args.artifacts) / f"blend_{args.split}.json"
    out.write_text(json.dumps(blend, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")

    if args.write_solution:
        target_file = root / "submission_safe_3model" / "solution.py"
        print(f"\nProposed weights for {target_file}:")
        for key, attr in (("baseline", "baseline_weights"), ("deeplob", "deep_weights"), ("tcn", "tcn_weights")):
            t0 = blend["weights"]["t0"].get(key, 0.0)
            t1 = blend["weights"]["t1"].get(key, 0.0)
            print(f"        self.{attr} = np.array([{t0:.4f}, {t1:.4f}], dtype=np.float32)")
        print("\n(Printed rather than patched: review before editing the submission.)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
