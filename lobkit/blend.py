"""Ensemble weight fitting.

The shipped weights were hardcoded literals, fitted by an unrecorded procedure on
the same set used for epoch selection. Measured on 300 validation sequences they
are near-optimal for t0 (+0.0034 over the best single model) and actively harmful
for t1 (-0.0065; the optimal t1 weight there is 100% baseline).

The important subtlety is that the metric is a **scale-invariant correlation**,
so fitting by least squares optimises the wrong thing. Maximising correlation has
a closed form -- ``w ∝ Σ⁻¹σ``, the weighted GLS solution, since correlation is
invariant to affine rescaling of the blend. It is tempting and it does not work
here. Measured on a split-half fit/test with these three models:

    t0  in-sample  GLS 0.40695   ->  out-of-sample 0.38642
        constrained simplex      ->  out-of-sample 0.38771   (better)
    weights: [2.69, -0.56, 0.47]

The base models are collinear (baseline-deeplob prediction correlation 0.88), so
the unconstrained optimum is a near-singular inversion that amplifies noise. The
simplex constraint is doing real regularisation work.

So: constrain to the simplex, optimise the actual metric rather than a proxy, and
shrink toward equal weights with the shrinkage strength chosen out-of-sample.
"""

from __future__ import annotations

import numpy as np

from .metric import weighted_pearson_correlation


def _blend(preds: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """``preds`` is ``(N, M)`` for one target; ``weights`` is ``(M,)``."""
    return preds @ weights


def best_single_model(y_true: np.ndarray, preds: np.ndarray) -> np.ndarray:
    """One-hot weight vector on the strongest individual model."""
    scores = [weighted_pearson_correlation(y_true, preds[:, m]) for m in range(preds.shape[1])]
    w = np.zeros(preds.shape[1])
    w[int(np.argmax(scores))] = 1.0
    return w


def fit_simplex_weights(
    y_true: np.ndarray,
    preds: np.ndarray,
    ridge: float = 0.0,
    n_restarts: int = 4,
    seed: int = 0,
    shrink_to: str = "best",
) -> np.ndarray:
    """Maximise weighted Pearson over the probability simplex, for one target.

    ``ridge`` shrinks toward a prior: ``wp(y, P w) - ridge * ||w - w_prior||^2``.

    ``shrink_to`` picks that prior, and the choice matters more than it looks.
    Shrinking to ``"uniform"`` encodes "the models are equally good"; here they
    are not (0.284 / 0.233 / 0.221 mean weighted Pearson), so heavy shrinkage
    walks the blend *toward* the weak models and scored 0.271 out-of-fold --
    worse than the 0.284 of the strongest member alone. ``"best"`` shrinks toward
    a one-hot vector on the strongest model measured on the fitting fold, so the
    high-shrinkage limit degrades to "use the best single model" rather than to
    "average everything".

    Uses SLSQP when scipy is available and falls back to projected coordinate
    search otherwise, so the harness has no hard scipy dependency.
    """
    n_models = preds.shape[1]
    uniform = np.full(n_models, 1.0 / n_models)
    prior = uniform if shrink_to == "uniform" else best_single_model(y_true, preds)

    def negative_objective(w: np.ndarray) -> float:
        score = weighted_pearson_correlation(y_true, _blend(preds, w))
        return -(score - ridge * float(np.sum((w - prior) ** 2)))

    starts = [uniform, prior]
    rng = np.random.default_rng(seed)
    for _ in range(max(0, n_restarts - 1)):
        draw = rng.dirichlet(np.ones(n_models))
        starts.append(draw)
    for m in range(n_models):
        corner = np.zeros(n_models)
        corner[m] = 1.0
        starts.append(corner)

    best_w, best_val = uniform, negative_objective(uniform)

    try:
        from scipy.optimize import minimize

        constraints = {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}
        bounds = [(0.0, 1.0)] * n_models
        for start in starts:
            res = minimize(
                negative_objective,
                start,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 300, "ftol": 1e-10},
            )
            if res.success and res.fun < best_val:
                best_val, best_w = float(res.fun), np.asarray(res.x, dtype=float)
    except ImportError:
        for start in starts:
            w = np.asarray(start, dtype=float)
            val = negative_objective(w)
            for step in (0.2, 0.1, 0.05, 0.02, 0.01, 0.005):
                improved = True
                while improved:
                    improved = False
                    for i in range(n_models):
                        for j in range(n_models):
                            if i == j:
                                continue
                            candidate = w.copy()
                            move = min(step, candidate[j])
                            candidate[i] += move
                            candidate[j] -= move
                            cand_val = negative_objective(candidate)
                            if cand_val < val - 1e-12:
                                w, val, improved = candidate, cand_val, True
            if val < best_val:
                best_val, best_w = val, w

    best_w = np.clip(best_w, 0.0, None)
    total = best_w.sum()
    return best_w / total if total > 0 else uniform


def fit_gls_weights(y_true: np.ndarray, preds: np.ndarray, ridge: float = 0.0) -> np.ndarray:
    """Closed-form ``w ∝ Σ⁻¹σ`` maximiser, kept for the ablation.

    Included so the writeup can show the theoretically optimal blend losing
    out-of-sample rather than merely asserting that it would. Do not ship this.
    """
    w = np.maximum(np.abs(y_true), 1e-8)
    w = w / w.sum()
    centred_p = preds - (preds * w[:, None]).sum(0)
    centred_y = y_true - float((y_true * w).sum())
    sigma = centred_p.T @ (centred_p * w[:, None])
    cov = centred_p.T @ (centred_y * w)
    return np.linalg.solve(sigma + ridge * np.eye(preds.shape[1]), cov)


def select_ridge(
    y_true: np.ndarray,
    preds: np.ndarray,
    seq_ids: np.ndarray,
    ridge_grid=(0.0, 0.001, 0.01, 0.05, 0.2, 1.0),
    n_folds: int = 4,
    seed: int = 0,
    shrink_to: str = "best",
) -> tuple[float, np.ndarray, dict]:
    """Pick shrinkage by grouped CV, then refit on everything.

    Folds are grouped by sequence for the same reason the outer splits are: the
    901 ticks inside a sequence are not independent observations.
    """
    unique_seqs = np.unique(seq_ids)
    rng = np.random.default_rng(seed)
    fold_of_seq = dict(zip(unique_seqs, rng.permutation(unique_seqs.size) % n_folds))
    fold = np.array([fold_of_seq[s] for s in seq_ids])

    scores = {}
    for ridge in ridge_grid:
        oof = np.empty_like(y_true)
        for f in range(n_folds):
            train_sel, test_sel = fold != f, fold == f
            w = fit_simplex_weights(
                y_true[train_sel], preds[train_sel], ridge=ridge, seed=seed, shrink_to=shrink_to
            )
            oof[test_sel] = _blend(preds[test_sel], w)
        scores[ridge] = weighted_pearson_correlation(y_true, oof)

    best_ridge = max(scores, key=scores.get)
    final_w = fit_simplex_weights(y_true, preds, ridge=best_ridge, seed=seed, shrink_to=shrink_to)
    return best_ridge, final_w, scores


def fit_blend(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    seq_ids: np.ndarray,
    n_folds: int = 4,
    seed: int = 0,
    shrink_to: str = "best",
) -> dict:
    """Fit per-target simplex weights with CV-selected shrinkage.

    Weights are fitted **per target** because t0 and t1 are different indicators
    with very different achievable skill (~0.40 vs ~0.17) and different model
    rankings -- not because they are different horizons.
    """
    names = list(predictions)
    result = {"models": names, "weights": {}, "ridge": {}, "cv_scores": {}}

    for idx, target in enumerate(("t0", "t1")):
        stacked = np.stack([predictions[n][:, idx] for n in names], axis=1)
        ridge, weights, scores = select_ridge(
            y_true[:, idx], stacked, seq_ids, n_folds=n_folds, seed=seed, shrink_to=shrink_to
        )
        result["weights"][target] = {n: float(w) for n, w in zip(names, weights)}
        result["ridge"][target] = float(ridge)
        result["cv_scores"][target] = {float(k): float(v) for k, v in scores.items()}

    return result


def apply_blend(predictions: dict[str, np.ndarray], blend: dict) -> np.ndarray:
    """Materialise blended predictions from a fitted blend spec."""
    n = next(iter(predictions.values())).shape[0]
    out = np.zeros((n, 2), dtype=np.float64)
    for idx, target in enumerate(("t0", "t1")):
        for name, weight in blend["weights"][target].items():
            out[:, idx] += weight * predictions[name][:, idx]
    return out
