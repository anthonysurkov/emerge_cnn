"""Coverage-weighted, regularized k-mer regression of editing rate.

The target is the per-sequence editing rate ``mle = k / n``. Because that rate
is a binomial estimate whose reliability scales with read depth, every sequence
is weighted by its coverage ``n``. This is the weighted-least-squares (Gaussian)
approximation to a binomial GLM: it is fast, its coefficients read directly as a
motif map, and it is the interpretable baseline the CNN must beat. A ridge
penalty absorbs the strong collinearity between a k-mer feature and its
constituent shorter k-mers; lasso/elastic-net are available for hard selection.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import r2_score

from .featurize import feature_name


@dataclass(frozen=True)
class KmerRegression:
    model: object
    blocks: list
    metrics: dict
    coefficients: pd.DataFrame
    parameters: dict


def _split(n, test_size, seed):
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n)
    cut = int(round(n * (1 - test_size)))
    return permutation[:cut], permutation[cut:]


def _weighted_r2(y_true, y_pred, weights):
    mean = np.average(y_true, weights=weights)
    ss_res = np.sum(weights * (y_true - y_pred) ** 2)
    ss_tot = np.sum(weights * (y_true - mean) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def _make_model(penalty, alpha, l1_ratio):
    if penalty == "ridge":
        return Ridge(alpha=alpha, fit_intercept=True)
    if penalty == "lasso":
        return Lasso(alpha=alpha, fit_intercept=True, max_iter=5000)
    if penalty == "elasticnet":
        return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True, max_iter=5000)
    raise ValueError(f"unknown penalty {penalty!r}")


def fit_kmer_regression(
    design: sparse.csr_matrix,
    editing_rate,
    coverage,
    blocks,
    *,
    penalty: str = "ridge",
    alphas=(0.1, 1.0, 10.0, 100.0),
    l1_ratio: float = 0.5,
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 0,
    top: int = 40,
    verbose: bool = False,
) -> KmerRegression:
    """Fit and evaluate the regression, selecting alpha on a validation split.

    ``alphas`` is searched on a held-out validation slice of the training data by
    coverage-weighted R2; the best alpha is refit on the full training split and
    scored once on the untouched test split.
    """
    editing_rate = np.asarray(editing_rate, dtype=float)
    coverage = np.asarray(coverage, dtype=float)
    n = design.shape[0]
    train_index, test_index = _split(n, test_size, seed)
    inner_train, inner_val = _split(len(train_index), val_size, seed + 1)
    fit_index = train_index[inner_train]
    val_index = train_index[inner_val]

    best_alpha, best_val = None, -np.inf
    for alpha in alphas:
        started = time.time()
        model = _make_model(penalty, alpha, l1_ratio)
        model.fit(design[fit_index], editing_rate[fit_index],
                  sample_weight=coverage[fit_index])
        prediction = model.predict(design[val_index])
        score = _weighted_r2(editing_rate[val_index], prediction, coverage[val_index])
        if score > best_val:
            best_alpha, best_val = alpha, score
        if verbose:
            print(f"  alpha={alpha:<8g} val weighted_r2={score:.4f} "
                  f"({time.time() - started:.1f}s)", flush=True)

    if verbose:
        print(f"  refitting best alpha={best_alpha:g} on full train split...", flush=True)
    model = _make_model(penalty, best_alpha, l1_ratio)
    model.fit(design[train_index], editing_rate[train_index],
              sample_weight=coverage[train_index])
    test_prediction = model.predict(design[test_index])
    editable = editing_rate[test_index] > 0

    metrics = {
        "penalty": penalty,
        "selected_alpha": float(best_alpha),
        "validation_weighted_r2": float(best_val),
        "test_r2": float(r2_score(editing_rate[test_index], test_prediction)),
        "test_weighted_r2": _weighted_r2(
            editing_rate[test_index], test_prediction, coverage[test_index]
        ),
        "test_pearson": float(np.corrcoef(editing_rate[test_index], test_prediction)[0, 1]),
        "test_pearson_editable_tail": (
            float(np.corrcoef(
                editing_rate[test_index][editable], test_prediction[editable]
            )[0, 1]) if editable.sum() > 2 else float("nan")
        ),
        "n_train": int(len(train_index)),
        "n_test": int(len(test_index)),
        "n_nonzero_coef": int(np.count_nonzero(model.coef_)),
    }
    coefficients = _coefficient_table(model.coef_, blocks, top)
    return KmerRegression(
        model=model,
        blocks=blocks,
        metrics=metrics,
        coefficients=coefficients,
        parameters={
            "penalty": penalty, "alphas": tuple(alphas), "l1_ratio": l1_ratio,
            "test_size": test_size, "val_size": val_size, "seed": seed, "top": top,
        },
    )


def _coefficient_table(coef, blocks, top):
    coef = np.asarray(coef)
    nonzero = np.flatnonzero(coef)
    order = nonzero[np.argsort(-np.abs(coef[nonzero]))]
    keep = order[: 2 * top] if top else order
    rows = []
    for column in keep:
        k, position, kmer = feature_name(int(column), blocks)
        rows.append({
            "k": k, "position": position, "kmer": kmer,
            "coefficient": float(coef[column]),
        })
    table = pd.DataFrame(rows)
    if len(table):
        table = table.sort_values("coefficient", ascending=False).reset_index(drop=True)
    return table
