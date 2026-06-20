"""
Statistical-rigor helpers: deterministic seeding and bootstrap confidence intervals.

Used by the seeds/significance notebooks to report results as mean ± std across seeds
with bootstrap 95% CIs (instead of single-run point estimates).
"""

from __future__ import annotations

import random

import numpy as np
import torch
from sklearn.metrics import f1_score


def set_seed(seed: int) -> None:
    """Seed python / numpy / torch (incl. CUDA) for repeatable runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "weighted",
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """Percentile bootstrap CI for the F1 score over test examples.

    Resamples (y_true, y_pred) pairs with replacement `n_boot` times and returns the
    point estimate plus the [alpha/2, 1-alpha/2] percentile interval.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    point = f1_score(y_true, y_pred, average=average, zero_division=0)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[b] = f1_score(y_true[idx], y_pred[idx], average=average, zero_division=0)
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "point": float(point),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "average": average,
        "n_boot": n_boot,
    }


def seed_summary(values: list[float]) -> dict:
    """Mean ± std (population-ish, ddof=1) over per-seed metric values."""
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "n": int(len(arr)),
        "values": [round(float(v), 4) for v in arr],
    }
