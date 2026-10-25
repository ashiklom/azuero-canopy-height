"""Calibrated predictive intervals via conformalized quantile regression (CQR).

The model's raw 5 %/95 % quantiles (lo, hi) are widened or narrowed by a conformity
offset q_hat. q_hat is estimated from out-of-fold residuals, separately per bin of
predicted height (Mondrian CQR), because errors grow with height:

    s_i = max(lo_i - y_i, y_i - hi_i),   q_hat = Quantile_{(1-alpha)(1+1/n)}(s)
    interval = [lo - q_hat, hi + q_hat]

To report honest coverage, offsets are cross-fitted: the coverage reported for spatial
fold k uses offsets estimated from the other folds. Final offsets (from all OOF
predictions) are saved for the product.

The interval targets a GEDI-equivalent rh98 measurement at the cell, so it includes
lidar measurement noise. A "true height" SD is also derived by removing the GEDI noise
variance estimated from repeat pairs: sd_true = sqrt(max(sd_total^2 - sigma_gedi^2, 0)).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import load_config, results_dir, results_path

PRED_BINS = [-np.inf, 3, 6, 10, 15, 20, 25, np.inf]


def _offset(s: np.ndarray, alpha: float) -> float:
    n = s.size
    if n == 0:
        return 0.0
    k = min(1.0, (1 - alpha) * (1 + 1 / n))
    return float(np.quantile(s, k))


def fit_offsets(y, pred, lo, hi, alpha: float) -> pd.DataFrame:
    b = pd.cut(pred, PRED_BINS, labels=False)
    s = np.maximum(lo - y, y - hi)
    rows = []
    for k in range(len(PRED_BINS) - 1):
        m = b == k
        rows.append({"bin": k, "lo_edge": PRED_BINS[k], "hi_edge": PRED_BINS[k + 1],
                     "n": int(m.sum()), "q_hat": _offset(s[m], alpha)})
    return pd.DataFrame(rows)


def apply_offsets(pred, lo, hi, offsets: pd.DataFrame):
    b = pd.cut(pred, PRED_BINS, labels=False)
    q = offsets.set_index("bin").q_hat.reindex(range(len(PRED_BINS) - 1)).fillna(0).to_numpy()
    qq = q[np.nan_to_num(b, nan=0).astype(int)]
    lo2 = np.maximum(lo - qq, 0)
    hi2 = hi + qq
    return lo2, hi2


def interval_to_sd(lo, hi, level: float):
    z = norm.ppf(0.5 + level / 2)
    return (hi - lo) / (2 * z)


def calibrate(oof: pd.DataFrame, name: str) -> dict:
    """Cross-fitted CQR evaluation + final offsets for an experiment's OOF table."""
    level = load_config()["model"]["interval_level"]
    alpha = 1 - level
    y, pred, lo, hi = (oof[c].to_numpy(float) for c in ("h_cal", "pred", "lo", "hi"))
    lo2 = np.full_like(lo, np.nan)
    hi2 = np.full_like(hi, np.nan)
    for f in oof.fold.unique():
        te = (oof.fold == f).to_numpy()
        off = fit_offsets(y[~te], pred[~te], lo[~te], hi[~te], alpha)
        lo2[te], hi2[te] = apply_offsets(pred[te], lo[te], hi[te], off)
    final = fit_offsets(y, pred, lo, hi, alpha)
    out_dir = results_dir("uncertainty", name)
    final.to_csv(out_dir / "cqr_offsets.csv", index=False)
    cov_raw = np.mean((y >= lo) & (y <= hi))
    cov_cal = np.mean((y >= lo2) & (y <= hi2))
    hb = pd.cut(oof.h_cal, [0, 5, 10, 15, 20, 25, 30, 80], include_lowest=True).astype(str)
    by = (pd.DataFrame({"hbin": hb, "raw": (y >= lo) & (y <= hi),
                        "cal": (y >= lo2) & (y <= hi2), "width_raw": hi - lo,
                        "width_cal": hi2 - lo2, "sensor": oof.sensor})
          .groupby(["sensor", "hbin"]).mean().reset_index())
    by.to_csv(out_dir / "coverage_by_class.csv", index=False)
    # PIT-style reliability: coverage at several nominal levels using a Gaussian
    # approximation centred on pred with the calibrated SD.
    sd = interval_to_sd(lo2, hi2, level)
    rel = []
    for lvl in (0.5, 0.68, 0.8, 0.9, 0.95):
        z = norm.ppf(0.5 + lvl / 2)
        rel.append({"nominal": lvl, "empirical": float(np.mean(np.abs(y - pred) <= z * sd))})
    pd.DataFrame(rel).to_csv(out_dir / "reliability.csv", index=False)
    summ = {"level": level, "coverage_raw": float(cov_raw), "coverage_crossfit": float(cov_cal),
            "width_raw": float(np.mean(hi - lo)), "width_crossfit": float(np.mean(hi2 - lo2))}
    (out_dir / "summary.json").write_text(json.dumps(summ, indent=2))
    return summ
