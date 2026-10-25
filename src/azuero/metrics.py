"""Accuracy metrics for canopy height and canopy-height change.

Height: n, RMSE, MAE, bias (pred - obs), R^2, Pearson r, and the OLS slope of obs on
pred. Intervals: empirical coverage and mean width.

Change, scored on out-of-fold predictions:
(i)   pair change: two observations from the same sensor within `radius` m and at least
      `min_dt` years apart. The observed rate (h2 - h1)/(t2 - t1) is compared with the
      predicted rate from the predictions at those two observations.
(ii)  block change: for each block and pair of consecutive years, the change in mean
      observed height is compared with the change in mean predicted height at the
      same observations (sample-matched, so footprint sampling differences cancel).
(iii) disturbance detection: pairs with an observed drop of more than `drop_m` m are
      positives; a predicted drop of more than `drop_m / 2` m counts as a detection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def height_metrics(obs, pred, w=None) -> dict:
    obs = np.asarray(obs, float)
    pred = np.asarray(pred, float)
    ok = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[ok], pred[ok]
    w = np.ones_like(obs) if w is None else np.asarray(w, float)[ok]
    if obs.size < 3:
        return {"n": int(obs.size)}
    e = pred - obs
    wm = lambda v: float(np.sum(w * v) / np.sum(w))  # noqa: E731
    ss_res = np.sum(w * e ** 2)
    ss_tot = np.sum(w * (obs - wm(obs)) ** 2)
    slope = np.polyfit(pred, obs, 1)[0] if np.ptp(pred) > 0 else np.nan
    return {"n": int(obs.size), "rmse": float(np.sqrt(wm(e ** 2))), "mae": wm(np.abs(e)),
            "bias": wm(e), "r2": float(1 - ss_res / ss_tot),
            "r": float(np.corrcoef(obs, pred)[0, 1]), "slope_obs_on_pred": float(slope),
            "obs_mean": wm(obs)}


def interval_metrics(obs, lo, hi) -> dict:
    obs, lo, hi = (np.asarray(v, float) for v in (obs, lo, hi))
    ok = np.isfinite(obs) & np.isfinite(lo) & np.isfinite(hi)
    return {"coverage": float(np.mean((obs[ok] >= lo[ok]) & (obs[ok] <= hi[ok]))),
            "width": float(np.mean(hi[ok] - lo[ok]))}


def grouped(df: pd.DataFrame, by, obs="h_cal", pred="pred") -> pd.DataFrame:
    rows = []
    for k, g in df.groupby(by, observed=True):
        m = height_metrics(g[obs], g[pred])
        m.update(dict(zip([by] if isinstance(by, str) else by, k if isinstance(k, tuple) else (k,))))
        rows.append(m)
    return pd.DataFrame(rows)


def change_pairs(df: pd.DataFrame, radius: float = 20.0, min_dt: float = 0.75,
                 max_dt: float = 8.0, same_sensor: bool = True) -> pd.DataFrame:
    """Observation pairs at (nearly) the same location from different years."""
    parts = []
    groups = df.groupby("sensor") if same_sensor else [("all", df)]
    for s, sub in groups:
        sub = sub.reset_index(drop=True)
        xy = sub[["x", "y"]].to_numpy()
        tree = cKDTree(xy)
        pairs = tree.query_pairs(radius, output_type="ndarray")
        if len(pairs) == 0:
            continue
        i, j = pairs[:, 0], pairs[:, 1]
        t = sub["t"].to_numpy()
        swap = t[i] > t[j]
        i, j = np.where(swap, j, i), np.where(swap, i, j)
        dt = t[j] - t[i]
        m = (dt >= min_dt) & (dt <= max_dt)
        i, j = i[m], j[m]
        p = pd.DataFrame({
            "sensor": s, "obs1": sub["obs_id"].to_numpy()[i], "obs2": sub["obs_id"].to_numpy()[j],
            "t1": t[i], "t2": t[j], "dt": t[j] - t[i],
            "h1": sub["h_cal"].to_numpy()[i], "h2": sub["h_cal"].to_numpy()[j],
            "p1": sub["pred"].to_numpy()[i], "p2": sub["pred"].to_numpy()[j],
            "x": xy[i, 0], "y": xy[i, 1],
        })
        parts.append(p)
    if not parts:
        return pd.DataFrame()
    p = pd.concat(parts, ignore_index=True)
    p["d_obs"] = p.h2 - p.h1
    p["d_pred"] = p.p2 - p.p1
    p["rate_obs"] = p.d_obs / p.dt
    p["rate_pred"] = p.d_pred / p.dt
    return p


def pair_change_metrics(p: pd.DataFrame, drop_m: float = 10.0) -> dict:
    if len(p) < 3:
        return {"n_pairs": int(len(p))}
    m = height_metrics(p.rate_obs, p.rate_pred)
    out = {f"rate_{k}": v for k, v in m.items()}
    md = height_metrics(p.d_obs, p.d_pred)
    out.update({f"delta_{k}": v for k, v in md.items() if k in ("rmse", "bias", "r")})
    pos = p.d_obs < -drop_m
    det = p.d_pred < -drop_m / 2
    out["drop_n"] = int(pos.sum())
    out["drop_recall"] = float((pos & det).sum() / max(pos.sum(), 1))
    out["drop_precision"] = float((pos & det).sum() / max(det.sum(), 1))
    out["n_pairs"] = int(len(p))
    return out


def block_change(df: pd.DataFrame, block_m: float = 2000.0, min_n: int = 20) -> pd.DataFrame:
    """Sample-matched change in block means between consecutive years."""
    d = df.copy()
    d["bx"] = np.floor(d.x / block_m).astype(int)
    d["by"] = np.floor(d.y / block_m).astype(int)
    agg = (d.groupby(["sensor", "bx", "by", "year"])
           .agg(n=("h_cal", "size"), obs=("h_cal", "mean"), pred=("pred", "mean"))
           .reset_index())
    agg = agg[agg.n >= min_n]
    nxt = agg.copy()
    nxt["year"] -= 1
    m = agg.merge(nxt, on=["sensor", "bx", "by", "year"], suffixes=("", "_next"))
    m["d_obs"] = m.obs_next - m.obs
    m["d_pred"] = m.pred_next - m.pred
    return m


def block_change_metrics(m: pd.DataFrame) -> dict:
    if len(m) < 3:
        return {"n_blocks": int(len(m))}
    r = height_metrics(m.d_obs, m.d_pred)
    return {"block_n": r["n"], "block_rmse": r["rmse"], "block_bias": r["bias"],
            "block_r": r["r"], "block_obs_sd": float(m.d_obs.std())}
