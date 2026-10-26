"""Sensor harmonization: map ICESat-2 PhoREAL canopy height onto the GEDI rh98 scale.

1. Near-coincident pairs: GEDI and ICESat-2 observations within `radius` m and one year.
2. Stability screening (`azuero.stability`) drops pairs with evidence of change between
   the two dates.
3. Measurement noise per sensor and class is estimated from stable same-sensor repeat
   pairs: Var(h_a - h_b) = 2 sigma^2 for two observations of the same class.
4. Per ICESat-2 class (day/night x strong/weak beam), a Deming regression
   h_gedi = a + b * h_is2 is fit with the error-variance ratio from step 3.
   Ordinary least squares would be biased by the noise in h_is2.

The output is used to put all labels on a common (GEDI rh98) scale, `h_cal`. The noise
estimates provide per-observation weights and label-noise floors for the metrics.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .config import data_path, results_path
from .labels import load
from .stability import screen_pairs

log = logging.getLogger(__name__)

CLASS_COLS = ["night", "strong"]


def find_pairs(a: pd.DataFrame, b: pd.DataFrame, radius: float, max_dt: float,
               same: bool = False, k: int = 16) -> pd.DataFrame:
    tree = cKDTree(b[["x", "y"]].to_numpy())
    d, j = tree.query(a[["x", "y"]].to_numpy(), k=k, distance_upper_bound=radius)
    parts = []
    ta, tb = a["t"].to_numpy(), b["t"].to_numpy()
    for kk in range(k):
        ok = np.isfinite(d[:, kk])
        ia = np.nonzero(ok)[0]
        jb = j[ok, kk]
        dt = np.abs(ta[ia] - tb[jb])
        m = dt <= max_dt
        if same:  # same table: keep each unordered pair once, drop self/same-track
            m &= (a["obs_id"].to_numpy()[ia] < b["obs_id"].to_numpy()[jb]) & (dt > 1 / 365)
        parts.append(pd.DataFrame({"ia": ia[m], "ib": jb[m], "dist": d[ok, kk][m], "dt": dt[m]}))
    p = pd.concat(parts, ignore_index=True)
    out = pd.DataFrame({
        "obs_a": a["obs_id"].to_numpy()[p.ia], "obs_b": b["obs_id"].to_numpy()[p.ib],
        "h_a": a["h"].to_numpy()[p.ia], "h_b": b["h"].to_numpy()[p.ib],
        "t1": ta[p.ia], "t2": tb[p.ib], "dist": p.dist, "dt": p.dt,
        "row": a["row_hls30"].to_numpy()[p.ia], "col": a["col_hls30"].to_numpy()[p.ia],
        "row20": a["row_aef20"].to_numpy()[p.ia], "col20": a["col_aef20"].to_numpy()[p.ia],
    })
    for c in CLASS_COLS:
        out[f"{c}_a"] = a[c].to_numpy()[p.ia]
        out[f"{c}_b"] = b[c].to_numpy()[p.ib]
    return out


def deming(x: np.ndarray, y: np.ndarray, lam: float) -> tuple[float, float]:
    """Deming regression y = a + b x with error-variance ratio lam = var(ey)/var(ex)."""
    mx, my = x.mean(), y.mean()
    sxx = np.mean((x - mx) ** 2)
    syy = np.mean((y - my) ** 2)
    sxy = np.mean((x - mx) * (y - my))
    b = (syy - lam * sxx + np.sqrt((syy - lam * sxx) ** 2 + 4 * lam * sxy ** 2)) / (2 * sxy)
    return my - b * mx, b


def robust_sigma(d: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(d - np.median(d))))


def run(radius: float = 20.0, max_dt: float = 1.0, n_boot: int = 200, suffix: str = "") -> pd.DataFrame:
    """Fit the calibration. A non-empty `suffix` writes to separate files (for comparisons)
    without replacing the tables that `apply` uses."""
    df = load()
    g = df[df.sensor == "gedi"].reset_index(drop=True)
    i = df[df.sensor == "icesat2"].reset_index(drop=True)

    cross = screen_pairs(find_pairs(i, g, radius, max_dt))
    cross.to_parquet(data_path("labels", f"pairs_cross{suffix}.parquet"))

    # Same-sensor repeat pairs (stable) for noise estimates.
    noise_rows = []
    for name, sub in (("gedi", g), ("icesat2", i)):
        rp = screen_pairs(find_pairs(sub, sub, radius, max_dt, same=True))
        rp.to_parquet(data_path("labels", f"pairs_repeat_{name}{suffix}.parquet"))
        st = rp[rp.stable]
        for key, grp in st.groupby([f"{c}_a" for c in CLASS_COLS] + [f"{c}_b" for c in CLASS_COLS]):
            na, sa, nb, sb = key
            if (na, sa) != (nb, sb) or len(grp) < 30:
                continue
            noise_rows.append({"sensor": name, "night": na, "strong": sa, "n_pairs": len(grp),
                               "sigma": robust_sigma(grp.h_a - grp.h_b) / np.sqrt(2)})
    noise = pd.DataFrame(noise_rows)
    noise.to_csv(results_path("labels", f"noise_by_class{suffix}.csv"), index=False)
    sig_g = float(np.average(noise.query("sensor == 'gedi'").sigma,
                             weights=noise.query("sensor == 'gedi'").n_pairs))

    rng = np.random.default_rng(0)
    rows = []
    for screened, sub in (("all_pairs", cross), ("stable_pairs", cross[cross.stable])):
        for (night, strong), grp in sub.groupby(["night_a", "strong_a"]):
            nrow = noise.query("sensor == 'icesat2' and night == @night and strong == @strong")
            sig_i = float(nrow.sigma.iloc[0]) if len(nrow) else robust_sigma(grp.h_a - grp.h_b)
            lam = sig_g ** 2 / sig_i ** 2  # var(e_gedi) / var(e_is2)
            x, y = grp.h_a.to_numpy(), grp.h_b.to_numpy()
            a, b = deming(x, y, lam)
            boots = []
            for _ in range(n_boot):
                k = rng.integers(0, len(x), len(x))
                boots.append(deming(x[k], y[k], lam))
            boots = np.array(boots)
            rows.append({"pairs": screened, "night": night, "strong": strong, "n": len(grp),
                         "a": a, "b": b, "a_se": boots[:, 0].std(), "b_se": boots[:, 1].std(),
                         "sigma_is2": sig_i, "sigma_gedi": sig_g,
                         "median_diff_raw": float(np.median(x - y)),
                         "r_raw": float(np.corrcoef(x, y)[0, 1])})
    cal = pd.DataFrame(rows)
    cal.to_csv(results_path("labels", f"calibration{suffix}.csv"), index=False)
    # Quantile mapping per class (stable pairs): monotone map matching the marginal
    # distributions of collocated ICESat-2 and GEDI heights.
    qs = np.linspace(0.01, 0.99, 99)
    qrows = []
    for (night, strong), grp in cross[cross.stable].groupby(["night_a", "strong_a"]):
        qi = np.quantile(grp.h_a, qs)
        qg = np.quantile(grp.h_b, qs)
        for q_, a_, b_ in zip(qs, qi, qg):
            qrows.append({"night": night, "strong": strong, "q": q_, "h_is2": a_, "h_gedi": b_})
    pd.DataFrame(qrows).to_csv(results_path("labels", f"quantile_map{suffix}.csv"), index=False)
    screen = cross[["flag_aef", "flag_gfc", "flag_hls", "flag_s1", "stable"]].mean()
    screen["n_pairs"] = len(cross)
    screen.to_frame("value").to_csv(results_path("labels", f"pair_screening{suffix}.csv"))
    log.info("calibration:\n%s", cal.round(3))
    return cal


METHODS = ("none", "offset", "qmap", "deming")


def apply(df: pd.DataFrame, method: str = "qmap", which: str = "stable_pairs") -> pd.DataFrame:
    """Add `h_cal` (GEDI-equivalent height) and `sigma` (label noise) columns.

    method: none (raw heights), offset (median difference per class), qmap (quantile
    mapping per class) or deming (linear errors-in-variables fit per class).
    """
    cal = pd.read_csv(results_path("labels", "calibration.csv")).query("pairs == @which")
    noise = pd.read_csv(results_path("labels", "noise_by_class.csv"))
    qmap = pd.read_csv(results_path("labels", "quantile_map.csv"))
    df = df.copy()
    df["h_cal"] = df["h"].astype("float64")
    df["sigma"] = np.nan
    for r in cal.itertuples():
        m = ((df.sensor == "icesat2") & (df.night == r.night) & (df.strong == r.strong)).to_numpy()
        h = df.loc[m, "h"].to_numpy(float)
        slope = 1.0
        if method == "offset":
            hc = h - r.median_diff_raw
        elif method == "deming":
            hc = r.a + r.b * h
            slope = abs(r.b)
        elif method == "qmap":
            t = qmap.query("night == @r.night and strong == @r.strong").sort_values("q")
            xi, yg = t.h_is2.to_numpy(), t.h_gedi.to_numpy()
            hc = np.interp(h, xi, yg)
            # Linear extrapolation with unit slope beyond the mapped range.
            hc = np.where(h > xi[-1], yg[-1] + (h - xi[-1]), hc)
            hc = np.where(h < xi[0], yg[0] + (h - xi[0]), hc)
            slope = float(np.median(np.diff(yg[10:90]) / np.maximum(np.diff(xi[10:90]), 1e-6)))
        else:
            hc = h
        df.loc[m, "h_cal"] = np.clip(hc, 0, None)
        # Noise on the GEDI scale.
        df.loc[m, "sigma"] = slope * r.sigma_is2
    for r in noise.query("sensor == 'gedi'").itertuples():
        m = (df.sensor == "gedi") & (df.night == r.night) & (df.strong == r.strong)
        df.loc[m, "sigma"] = r.sigma
    gsig = noise.query("sensor == 'gedi'").sigma.mean()
    df["sigma"] = df["sigma"].fillna(gsig).clip(lower=1.0)
    return df


def cli(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(prog="azuero calibrate")
    p.add_argument("--radius", type=float, default=20.0)
    a = p.parse_args(argv)
    print(run(radius=a.radius).round(3).to_string())
