"""Derived AVIRIS-3 features on the analysis grids (AVUELO, February 2025).

From the binned 30 nm spectra (`io_airborne`):
* the first 10 principal components of brightness-normalized reflectance
  (PCA fit on a random sample of covered cells, saved to results/airborne/pca.json);
* five spectral indices: NDVI, NDRE (red edge), a broad-band PRI proxy, NDWI
  (860/1240 nm, canopy water) and CAI (cellulose absorption, 2000-2200 nm, dry
  matter / non-photosynthetic vegetation).
Features are defined only for year 2025 and only where flown; elsewhere they are NaN.
"""

from __future__ import annotations

import json
from functools import lru_cache

import numpy as np
import pandas as pd
from rasterio.windows import Window

from .config import data_path, results_path
from .io_airborne import bin_names

YEARS = (2025,)
N_PC = 10


def feature_names() -> list[str]:
    return [f"av3_pc{i + 1}" for i in range(N_PC)] + [
        "av3_ndvi", "av3_ndre", "av3_pri", "av3_ndwi", "av3_cai"]


def _b(df: pd.DataFrame, lo: int) -> np.ndarray:
    return df[f"av3_{lo}_{lo + 30}"].to_numpy(np.float64)


def _indices(df: pd.DataFrame) -> np.ndarray:
    nir, red = _b(df, 790), _b(df, 640)
    re1 = _b(df, 700)
    g1, g2 = _b(df, 520), _b(df, 550)
    swir = _b(df, 1240)
    with np.errstate(all="ignore"):
        ndvi = (nir - red) / (nir + red)
        ndre = (nir - re1) / (nir + re1)
        pri = (g1 - g2) / (g1 + g2)
        ndwi = (_b(df, 850) - swir) / (_b(df, 850) + swir)
        cai = 0.5 * (_b(df, 2020) + _b(df, 2200)) - _b(df, 2110)
    return np.column_stack([ndvi, ndre, pri, ndwi, cai])


@lru_cache(maxsize=4)
def table(grid: str) -> pd.DataFrame:
    """Feature table indexed by (row, col) for one grid."""
    df = pd.read_parquet(data_path("interim", "airborne", f"av3_{grid}.parquet"))
    spec = df[bin_names()].to_numpy(np.float64)
    ok = np.isfinite(spec).all(1) & (spec.mean(1) > 0.005)
    df, spec = df[ok], spec[ok]
    norm = spec / np.linalg.norm(spec, axis=1, keepdims=True)
    pca_f = results_path("airborne", "pca.json")
    if pca_f.exists():
        p = json.loads(pca_f.read_text())
        mean, comps = np.array(p["mean"]), np.array(p["components"])
    else:
        rng = np.random.default_rng(0)
        s = norm[rng.choice(len(norm), min(200000, len(norm)), replace=False)]
        mean = s.mean(0)
        _, sv, vt = np.linalg.svd(s - mean, full_matrices=False)
        comps = vt[:N_PC]
        pca_f.write_text(json.dumps({"mean": mean.tolist(), "components": comps.tolist(),
                                     "explained": (sv[:N_PC] ** 2 / (sv ** 2).sum()).tolist(),
                                     "fit_grid": grid}))
    pcs = (norm - mean) @ comps.T
    feats = np.column_stack([pcs, _indices(df)]).astype(np.float32)
    out = pd.DataFrame(feats, columns=feature_names())
    out["row"] = df["row"].to_numpy()
    out["col"] = df["col"].to_numpy()
    return out.set_index(["row", "col"])


def sample(grid: str, year: int, rows, cols) -> np.ndarray:
    n = len(rows)
    out = np.full((n, len(feature_names())), np.nan, np.float32)
    if year not in YEARS:
        return out
    t = table(grid)
    idx = pd.MultiIndex.from_arrays([np.asarray(rows), np.asarray(cols)])
    hit = idx.isin(t.index)
    if hit.any():
        out[hit] = t.loc[idx[hit]].to_numpy(np.float32)
    return out


def stack_window(grid: str, year: int, win: Window) -> np.ndarray:
    h, w = int(win.height), int(win.width)
    out = np.full((len(feature_names()), h, w), np.nan, np.float32)
    if year not in YEARS:
        return out
    t = table(grid).reset_index()
    m = (t.row >= win.row_off) & (t.row < win.row_off + h) & (t.col >= win.col_off) & \
        (t.col < win.col_off + w)
    sub = t[m]
    if len(sub):
        out[:, sub.row - int(win.row_off), sub.col - int(win.col_off)] = \
            sub[feature_names()].to_numpy(np.float32).T
    return out


def coverage_mask(grid: str, rows, cols) -> np.ndarray:
    t = table(grid)
    return pd.MultiIndex.from_arrays([np.asarray(rows), np.asarray(cols)]).isin(t.index)


def compare(base: str = "base_aef_hls30", with_av3: str = "aef_av3", grid: str = "hls30"):
    """Score two CV experiments on the same held-out labels: year 2025 and inside the
    AVIRIS-3 flight coverage (Feb 2025 flights; lidar from any time in 2025)."""
    from .config import results_dir
    from .metrics import height_metrics

    a = pd.read_parquet(data_path("cv", base, "oof.parquet"))
    b = pd.read_parquet(data_path("cv", with_av3, "oof.parquet"))
    m = a.merge(b[["obs_id", "pred"]], on="obs_id", suffixes=("_base", "_av3"))
    m = m[m.year.isin(YEARS)]
    cov = coverage_mask(grid, m[f"row_{grid}"].to_numpy(), m[f"col_{grid}"].to_numpy())
    rows = []
    for subset, d in (("2025, inside AVIRIS-3 coverage", m[cov]),
                      ("2025, outside coverage", m[~cov])):
        for model, col in (("AlphaEarth", "pred_base"), ("AlphaEarth + AVIRIS-3", "pred_av3")):
            for sensor, dd in [("all", d)] + list(d.groupby("sensor")):
                rows.append({"subset": subset, "model": model, "sensor": sensor,
                             **height_metrics(dd.h_cal, dd[col])})
    out = pd.DataFrame(rows)
    out.to_csv(results_dir("cv") / "av3_compare.csv", index=False)
    return out
