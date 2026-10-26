"""Data-driven observation operators for the monthly state-space model.

For each sensor family (optical = HLS monthly composite; radar = Sentinel-1 monthly
composite) a LightGBM regressor maps that month's per-pixel features (+ static terrain)
to lidar canopy height. It is trained on lidar observations matched to the composite of
the observation's own month. Its errors have two parts:
  * a persistent per-pixel bias (the same pixel looks "too green" every month), and
  * month-to-month noise (residual cloud, phenology, soil moisture, viewing geometry).
The two are separated using pairs of out-of-fold residuals at the same location in
different months: cov(e1, e2) is approximately the bias variance, and the remainder of
the residual variance, after removing lidar noise, is the monthly noise. These set the
prior SD of the bias state and the observation variance in `ssm.kalman`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ..config import data_path
from ..models import get_model
from ..sample import band_names, sample_cells

log = logging.getLogger(__name__)

FAMILIES = {
    "optical": {"file": ("hls", "hls_monthly_{y}.tif"),
                "vars": ["RED", "NIR", "SWIR1", "SWIR2", "NDVI", "NBR", "NDMI", "count"],
                "scale": 1e-4, "fill": -9999},
    "radar": {"file": ("s1", "s1_monthly_{y}.tif"), "vars": ["vv_db", "vh_db", "count"],
              "scale": 0.01, "fill": -32768},
}


def monthly_file(family: str, year: int):
    sub, pat = FAMILIES[family]["file"]
    return data_path("interim", sub, "hls30", pat.format(y=year), mkdir=False)


def sample_monthly(family: str, year: int, rows, cols, months) -> np.ndarray:
    """Features (n, nvars) for each cell at its month (1-12); NaN where missing."""
    spec = FAMILIES[family]
    f = monthly_file(family, year)
    nv = len(spec["vars"])
    out = np.full((len(rows), nv), np.nan, np.float32)
    if not f.exists():
        return out
    names = band_names(f)
    months = np.asarray(months)
    for m in np.unique(months):
        sel = months == m
        idx = [names.index(f"{v}_m{m:02d}") + 1 for v in spec["vars"]]
        raw = sample_cells(f, np.asarray(rows)[sel], np.asarray(cols)[sel], bands=idx,
                           out_dtype=np.float32)
        raw[raw == spec["fill"]] = np.nan
        vals = raw.copy()
        vals[:, :-1] *= spec["scale"]
        vals[:, -1] = raw[:, -1]  # count is unscaled
        vals[np.nan_to_num(vals[:, -1]) <= 0] = np.nan
        out[sel] = vals
    return out


def sample_all_months(family: str, years, rows, cols) -> np.ndarray:
    """(n_cells, 12 * len(years), nvars) monthly features for a fixed set of cells."""
    spec = FAMILIES[family]
    nv = len(spec["vars"])
    out = np.full((len(rows), 12 * len(years), nv), np.nan, np.float32)
    for yi, y in enumerate(years):
        f = monthly_file(family, y)
        if not f.exists():
            continue
        names = band_names(f)
        idx = [names.index(f"{v}_m{m:02d}") + 1 for m in range(1, 13) for v in spec["vars"]]
        raw = sample_cells(f, rows, cols, bands=idx).reshape(len(rows), 12, nv)
        raw[raw == spec["fill"]] = np.nan
        vals = raw.copy()
        vals[..., :-1] *= spec["scale"]
        vals[np.nan_to_num(vals[..., -1]) <= 0] = np.nan
        out[:, yi * 12:(yi + 1) * 12] = vals
    return out


@dataclass
class Operator:
    family: str
    model: object
    var_noise: float      # month-to-month error variance (m^2), lidar noise removed
    bias_sd: float        # SD of the persistent per-pixel bias (m)

    def predict(self, feats: np.ndarray, static: np.ndarray) -> np.ndarray:
        """feats: (..., nvars), static: (..., nstatic) broadcastable. Predictions are made
        on the product's measurement standard (GEDI power beam at night), like v1."""
        shp = feats.shape[:-1]
        n = int(np.prod(shp))
        X = np.concatenate([feats.reshape(-1, feats.shape[-1]),
                            np.broadcast_to(static, shp + static.shape[-1:]).reshape(
                                -1, static.shape[-1]),
                            np.tile(STANDARD, (n, 1))], axis=1)
        ok = np.isfinite(X[:, 0])
        out = np.full(X.shape[0], np.nan, np.float32)
        if ok.any():
            out[ok] = self.model.predict(X[ok])
        return out.reshape(shp)


# Observation covariates (is_icesat2, night, strong) and the standard used for prediction.
STANDARD = np.array([[0.0, 1.0, 1.0]], np.float32)


def training_rows(labels: pd.DataFrame, family: str, static: np.ndarray):
    """Features for lidar labels at their own month, plus observation covariates."""
    months = labels["time"].dt.month.to_numpy()
    feats = np.full((len(labels), len(FAMILIES[family]["vars"])), np.nan, np.float32)
    for y, idx in labels.groupby("year").indices.items():
        sub = labels.iloc[idx]
        feats[idx] = sample_monthly(family, int(y), sub.row_hls30.to_numpy(),
                                    sub.col_hls30.to_numpy(), months[idx])
    cov = np.column_stack([(labels.sensor == "icesat2").to_numpy(np.float32),
                           labels.night.to_numpy(np.float32), labels.strong.to_numpy(np.float32)])
    X = np.concatenate([feats, static, cov], axis=1)
    return X


def fit(labels: pd.DataFrame, family: str, static: np.ndarray, sigma_label: np.ndarray,
        folds=None, max_rows: int = 300_000, seed: int = 0) -> Operator:
    """Fit on all given labels; estimate the error structure from inner spatial folds."""
    X = training_rows(labels, family, static)
    y = labels["h_cal"].to_numpy(np.float32)
    ok = np.isfinite(X[:, 0])
    rng = np.random.default_rng(seed)
    params = dict(quantiles=(), n_estimators=400)
    # Inner out-of-fold residuals for the error decomposition.
    if folds is None:
        from ..cv import spatial_folds
        folds = spatial_folds(labels.reset_index(drop=True), n_folds=3, regions_per_fold=3)
    resid = np.full(len(y), np.nan)
    for f in folds:
        tr = np.nonzero(f.train & ok)[0]
        if tr.size > max_rows:
            tr = rng.choice(tr, max_rows, replace=False)
        m = get_model("lightgbm", **params).fit(X[tr], y[tr])
        te = np.nonzero(f.test & ok)[0]
        resid[te] = m.predict(X[te]) - y[te]
    var_total = np.nanvar(resid)
    # Bias variance from residual pairs at the same place in different months.
    good = np.isfinite(resid)
    xy = labels[["x", "y"]].to_numpy()[good]
    rr = resid[good]
    t = labels["t"].to_numpy()[good]
    pairs = cKDTree(xy).query_pairs(20.0, output_type="ndarray")
    if len(pairs):
        dtp = np.abs(t[pairs[:, 0]] - t[pairs[:, 1]])
        pairs = pairs[(dtp > 1 / 12) & (dtp < 1.0)]
    bias_var = float(np.cov(rr[pairs[:, 0]], rr[pairs[:, 1]])[0, 1]) if len(pairs) > 50 else 0.5 * var_total
    bias_var = float(np.clip(bias_var, 0.0, var_total))
    label_var = float(np.nanmean(sigma_label[good] ** 2))
    var_noise = float(max(var_total - bias_var - label_var, 1.0))
    tr = np.nonzero(ok)[0]
    if tr.size > max_rows:
        tr = rng.choice(tr, max_rows, replace=False)
    model = get_model("lightgbm", **params).fit(X[tr], y[tr])
    log.info("operator %s: n=%d var_total=%.1f bias_sd=%.2f noise_sd=%.2f pairs=%d", family,
             ok.sum(), var_total, np.sqrt(bias_var), np.sqrt(var_noise), len(pairs))
    return Operator(family, model, var_noise, float(np.sqrt(bias_var)))
