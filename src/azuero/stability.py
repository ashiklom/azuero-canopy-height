"""Screening for height-changing events between two lidar observations of one place.

For a pair of observations at the same grid cell with dates t1 and t2, we look for
evidence of disturbance or rapid change over the years they span. The evidence comes
from data that are independent of the lidar:

* AlphaEarth: the minimum cosine similarity between consecutive annual embeddings over
  [year(t1), year(t2)]. The threshold is a low percentile of the same statistic in
  stable reference forest (GFC tree cover >= 90 % in 2000, no recorded loss, ETH height
  >= 20 m).
* Hansen GFC: any recorded loss in a year within [year(t1), year(t2)].
* HLS: the largest change in annual median NBR between consecutive years over the span.
* Sentinel-1: the largest change in annual median VH backscatter (dB) over the span.

A pair is "stable" when no available indicator crosses its threshold. Pairs within a
single calendar year have no annual AlphaEarth/HLS/S1 contrast. Their only check is GFC
loss in that year.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
import pandas as pd

from .config import data_path, load_config, results_path
from .sample import band_names, sample_cells

log = logging.getLogger(__name__)

THRESHOLDS = {"hls_dnbr": 0.10, "s1_dvh_db": 1.5}
GRID = "hls30"


def _aef_path(year):
    return data_path("interim", "alphaearth", GRID, f"aef_{year}.tif", mkdir=False)


def aef_consecutive_similarity(rows, cols, years) -> dict[int, np.ndarray]:
    """cos(emb[y], emb[y+1]) at the cells, for each y where both years exist."""
    from .io_alphaearth import dequantize

    emb = {}
    for y in years:
        f = _aef_path(y)
        if f.exists():
            emb[y] = dequantize(sample_cells(f, rows, cols, out_dtype=np.int16).astype(np.int8))
    sims = {}
    for y in years:
        if y in emb and y + 1 in emb:
            a, b = emb[y], emb[y + 1]
            num = np.nansum(a * b, 1)
            den = np.sqrt(np.nansum(a * a, 1) * np.nansum(b * b, 1))
            with np.errstate(all="ignore"):
                s = num / den
            s[~np.isfinite(a).all(1) | ~np.isfinite(b).all(1)] = np.nan
            sims[y] = s
    return sims


def annual_band(rows, cols, years, kind: str, band: str) -> dict[int, np.ndarray]:
    out = {}
    for y in years:
        if kind == "hls":
            f = data_path("interim", "hls", GRID, f"hls_annual_{y}.tif", mkdir=False)
        else:
            f = data_path("interim", "s1", GRID, f"s1_annual_{y}.tif", mkdir=False)
        if f.exists():
            i = band_names(f).index(band) + 1
            out[y] = sample_cells(f, rows, cols, bands=[i])[:, 0]
    return out


@lru_cache(maxsize=1)
def reference_threshold(pct: float = 2.0) -> float:
    """Low percentile of consecutive-year AlphaEarth similarity in stable forest."""
    import rasterio

    with rasterio.open(data_path("interim", "reference", f"gfc_{GRID}.tif")) as s:
        gfc = s.read()
    with rasterio.open(data_path("interim", "reference", f"eth_chm_{GRID}.tif")) as s:
        eth = s.read(1)
    ref = (gfc[1] >= 90) & ~(gfc[0] > 0) & (eth >= 20)
    r, c = np.nonzero(ref)
    rng = np.random.default_rng(0)
    pick = rng.choice(r.size, size=min(20000, r.size), replace=False)
    years = load_config()["years"]
    sims = aef_consecutive_similarity(r[pick], c[pick], years)
    allv = np.concatenate([v[np.isfinite(v)] for v in sims.values()])
    thr = float(np.percentile(allv, pct))
    pd.DataFrame({"n_ref_cells": [pick.size], "pct": [pct], "threshold": [thr],
                  "median": [float(np.median(allv))]}).to_csv(
        results_path("labels", "aef_stability_threshold.csv"), index=False)
    log.info("AlphaEarth stability threshold (p%s of stable forest): %.4f", pct, thr)
    return thr


def screen_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    """Add change indicators and a `stable` flag to a table with columns
    row, col, t1, t2 (decimal years)."""
    rows = pairs["row"].to_numpy()
    cols = pairs["col"].to_numpy()
    y1 = np.floor(np.minimum(pairs["t1"], pairs["t2"])).astype(int).to_numpy()
    y2 = np.floor(np.maximum(pairs["t1"], pairs["t2"])).astype(int).to_numpy()
    years = list(range(y1.min(), y2.max() + 1))
    out = pairs.copy()

    sims = aef_consecutive_similarity(rows, cols, years)
    aef_min = np.full(len(pairs), np.nan)
    for y, s in sims.items():
        m = (y >= y1) & (y + 1 <= y2)
        aef_min[m] = np.fmin(aef_min[m], s[m])
    out["aef_min_cos"] = aef_min

    loss = sample_cells(data_path("interim", "reference", f"gfc_{GRID}.tif"), rows, cols,
                        bands=[1])[:, 0]
    lossyear = np.where(loss > 0, 2000 + loss, np.nan)
    out["gfc_lossyear"] = lossyear
    out["gfc_loss_in_span"] = (lossyear >= y1) & (lossyear <= y2)

    for kind, band, name in (("hls", "NBR_p50", "hls_dnbr"), ("s1", "vh_p50", "s1_dvh_db")):
        vals = annual_band(rows, cols, years, kind, band)
        dmax = np.full(len(pairs), np.nan)
        for y in years:
            if y in vals and y + 1 in vals:
                d = np.abs(vals[y + 1] - vals[y])
                m = (y >= y1) & (y + 1 <= y2)
                dmax[m] = np.fmax(dmax[m], d[m])
        out[name] = dmax

    thr = reference_threshold()
    out["flag_aef"] = out["aef_min_cos"] < thr
    out["flag_gfc"] = out["gfc_loss_in_span"]
    out["flag_hls"] = out["hls_dnbr"] > THRESHOLDS["hls_dnbr"]
    out["flag_s1"] = out["s1_dvh_db"] > THRESHOLDS["s1_dvh_db"]
    out["stable"] = ~(out[["flag_aef", "flag_gfc", "flag_hls", "flag_s1"]].any(axis=1))
    return out
