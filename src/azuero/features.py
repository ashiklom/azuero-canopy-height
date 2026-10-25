"""Predictor sets, training-table extraction and inference stacks.

Feature sets (each optional, selected per experiment):
* aef   - AlphaEarth 64-d annual embedding for the observation year (de-quantized).
* aefctx - the embedding averaged over a ~90-100 m neighbourhood (spatial context).
* dem   - Copernicus GLO-30 terrain (static).
* hls   - HLS annual summaries for the observation year.
* s1    - Sentinel-1 annual summaries for the observation year.
* av3   - AVIRIS-3 (Feb 2025) spectral PCs and indices; only where flown.
* time  - decimal year (lets the model express trends not captured by covariates).

Training rows are lidar observations; features are sampled at the observation's grid cell
for the observation's calendar year. Samples are cached per (grid, set, year).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from .config import data_path, load_config
from .sample import band_names, sample_cells

log = logging.getLogger(__name__)

ALL_SETS = ["aef", "aefctx", "dem", "hls", "s1", "av3", "time"]


def raster_for(set_name: str, grid: str, year: int) -> Path | None:
    if set_name == "aef":
        return data_path("interim", "alphaearth", grid, f"aef_{year}.tif", mkdir=False)
    if set_name == "aefctx":
        return data_path("interim", "alphaearth", grid, f"aefctx_{year}.tif", mkdir=False)
    if set_name == "dem":
        return data_path("interim", "dem", f"terrain_{grid}.tif", mkdir=False)
    if set_name == "hls":
        return data_path("interim", "hls", "hls30", f"hls_annual_{year}.tif", mkdir=False)
    if set_name == "s1":
        return data_path("interim", "s1", "hls30", f"s1_annual_{year}.tif", mkdir=False)
    return None


def names_for(set_name: str, grid: str) -> list[str]:
    if set_name == "aef":
        return [f"A{i:02d}" for i in range(64)]
    if set_name == "aefctx":
        return [f"C{i:02d}" for i in range(64)]
    if set_name == "time":
        return ["t"]
    if set_name == "av3":
        from .av3_features import feature_names
        return feature_names()
    for y in load_config()["years"]:
        f = raster_for(set_name, grid, y)
        if f is not None and f.exists():
            return [f"{set_name}_{n}" for n in band_names(f)]
    raise FileNotFoundError(f"no rasters for feature set {set_name}")


def _cells_on(grid: str, src_grid: str, rows, cols):
    """Map cells of `grid` to the containing cell of `src_grid` (both aligned)."""
    if grid == src_grid:
        return rows, cols
    from .grid import get_grid

    g, s = get_grid(grid), get_grid(src_grid)
    x, y = g.xy(rows, cols)
    return s.rowcol(x, y)


def sample_set(set_name: str, grid: str, year: int, rows, cols) -> np.ndarray:
    if set_name in ("aef", "aefctx"):
        from .io_alphaearth import dequantize

        f = raster_for(set_name, grid, year)
        raw = sample_cells(f, rows, cols, out_dtype=np.int16)
        return dequantize(raw.astype(np.int8))
    if set_name in ("hls", "s1"):
        f = raster_for(set_name, grid, year)
        if not f.exists():
            return np.full((len(rows), len(names_for(set_name, grid))), np.nan, np.float32)
        r, c = _cells_on(grid, "hls30", rows, cols)
        return sample_cells(f, r, c)
    if set_name == "dem":
        return sample_cells(raster_for("dem", grid, year), rows, cols)
    if set_name == "av3":
        from .av3_features import sample as av3_sample
        return av3_sample(grid, year, rows, cols)
    raise ValueError(set_name)


def training_table(labels: pd.DataFrame, grid: str, sets: list[str],
                   cache: bool = True) -> pd.DataFrame:
    """Feature columns for every label row (index aligned with `labels`)."""
    rc, cc = f"row_{grid}", f"col_{grid}"
    out = []
    for s in sets:
        if s == "time":
            out.append(pd.DataFrame({"t": labels["t"].to_numpy()}, index=labels.index))
            continue
        cols = names_for(s, grid)
        block = pd.DataFrame(np.nan, index=labels.index, columns=cols, dtype=np.float32)
        for y, sub in labels.groupby("year"):
            if not cache:
                block.loc[sub.index] = sample_set(s, grid, int(y), sub[rc].to_numpy(),
                                                  sub[cc].to_numpy())
                continue
            f = data_path("features", grid, f"{s}_{y}.parquet")
            cached = pd.read_parquet(f) if f.exists() else None
            need = (~sub["obs_id"].isin(cached.index)).to_numpy() if cached is not None \
                else np.ones(len(sub), bool)
            if need.any():
                todo = sub[need]
                vals = sample_set(s, grid, int(y), todo[rc].to_numpy(), todo[cc].to_numpy())
                new = pd.DataFrame(vals, index=todo["obs_id"].to_numpy(), columns=cols)
                cached = new if cached is None else pd.concat([cached, new])
                cached.to_parquet(f)
            block.loc[sub.index] = cached.loc[sub["obs_id"].to_numpy()].to_numpy()
        out.append(block)
    return pd.concat(out, axis=1)


def stack_window(grid: str, sets: list[str], year: int, win: Window,
                 t_value: float | None = None) -> tuple[np.ndarray, list[str]]:
    """Feature stack (n_pixels, n_features) for a window of `grid`, row-major."""
    from .grid import get_grid

    g = get_grid(grid)
    h, w = int(win.height), int(win.width)
    feats, names = [], []
    for s in sets:
        if s == "time":
            feats.append(np.full((1, h, w), t_value if t_value is not None else year + 0.5,
                                 np.float32))
            names.append("t")
            continue
        if s == "av3":
            from .av3_features import stack_window as av3_window
            a = av3_window(grid, year, win)
            feats.append(a)
            names += names_for("av3", grid)
            continue
        f = raster_for(s, grid, year)
        if s in ("hls", "s1") and grid != "hls30":
            a = _read_resampled_hls30(f, g, win, len(names_for(s, grid)))
        elif f is None or not f.exists():
            a = np.full((len(names_for(s, grid)), h, w), np.nan, np.float32)
        else:
            with rasterio.open(f) as src:
                a = src.read(window=win).astype(np.float32)
                if src.nodata is not None and not np.isnan(src.nodata):
                    a[a == src.nodata] = np.nan
        if s in ("aef", "aefctx"):
            from .io_alphaearth import dequantize
            a = dequantize(np.where(np.isnan(a), -128, a).astype(np.int8))
        feats.append(a)
        names += names_for(s, grid)
    arr = np.concatenate(feats, 0)
    return arr.reshape(arr.shape[0], -1).T, names


def _read_resampled_hls30(f, g, win, nb):
    """Nearest-neighbour lookup of hls30 cells for a window of a finer grid."""
    from .grid import get_grid

    h, w = int(win.height), int(win.width)
    if f is None or not f.exists():
        return np.full((nb, h, w), np.nan, np.float32)
    s = get_grid("hls30")
    rr, cc = np.meshgrid(np.arange(win.row_off, win.row_off + h),
                         np.arange(win.col_off, win.col_off + w), indexing="ij")
    x, y = g.xy(rr, cc)
    r30, c30 = s.rowcol(x, y)
    r0, r1 = r30.min(), r30.max() + 1
    c0, c1 = c30.min(), c30.max() + 1
    with rasterio.open(f) as src:
        a = src.read(window=Window(c0, r0, c1 - c0, r1 - r0)).astype(np.float32)
    return a[:, r30 - r0, c30 - c0]
