"""AlphaEarth Foundations annual embeddings (Source Cooperative copy).

The COGs are stored "bottom-up" (origin at the lower-left, positive y resolution), so we
read the GeoTIFFs directly and flip rows ourselves instead of going through the warped
VRTs. Each year is assembled on its native grid (10 m at overview level 0, 20 m at level
1, per the `alphaearth_read_level` resource toggle), masked to the AOI, and written as
an int8 GeoTIFF using the dataset's own quantization. Other grids (e.g. hls30) are
derived by exact area-weighted averaging of the de-quantized values.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from importlib import resources as ires
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from .aoi import aoi_mask
from .config import data_path, resources
from .grid import get_grid

log = logging.getLogger(__name__)

HTTP_ROOT = "https://data.source.coop/"
S3_PREFIX = "s3://us-west-2.opendata.source.coop/"
NBANDS = 64
NODATA = -128
GDAL_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tiff",
                GDAL_HTTP_MULTIRANGE="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
                GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="5", GDAL_CACHEMAX=512)


def dequantize(v: np.ndarray) -> np.ndarray:
    """int8 -> float32 in [-1, 1]; nodata (-128) -> NaN."""
    v = np.asarray(v)
    out = ((v.astype(np.float32) / 127.5) ** 2) * np.sign(v)
    out[v == NODATA] = np.nan
    return out


def quantize(x: np.ndarray) -> np.ndarray:
    """Inverse of `dequantize` (NaN -> -128)."""
    q = np.sign(x) * np.sqrt(np.abs(x)) * 127.5
    q = np.clip(np.round(np.nan_to_num(q, nan=0)), -127, 127).astype(np.int8)
    q[~np.isfinite(x)] = NODATA
    return q


def index() -> pd.DataFrame:
    """Tiles of the AlphaEarth annual index (UTM 17N) intersecting the AOI."""
    f = ires.files("azuero.resources") / "aef_index_17N_azuero.csv"
    df = pd.read_csv(f)
    df["url"] = "/vsicurl/" + df["path"].str.replace(S3_PREFIX, HTTP_ROOT, regex=False)
    return df


def native_grid_name(level: int | None = None) -> str:
    level = resources().alphaearth_read_level if level is None else level
    return {0: "aef10", 1: "aef20"}[level]


def stack_path(year: int, grid_name: str) -> Path:
    return data_path("interim", "alphaearth", grid_name, f"aef_{year}.tif")


def _read_tile_part(args):
    url, level, win, bands = args
    # rasterio's overview_level counts from the first overview (0 = 2x reduction).
    kw = {} if level == 0 else {"overview_level": level - 1}
    with rasterio.Env(**GDAL_ENV):
        with rasterio.open(url, **kw) as src:
            return src.read(bands, window=win)


def _assemble_native(year: int, level: int) -> np.ndarray:
    grid = get_grid(native_grid_name(level))
    res = grid.res
    out = np.full((NBANDS, *grid.shape), NODATA, np.int8)
    tiles = index().query("year == @year")
    jobs, places = [], []
    block = 1024
    for t in tiles.itertuples():
        x0, x1 = max(t.utm_west, grid.xmin), min(t.utm_east, grid.xmax)
        y0, y1 = max(t.utm_south, grid.ymin), min(t.utm_north, grid.ymax)
        if x0 >= x1 or y0 >= y1:
            continue
        # File rows count upward from the tile's southern edge.
        fc0, fc1 = int((x0 - t.utm_west) / res), int((x1 - t.utm_west) / res)
        fr0, fr1 = int((y0 - t.utm_south) / res), int((y1 - t.utm_south) / res)
        for r in range(fr0, fr1, block):
            for c in range(fc0, fc1, block):
                h, w = min(block, fr1 - r), min(block, fc1 - c)
                win = Window(c, r, w, h)
                # Destination (north-up) rows for file rows r..r+h-1, flipped.
                y_top = t.utm_south + (r + h) * res
                gr = int(round((grid.ymax - y_top) / res))
                gc = int(round((t.utm_west + c * res - grid.xmin) / res))
                for b0 in range(0, NBANDS, 16):
                    jobs.append((t.url, level, win, list(range(b0 + 1, b0 + 17))))
                    places.append((b0, gr, gc, h, w))
    n = resources().n_workers * 2
    log.info("AlphaEarth %s: %d read jobs over %d tiles", year, len(jobs), len(tiles))
    with ThreadPoolExecutor(n) as ex:
        for (b0, gr, gc, h, w), arr in zip(places, ex.map(_read_tile_part, jobs)):
            out[b0:b0 + 16, gr:gr + h, gc:gc + w] = arr[:, ::-1, :]
    return out


def _write(path: Path, arr: np.ndarray, grid) -> None:
    prof = grid.profile(count=arr.shape[0], dtype="int8", nodata=NODATA, tiled=True,
                        blockxsize=256, blockysize=256, compress="zstd", zstd_level=9,
                        predictor=2, interleave="pixel", BIGTIFF="IF_SAFER")
    tmp = path.with_suffix(".tmp.tif")
    with rasterio.open(tmp, "w", **prof) as dst:
        dst.write(arr)
        dst.descriptions = tuple(f"A{i:02d}" for i in range(arr.shape[0]))
    tmp.rename(path)


def build_year(year: int, grid_names=("aef20", "hls30"), force: bool = False) -> dict:
    """Assemble the native-resolution stack for `year` and derive the requested grids."""
    from .regrid import regrid_mean

    level = resources().alphaearth_read_level
    native = native_grid_name(level)
    paths = {g: stack_path(year, g) for g in grid_names}
    native_path = stack_path(year, native)
    if all(p.exists() for p in paths.values()) and not force:
        return paths
    ngrid = get_grid(native)
    if native_path.exists() and not force:
        with rasterio.open(native_path) as src:
            arr = src.read()
    else:
        arr = _assemble_native(year, level)
        arr[:, ~aoi_mask(ngrid)] = NODATA
        _write(native_path, arr, ngrid)
        log.info("wrote %s", native_path)
    for g in grid_names:
        if g == native or (paths[g].exists() and not force):
            continue
        dgrid = get_grid(g)
        if dgrid.res < ngrid.res:
            raise RuntimeError(f"grid {g} is finer than the AlphaEarth read level {native}")
        valid = arr[0] != NODATA
        out = np.empty((NBANDS, *dgrid.shape), np.int8)
        for b0 in range(0, NBANDS, 8):  # bound memory: 8 float bands at a time
            f = regrid_mean(dequantize(arr[b0:b0 + 8]), ngrid, dgrid, valid=valid)
            out[b0:b0 + 8] = quantize(f)
        out[:, ~aoi_mask(dgrid)] = NODATA
        _write(paths[g], out, dgrid)
        log.info("wrote %s", paths[g])
    paths[native] = native_path
    return paths


def sample(year: int, grid_name: str, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """De-quantized embeddings (n, 64) at grid row/col indices for one year."""
    with rasterio.open(stack_path(year, grid_name)) as src:
        arr = src.read()
    return dequantize(arr[:, rows, cols].T)


def cli(argv=None) -> None:
    import argparse

    from .config import load_config

    p = argparse.ArgumentParser(prog="azuero alphaearth")
    p.add_argument("--years", type=int, nargs="*", default=None)
    p.add_argument("--grids", nargs="*", default=["aef20", "hls30"])
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    for y in a.years or load_config()["years"]:
        build_year(y, tuple(a.grids), a.force)
