"""Copernicus DEM GLO-30 and derived terrain predictors.

Tiles are read from the public AWS bucket, mosaicked and warped (bilinear) to each analysis
grid, then terrain derivatives are computed on that grid: slope, aspect (as northness /
eastness), topographic position index at two scales, and a simple topographic wetness
proxy from local relief. Heights are EGM2008 orthometric.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from scipy import ndimage

from .aoi import aoi_bounds_wgs84, aoi_mask
from .config import data_path
from .grid import get_grid

log = logging.getLogger(__name__)

COP_URL = ("/vsicurl/https://copernicus-dem-30m.s3.amazonaws.com/"
           "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM/"
           "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM.tif")
BANDS = ["elevation", "slope", "northness", "eastness", "tpi_150m", "tpi_1km", "relief_1km"]


def tile_urls() -> list[str]:
    w, s, e, n = aoi_bounds_wgs84(pad_deg=0.05)
    urls = []
    for lat in range(math.floor(s), math.floor(n) + 1):
        for lon in range(math.floor(w), math.floor(e) + 1):
            urls.append(COP_URL.format(ns="N" if lat >= 0 else "S", lat=abs(lat),
                                       ew="E" if lon >= 0 else "W", lon=abs(lon)))
    return urls


def dem_on_grid(grid) -> np.ndarray:
    """Bilinear-warp the GLO-30 tiles onto `grid` (NaN outside data / ocean)."""
    out = np.full(grid.shape, np.nan, np.float32)
    for url in tile_urls():
        try:
            src = rasterio.open(url)
        except rasterio.errors.RasterioIOError:
            log.info("no DEM tile %s (ocean)", url)
            continue
        with src, WarpedVRT(src, crs=grid.crs, transform=grid.transform, width=grid.width,
                            height=grid.height, resampling=Resampling.bilinear,
                            src_nodata=src.nodata, nodata=np.nan, dtype="float32") as vrt:
            a = vrt.read(1)
        ok = np.isfinite(a)
        out[ok] = a[ok]
    return out


def terrain(elev: np.ndarray, res: float) -> dict[str, np.ndarray]:
    z = np.where(np.isfinite(elev), elev, 0.0)
    gy, gx = np.gradient(z, res)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    aspect = np.arctan2(-gx, gy)  # 0 = north-facing downslope direction
    def mean_filter(r_m):
        k = max(int(round(r_m / res)), 1)
        return ndimage.uniform_filter(z, size=2 * k + 1, mode="nearest")
    def relief(r_m):
        k = max(int(round(r_m / res)), 1)
        return (ndimage.maximum_filter(z, size=2 * k + 1, mode="nearest")
                - ndimage.minimum_filter(z, size=2 * k + 1, mode="nearest"))
    return {
        "elevation": elev,
        "slope": slope,
        "northness": np.cos(aspect) * np.sin(np.radians(slope)),
        "eastness": np.sin(aspect) * np.sin(np.radians(slope)),
        "tpi_150m": z - mean_filter(150),
        "tpi_1km": z - mean_filter(1000),
        "relief_1km": relief(1000),
    }


def stack_path(grid_name: str):
    return data_path("interim", "dem", f"terrain_{grid_name}.tif")


def build(grid_name: str, force: bool = False):
    f = stack_path(grid_name)
    if f.exists() and not force:
        return f
    grid = get_grid(grid_name)
    elev = dem_on_grid(grid)
    t = terrain(elev, grid.res)
    mask = aoi_mask(grid) & np.isfinite(elev)
    arr = np.stack([np.where(mask, t[b], np.nan).astype(np.float32) for b in BANDS])
    prof = grid.profile(count=len(BANDS), dtype="float32", nodata=np.nan, tiled=True,
                        blockxsize=256, blockysize=256, compress="zstd", predictor=3)
    with rasterio.open(f, "w", **prof) as dst:
        dst.write(arr)
        dst.descriptions = tuple(BANDS)
    log.info("wrote %s", f)
    return f


def cli(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(prog="azuero dem")
    p.add_argument("--grids", nargs="*", default=["hls30", "aef20"])
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    for g in a.grids:
        build(g, a.force)
