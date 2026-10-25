"""Sentinel-1 radiometrically terrain-corrected backscatter: monthly and annual composites.

Primary source would be NASA's OPERA RTC-S1 (ASF DAAC). The Earthdata account used here
is not authorized for ASF downloads, so the default source is the Microsoft Planetary
Computer `sentinel-1-rtc` collection (gamma0 RTC, 10 m, UTM COGs), which is the same
physical quantity. Each scene is averaged (in linear power) onto the hls30 grid with
GDAL picking a suitable overview, then per-month medians are taken. Annual predictors
are summarized from the monthly composites: median VV/VH (dB), VH p10/p90, the VH-VV
ratio, and the month-to-month VH standard deviation.

`s1_max_scenes_per_month` caps scenes per relative orbit per month (small profile).
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from .aoi import aoi_bounds_wgs84, aoi_mask
from .config import data_path, load_config, resources
from .grid import get_grid

log = logging.getLogger(__name__)

STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
TOKEN_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token/sentinel-1-rtc"
POLS = ("vv", "vh")
FILL = -32768
DB_SCALE = 100  # stored as int16 centi-dB
_token = {"value": None, "t": 0.0}


def sas_token() -> str:
    if _token["value"] is None or time.time() - _token["t"] > 1800:
        _token["value"] = requests.get(TOKEN_URL, timeout=60).json()["token"]
        _token["t"] = time.time()
    return _token["value"]


def search(year: int):
    import pystac_client

    cat = pystac_client.Client.open(STAC)
    items = list(cat.search(collections=["sentinel-1-rtc"], bbox=list(aoi_bounds_wgs84()),
                            datetime=f"{year}-01-01/{year}-12-31").items())
    cap = resources().s1_max_scenes_per_month
    if cap:
        groups = defaultdict(list)
        for it in sorted(items, key=lambda i: i.datetime):
            key = (it.datetime.month, it.properties.get("sat:relative_orbit"))
            groups[key].append(it)
        items = [it for g in groups.values() for it in g[:cap]]
    log.info("S1 %s: %d scenes", year, len(items))
    return items


def _read_item(item, grid) -> dict | None:
    out = {}
    env = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="5",
               GDAL_HTTP_RETRY_DELAY="5")
    for pol in POLS:
        if pol not in item.assets:
            return None
        href = "/vsicurl/" + item.assets[pol].href + "?" + sas_token()
        for attempt in range(4):
            try:
                with rasterio.Env(**env), rasterio.open(href) as src:
                    with WarpedVRT(src, crs=grid.crs, transform=grid.transform,
                                   width=grid.width, height=grid.height,
                                   resampling=Resampling.average, src_nodata=src.nodata,
                                   nodata=np.nan, dtype="float32") as vrt:
                        a = vrt.read(1)
                break
            except rasterio.errors.RasterioIOError as e:
                if attempt == 3:
                    log.warning("failed %s: %s", item.id, e)
                    return None
                time.sleep(10 * (attempt + 1))
        a[(a <= 0)] = np.nan
        out[pol] = a
    return out


def monthly_path(year: int) -> Path:
    return data_path("interim", "s1", "hls30", f"s1_monthly_{year}.tif")


def build_monthly(year: int, force: bool = False) -> Path:
    f = monthly_path(year)
    if f.exists() and not force:
        return f
    grid = get_grid("hls30")
    items = search(year)
    mask = aoi_mask(grid)
    out = np.full((12, 3, *grid.shape), FILL, np.int16)  # vv, vh, count
    for m in range(1, 13):
        ms = [it for it in items if it.datetime.month == m]
        if not ms:
            out[m - 1, 2] = 0
            continue
        with ThreadPoolExecutor(resources().n_workers) as ex:
            arrs = [a for a in ex.map(lambda it: _read_item(it, grid), ms) if a is not None]
        for i, pol in enumerate(POLS):
            stack = np.stack([a[pol] for a in arrs])
            with np.errstate(all="ignore"):
                med = 10 * np.log10(np.nanmedian(stack, axis=0))
            out[m - 1, i] = np.where(np.isfinite(med), np.round(med * DB_SCALE), FILL)
        out[m - 1, 2] = np.sum([np.isfinite(a["vh"]) for a in arrs], axis=0)
        log.info("S1 %s month %02d: %d scenes", year, m, len(arrs))
    out = out.reshape(36, *grid.shape)
    out[:, ~mask] = FILL
    prof = grid.profile(count=36, dtype="int16", nodata=FILL, tiled=True, blockxsize=256,
                        blockysize=256, compress="zstd", predictor=2)
    with rasterio.open(f, "w", **prof) as dst:
        dst.write(out)
        dst.descriptions = tuple(f"{v}_m{m:02d}" for m in range(1, 13)
                                 for v in ("vv_db", "vh_db", "count"))
    return f


ANNUAL_BANDS = ["vv_p50", "vh_p50", "vh_p10", "vh_p90", "vhvv_p50", "vh_sd",
                "s1_n_months"]


def annual_from_monthly(year: int, force: bool = False) -> Path:
    f = data_path("interim", "s1", "hls30", f"s1_annual_{year}.tif")
    if f.exists() and not force:
        return f
    grid = get_grid("hls30")
    with rasterio.open(monthly_path(year)) as src:
        a = src.read().astype(np.float32).reshape(12, 3, *grid.shape)
    cnt = np.where(a[:, 2] == FILL, 0, a[:, 2])
    db = a[:, :2]
    db[db == FILL] = np.nan
    db /= DB_SCALE
    with np.errstate(all="ignore"):
        vv, vh = db[:, 0], db[:, 1]
        feats = [np.nanmedian(vv, 0), np.nanmedian(vh, 0), np.nanpercentile(vh, 10, 0),
                 np.nanpercentile(vh, 90, 0), np.nanmedian(vh - vv, 0), np.nanstd(vh, 0),
                 (cnt > 0).sum(0).astype(np.float32)]
    arr = np.stack(feats).astype(np.float32)
    arr[:, ~aoi_mask(grid)] = np.nan
    prof = grid.profile(count=arr.shape[0], dtype="float32", nodata=np.nan, tiled=True,
                        blockxsize=256, blockysize=256, compress="zstd", predictor=3)
    with rasterio.open(f, "w", **prof) as dst:
        dst.write(arr)
        dst.descriptions = tuple(ANNUAL_BANDS)
    return f


def build_year(year: int, force: bool = False) -> Path:
    build_monthly(year, force)
    return annual_from_monthly(year, force)


def cli(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(prog="azuero s1")
    p.add_argument("--years", type=int, nargs="*", default=None)
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    for y in a.years or load_config()["years"]:
        build_year(y, a.force)
