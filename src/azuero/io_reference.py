"""Independent reference layers, resampled onto the analysis grids.

* Hansen Global Forest Change v1.13 (2000-2025): `lossyear` (mode) and `treecover2000`
  (mean), used for stability screening of lidar pairs and as context. GFC is never
  used as a training target.
* Meta / WRI global 1 m canopy height (Tolan et al. 2024; mostly 2018-2020 Maxar imagery)
  and ETH global 10 m canopy height 2020 (Lang et al. 2023), used only for comparison
  with our product.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds

from .aoi import aoi_bounds_wgs84, aoi_mask, aoi_utm
from .config import data_path, resources
from .grid import get_grid

log = logging.getLogger(__name__)

GFC_URL = ("https://storage.googleapis.com/earthenginepartners-hansen/GFC-2025-v1.13/"
           "Hansen_GFC-2025-v1.13_{layer}_{tile}.tif")
META_ROOT = "https://dataforgood-fb-data.s3.amazonaws.com/forests/v1/alsgedi_global_v6_float"
ETH_URL = ("https://libdrive.ethz.ch/index.php/s/cO8or7iOe5dT2Rt/download?path=%2F3deg_cogs"
           "&files=ETH_GlobalCanopyHeight_10m_2020_{tile}_{kind}.tif")
VSI_ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="5",
               GDAL_HTTP_RETRY_DELAY="5", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif")


def _warp_into(out: np.ndarray, src, grid, resampling, src_nodata=None) -> None:
    with WarpedVRT(src, crs=grid.crs, transform=grid.transform, width=grid.width,
                   height=grid.height, resampling=resampling, src_nodata=src_nodata,
                   nodata=np.nan, dtype="float32") as vrt:
        a = vrt.read(1)
    ok = np.isfinite(a)
    out[ok] = a[ok]


def _write(path: Path, arrays: dict[str, np.ndarray], grid) -> Path:
    arr = np.stack(list(arrays.values())).astype(np.float32)
    arr[:, ~aoi_mask(grid)] = np.nan
    prof = grid.profile(count=arr.shape[0], dtype="float32", nodata=np.nan, tiled=True,
                        blockxsize=256, blockysize=256, compress="zstd", predictor=3)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(arr)
        dst.descriptions = tuple(arrays)
    log.info("wrote %s", path)
    return path


def _download(url: str, dst: Path) -> Path:
    if dst.exists():
        return dst
    tmp = dst.with_suffix(".part")
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 22):
                f.write(chunk)
    tmp.rename(dst)
    return dst


# --- Hansen GFC -----------------------------------------------------------------------

def gfc_tiles() -> list[str]:
    w, s, e, n = aoi_bounds_wgs84(pad_deg=0.02)
    tiles = set()
    for lat in (s, n):
        for lon in (w, e):
            top = math.ceil(lat / 10) * 10
            left = math.floor(lon / 10) * 10
            tiles.add(f"{abs(top):02d}{'N' if top >= 0 else 'S'}_"
                      f"{abs(left):03d}{'E' if left >= 0 else 'W'}")
    return sorted(tiles)


def build_gfc(grid_name: str, force: bool = False) -> Path:
    f = data_path("interim", "reference", f"gfc_{grid_name}.tif")
    if f.exists() and not force:
        return f
    grid = get_grid(grid_name)
    loss = np.full(grid.shape, np.nan, np.float32)
    tc = np.full(grid.shape, np.nan, np.float32)
    with rasterio.Env(**VSI_ENV):
        for t in gfc_tiles():
            with rasterio.open("/vsicurl/" + GFC_URL.format(layer="lossyear", tile=t)) as src:
                _warp_into(loss, src, grid, Resampling.mode)
            with rasterio.open("/vsicurl/" + GFC_URL.format(layer="treecover2000", tile=t)) as src:
                _warp_into(tc, src, grid, Resampling.average)
    return _write(f, {"lossyear": loss, "treecover2000": tc}, grid)


# --- Meta / WRI 1 m CHM -----------------------------------------------------------------

def meta_tiles() -> list[str]:
    import shapely.geometry as sg

    aoi = aoi_utm().to_crs(4326).geometry.iloc[0]
    fc = requests.get(f"{META_ROOT}/tiles.geojson", timeout=300).json()
    return [ft["properties"]["tile"] for ft in fc["features"]
            if sg.shape(ft["geometry"]).intersects(aoi)]


def build_meta(grid_names, force: bool = False) -> list[Path]:
    """Mean of the 1 m CHM within each grid cell, for every grid in `grid_names`
    (each 1 m tile is downloaded once, warped to all grids, then discarded)."""
    grid_names = [grid_names] if isinstance(grid_names, str) else list(grid_names)
    files = {g: data_path("interim", "reference", f"meta_chm_{g}.tif") for g in grid_names}
    todo = [g for g in grid_names if force or not files[g].exists()]
    if not todo:
        return list(files.values())
    outs = {g: np.full(get_grid(g).shape, np.nan, np.float32) for g in todo}
    tmpdir = data_path("tmp", "meta")
    for t in meta_tiles():
        local = _download(f"{META_ROOT}/chm/{t}.tif", tmpdir / f"{t}.tif")
        with rasterio.open(local) as src:
            for g in todo:
                _warp_into(outs[g], src, get_grid(g), Resampling.average)
        if resources().delete_raw_after_composite:
            local.unlink()
        log.info("Meta CHM tile %s done", t)
    for g in todo:
        _write(files[g], {"meta_chm": outs[g]}, get_grid(g))
    return list(files.values())


# --- ETH 10 m CHM 2020 --------------------------------------------------------------------

def eth_tiles() -> list[str]:
    w, s, e, n = aoi_bounds_wgs84(pad_deg=0.02)
    tiles = set()
    for lat in (s, n):
        for lon in (w, e):
            b = math.floor(lat / 3) * 3
            l = math.floor(lon / 3) * 3
            tiles.add(f"{'N' if b >= 0 else 'S'}{abs(b):02d}{'E' if l >= 0 else 'W'}{abs(l):03d}")
    return sorted(tiles)


def build_eth(grid_names, force: bool = False) -> list[Path]:
    grid_names = [grid_names] if isinstance(grid_names, str) else list(grid_names)
    files = {g: data_path("interim", "reference", f"eth_chm_{g}.tif") for g in grid_names}
    todo = [g for g in grid_names if force or not files[g].exists()]
    if not todo:
        return list(files.values())
    outs = {g: {k: np.full(get_grid(g).shape, np.nan, np.float32)
                for k in ("eth_chm", "eth_chm_sd")} for g in todo}
    tmpdir = data_path("tmp", "eth")
    for t in eth_tiles():
        for kind, key in (("Map", "eth_chm"), ("Map_SD", "eth_chm_sd")):
            local = _download(ETH_URL.format(tile=t, kind=kind), tmpdir / f"{t}_{kind}.tif")
            with rasterio.open(local) as src:
                for g in todo:
                    _warp_into(outs[g][key], src, get_grid(g), Resampling.average,
                               src_nodata=255)
            if resources().delete_raw_after_composite:
                local.unlink()
    for g in todo:
        _write(files[g], outs[g], get_grid(g))
    return list(files.values())


def cli(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(prog="azuero reference")
    p.add_argument("--layers", nargs="*", default=["gfc", "eth", "meta"])
    p.add_argument("--grids", nargs="*", default=["hls30", "aef20"])
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    for layer in a.layers:
        if layer == "gfc":
            for g in a.grids:
                build_gfc(g, a.force)
        elif layer == "eth":
            build_eth(a.grids, a.force)
        elif layer == "meta":
            build_meta(a.grids, a.force)
