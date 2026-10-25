"""Harmonized Landsat Sentinel-2 (HLS v2.0) monthly and annual composites.

For each MGRS tile touching the AOI and each month, cloud/shadow-masked (Fmask) scenes
from HLSL30 and HLSS30 are reduced to a per-pixel median of red, NIR, SWIR1, SWIR2 and
the NDVI / NBR / NDMI indices. Annual predictors are then summarized from the 12 monthly
composites (p10/p50/p90 of the indices, median reflectance, dry-season Jan-Apr median),
which weights months equally regardless of how many clear scenes each one has.

Resource toggles: `hls_max_cloud_pct` filters scenes by granule cloud cover,
`hls_bands_subset` limits the reflectance bands, and `delete_raw_after_composite`
discards downloaded files once a month is composited.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds

from .aoi import aoi_bounds_wgs84, aoi_mask
from .config import data_path, load_config, resources
from .earthdata import login, session
from .grid import get_grid

log = logging.getLogger(__name__)

BAND_IDS = {
    "L30": {"BLUE": "B02", "GREEN": "B03", "RED": "B04", "NIR": "B05", "SWIR1": "B06",
            "SWIR2": "B07"},
    "S30": {"BLUE": "B02", "GREEN": "B03", "RED": "B04", "NIR": "B8A", "SWIR1": "B11",
            "SWIR2": "B12"},
}
INDICES = {"NDVI": ("NIR", "RED"), "NBR": ("NIR", "SWIR2"), "NDMI": ("NIR", "SWIR1")}
# Fmask bits: 1 cloud, 2 adjacent to cloud/shadow, 3 cloud shadow, 4 snow/ice.
FMASK_BAD = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)
FILL = -9999
SCALE = 1e4
ANNUAL_VARS = None  # set by annual_bands()


@dataclass
class Scene:
    gid: str
    sensor: str
    tile: str
    date: datetime
    cloud: float
    urls: dict


def variables() -> list[str]:
    bands = [b for b in resources().hls_bands_subset if b in BAND_IDS["L30"]]
    for need in ("RED", "NIR", "SWIR1", "SWIR2"):
        if need not in bands:
            bands.append(need)
    return bands + list(INDICES)


def search(year: int) -> list[Scene]:
    import earthaccess

    login()
    res = resources()
    w, s, e, n = aoi_bounds_wgs84()
    scenes = []
    for sensor in ("L30", "S30"):
        gs = earthaccess.search_data(short_name=f"HLS{sensor}", version="2.0",
                                     bounding_box=(w, s, e, n),
                                     temporal=(f"{year}-01-01", f"{year}-12-31T23:59:59"),
                                     count=-1)
        for g in gs:
            attrs = {a["Name"]: a["Values"][0] for a in g["umm"].get("AdditionalAttributes", [])}
            cloud = float(attrs.get("CLOUD_COVERAGE", 100))
            if cloud > res.hls_max_cloud_pct:
                continue
            gid = g["umm"]["GranuleUR"]
            links = [u for u in g.data_links(access="external") if u.endswith(".tif")]
            urls = {}
            for name in variables():
                if name in INDICES:
                    continue
                bid = BAND_IDS[sensor][name]
                urls[name] = next(u for u in links if u.endswith(f".{bid}.tif"))
            urls["Fmask"] = next(u for u in links if u.endswith(".Fmask.tif"))
            date = datetime.strptime(gid.split(".")[3][:7], "%Y%j")
            scenes.append(Scene(gid, sensor, "T" + attrs["MGRS_TILE_ID"].lstrip("T"), date,
                                cloud, urls))
    log.info("HLS %s: %d scenes <= %s%% cloud", year, len(scenes), res.hls_max_cloud_pct)
    return scenes


class Downloader:
    def __init__(self, root: Path):
        self.root = root
        self.sess = session()

    def get(self, url: str) -> Path:
        f = self.root / url.rsplit("/", 1)[-1]
        if f.exists():
            return f
        for attempt in range(5):
            try:
                r = self.sess.get(url, timeout=300)
                r.raise_for_status()
                tmp = f.with_suffix(".part")
                tmp.write_bytes(r.content)
                tmp.rename(f)
                return f
            except Exception as e:  # pragma: no cover - network
                log.warning("download failed %s (%s), retry %d", url, e, attempt)
        raise RuntimeError(f"failed to download {url}")


def _scene_vars(files: dict, win, vars_: list[str]) -> dict | None:
    """Masked reflectance + indices for one scene within a pixel window."""
    with rasterio.open(files["Fmask"]) as src:
        fm = src.read(1, window=win, boundless=True, fill_value=255)
    valid = ((fm & FMASK_BAD) == 0) & (fm != 255)
    if not valid.any():
        return None
    out = {}
    for k, f in files.items():
        if k == "Fmask":
            continue
        with rasterio.open(f) as src:
            a = src.read(1, window=win, boundless=True, fill_value=FILL).astype(np.float32)
        valid &= a != FILL
        out[k] = a / SCALE
    for k in out:
        out[k][~valid] = np.nan
    for name, (a, b) in INDICES.items():
        with np.errstate(divide="ignore", invalid="ignore"):
            out[name] = (out[a] - out[b]) / (out[a] + out[b])
    return out


def tile_bounds(tile_scenes: list[Scene], dl: Downloader, grid):
    """Intersection of the tile extent (from its Fmask file) with the grid frame."""
    f = dl.get(tile_scenes[0].urls["Fmask"])
    with rasterio.open(f) as src:
        b = src.bounds
    return (max(b.left, grid.xmin), max(b.bottom, grid.ymin),
            min(b.right, grid.xmax), min(b.top, grid.ymax))


def monthly_path(year: int, tile: str) -> Path:
    return data_path("interim", "hls", "tiles", f"{tile}_{year}_monthly.tif")


def composite_tile_year(year: int, tile: str, scenes: list[Scene], force=False,
                        strip_rows: int = 512) -> Path | None:
    """12-month median composites for one tile (bands: month-major, vars + count).

    Memory is bounded by processing one month and one row strip at a time and writing
    each strip straight to the output file.
    """
    out_f = monthly_path(year, tile)
    if out_f.exists() and not force:
        return out_f
    res = resources()
    grid = get_grid("hls30")
    raw_dir = (Path(tempfile.mkdtemp(dir=data_path("tmp"))) if res.delete_raw_after_composite
               else data_path("raw", "hls", tile))
    dl = Downloader(raw_dir)
    tmp_f = out_f.with_suffix(".part.tif")
    try:
        bounds = tile_bounds(scenes, dl, grid)
        if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
            return None
        vars_ = variables()
        nv = len(vars_) + 1
        h = int(round((bounds[3] - bounds[1]) / 30))
        w = int(round((bounds[2] - bounds[0]) / 30))
        transform = rasterio.transform.from_origin(bounds[0], bounds[3], 30, 30)
        prof = dict(driver="GTiff", width=w, height=h, count=12 * nv, dtype="int16",
                    crs=grid.crs, transform=transform, nodata=FILL, tiled=True,
                    blockxsize=256, blockysize=256, compress="zstd", predictor=2,
                    BIGTIFF="IF_SAFER")
        with rasterio.open(tmp_f, "w", **prof) as dst:
            dst.descriptions = tuple(f"{v}_m{m:02d}" for m in range(1, 13)
                                     for v in vars_ + ["count"])
            for m in range(1, 13):
                ms = [s for s in scenes if s.date.month == m]
                with ThreadPoolExecutor(res.n_workers * 2) as ex:
                    files = list(ex.map(lambda sc: {k: dl.get(u) for k, u in sc.urls.items()}, ms))
                # Pixel window of `bounds` in each scene's file (same tile grid).
                n_used = 0
                for r0 in range(0, h, strip_rows):
                    hh = min(strip_rows, h - r0)
                    arrs = []
                    for fs in files:
                        with rasterio.open(fs["Fmask"]) as src:
                            win = from_bounds(*bounds, transform=src.transform)
                            win = rasterio.windows.Window(int(round(win.col_off)),
                                                          int(round(win.row_off)) + r0, w, hh)
                        a = _scene_vars(fs, win, vars_)
                        if a is not None:
                            arrs.append(a)
                    n_used = max(n_used, len(arrs))
                    block = np.full((nv, hh, w), FILL, np.int16)
                    if arrs:
                        for i, v in enumerate(vars_):
                            stack = np.stack([a[v] for a in arrs])
                            with np.errstate(all="ignore"):
                                med = np.nanmedian(stack, axis=0)
                            block[i] = np.where(np.isfinite(med), np.round(med * SCALE), FILL)
                        block[-1] = np.sum([np.isfinite(a["NDVI"]) for a in arrs], axis=0)
                    else:
                        block[-1] = 0
                    win_out = rasterio.windows.Window(0, r0, w, hh)
                    dst.write(block, indexes=list(range((m - 1) * nv + 1, m * nv + 1)),
                              window=win_out)
                if res.delete_raw_after_composite:
                    keep = scenes[0].urls["Fmask"].rsplit("/", 1)[-1]
                    for fs in files:
                        for f in fs.values():
                            if f.name != keep:
                                f.unlink(missing_ok=True)
                log.info("HLS %s %s month %02d: %d scenes", tile, year, m, n_used)
        tmp_f.rename(out_f)
        return out_f
    finally:
        tmp_f.unlink(missing_ok=True)
        if res.delete_raw_after_composite:
            shutil.rmtree(raw_dir, ignore_errors=True)


def mosaic_monthly(year: int, tiles: list[str]) -> Path:
    """Mosaic tile composites onto the hls30 grid one month at a time; where tiles
    overlap, keep the tile with more clear observations in that month."""
    grid = get_grid("hls30")
    vars_ = variables()
    nv = len(vars_) + 1
    mask = aoi_mask(grid)
    f = data_path("interim", "hls", "hls30", f"hls_monthly_{year}.tif")
    prof = grid.profile(count=12 * nv, dtype="int16", nodata=FILL, tiled=True,
                        blockxsize=256, blockysize=256, compress="zstd", predictor=2,
                        BIGTIFF="IF_SAFER")
    srcs = [rasterio.open(monthly_path(year, t)) for t in tiles if monthly_path(year, t).exists()]
    try:
        with rasterio.open(f, "w", **prof) as dst:
            dst.descriptions = tuple(f"{v}_m{m:02d}" for m in range(1, 13)
                                     for v in vars_ + ["count"])
            for m in range(12):
                idx = list(range(m * nv + 1, (m + 1) * nv + 1))
                out = np.full((nv, *grid.shape), FILL, np.int16)
                best = np.zeros(grid.shape, np.int16)
                for src in srcs:
                    a = src.read(idx)
                    r0 = int(round((grid.ymax - src.bounds.top) / 30))
                    c0 = int(round((src.bounds.left - grid.xmin) / 30))
                    h, w = a.shape[1:]
                    sl = (slice(r0, r0 + h), slice(c0, c0 + w))
                    take = a[-1] > best[sl]
                    for i in range(nv):
                        out[i][sl][take] = a[i][take]
                    best[sl][take] = a[-1][take]
                out[:, ~mask] = FILL
                dst.write(out, indexes=idx)
    finally:
        for src in srcs:
            src.close()
    return f


def annual_bands() -> list[str]:
    vars_ = variables()
    refl = [v for v in vars_ if v not in INDICES]
    names = [f"{v}_p50" for v in refl]
    for v in INDICES:
        names += [f"{v}_p10", f"{v}_p50", f"{v}_p90", f"{v}_dry"]
    names += ["n_clear_months", "n_clear_obs"]
    return names


def annual_from_monthly(year: int, strip_rows: int = 512) -> Path:
    grid = get_grid("hls30")
    vars_ = variables()
    nv = len(vars_) + 1
    mask = aoi_mask(grid)
    f = data_path("interim", "hls", "hls30", f"hls_annual_{year}.tif")
    names = annual_bands()
    prof = grid.profile(count=len(names), dtype="float32", nodata=np.nan, tiled=True,
                        blockxsize=256, blockysize=256, compress="zstd", predictor=3)
    with rasterio.open(data_path("interim", "hls", "hls30", f"hls_monthly_{year}.tif")) as src, \
            rasterio.open(f, "w", **prof) as dst:
        dst.descriptions = tuple(names)
        for r0 in range(0, grid.height, strip_rows):
            hh = min(strip_rows, grid.height - r0)
            win = rasterio.windows.Window(0, r0, grid.width, hh)
            a = src.read(window=win).astype(np.float32)
            a[a == FILL] = np.nan
            a = a.reshape(12, nv, hh, grid.width)
            cnt = np.nan_to_num(a[:, -1])
            a = a[:, :-1] / SCALE
            feats = []
            with np.errstate(all="ignore"):
                for i, v in enumerate(vars_):
                    if v not in INDICES:
                        feats.append(np.nanmedian(a[:, i], axis=0))
                for i, v in enumerate(vars_):
                    if v in INDICES:
                        p = np.nanpercentile(a[:, i], [10, 50, 90], axis=0)
                        feats += [p[0], p[1], p[2], np.nanmedian(a[:4, i], axis=0)]
            feats.append((cnt > 0).sum(0).astype(np.float32))
            feats.append(cnt.sum(0).astype(np.float32))
            arr = np.stack(feats).astype(np.float32)
            arr[:, ~mask[r0:r0 + hh]] = np.nan
            dst.write(arr, window=win)
    return f


def build_year(year: int, force: bool = False) -> Path:
    f = data_path("interim", "hls", "hls30", f"hls_annual_{year}.tif")
    if f.exists() and not force:
        return f
    scenes = search(year)
    tiles = sorted({s.tile for s in scenes}, key=lambda t: -sum(s.tile == t for s in scenes))
    used = []
    for t in tiles:
        ts = [s for s in scenes if s.tile == t]
        if composite_tile_year(year, t, ts, force) is not None:
            used.append(t)
    mosaic_monthly(year, used)
    return annual_from_monthly(year)


def cli(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(prog="azuero hls")
    p.add_argument("--years", type=int, nargs="*", default=None)
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    for y in a.years or load_config()["years"]:
        build_year(y, a.force)
