"""Spaceborne lidar retrieval via SlideRule Earth: GEDI L2A and ICESat-2 (PhoREAL).

Both sensors are pulled for the AOI polygon in quarterly chunks and cached as GeoParquet
under `data/raw/lidar/{sensor}/{year}Q{q}.parquet`. Quality screening, harmonization and
gridding happen later (`azuero.labels`), so the raw tables keep the QC fields.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from . import sr
from .aoi import aoi_utm
from .config import data_path, load_config

log = logging.getLogger(__name__)

GEDI_RH = [25, 50, 75, 90, 95, 98, 100]
GEDI_ANC = ["rh", "quality_flag", "degrade_flag", "surface_flag", "selected_algorithm",
            "num_detectedmodes", "land_cover_data/landsat_treecover",
            "land_cover_data/urban_proportion", "land_cover_data/region_class",
            "digital_elevation_model", "rx_assess/quality_flag"]
GEDI_COVERAGE_BEAMS = {0, 1, 2, 3}
IS2_ATL08_FIELDS = ["night_flag", "segment_watermask", "segment_landcover",
                    "canopy/h_canopy_20m", "terrain/h_te_best_fit", "msw_flag",
                    "cloud_flag_atm", "sigma_topo", "terrain/terrain_slope"]
IS2_PCTS = list(range(5, 100, 5))


def sliderule_region(buffer_m: float = 500.0, tolerance_m: float = 300.0) -> list[dict]:
    """AOI polygon as a SlideRule `poly` (counter-clockwise lon/lat ring)."""
    g = aoi_utm().geometry.iloc[0].buffer(buffer_m).simplify(tolerance_m)
    g = gpd.GeoSeries([g], crs=aoi_utm().crs).to_crs(4326).iloc[0]
    from shapely.geometry.polygon import orient

    ring = orient(g, sign=1.0).exterior.coords
    return [{"lon": float(x), "lat": float(y)} for x, y in ring]


def quarters(year: int):
    for q, (m0, m1) in enumerate([(1, 4), (4, 7), (7, 10), (10, 13)], start=1):
        t0 = f"{year}-{m0:02d}-01T00:00:00Z"
        t1 = f"{year + (m1 == 13)}-{(m1 - 1) % 12 + 1:02d}-01T00:00:00Z"
        yield q, t0, t1


def _retry(fn, tries: int = 4):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # network / server hiccups
            if i == tries - 1:
                raise
            wait = 30 * 2 ** i
            log.warning("request failed (%s); retrying in %ss", e, wait)
            time.sleep(wait)


def _to_utm(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    crs = load_config()["crs"]
    df = df.set_crs(4326, allow_override=True) if df.crs is None else df
    lon, lat = df.geometry.x.to_numpy(), df.geometry.y.to_numpy()
    utm = df.to_crs(crs)
    df = df.assign(lon=lon, lat=lat, x=utm.geometry.x.to_numpy(), y=utm.geometry.y.to_numpy())
    return df


def fetch_gedi(year: int, force: bool = False) -> list[Path]:
    from sliderule import gedi

    sr.init()
    poly = sliderule_region()
    out = []
    for q, t0, t1 in quarters(year):
        f = data_path("raw", "lidar", "gedi", f"{year}Q{q}.parquet")
        out.append(f)
        if f.exists() and not force:
            continue
        parms = {"poly": poly, "t0": t0, "t1": t1, "anc_fields": GEDI_ANC}
        df = _retry(lambda: gedi.gedi02ap(parms))
        if df is None or len(df) == 0:
            pd.DataFrame().to_parquet(f)
            log.info("GEDI %s Q%s: no shots", year, q)
            continue
        df = df.rename(columns={c: c.replace("/", "_") for c in df.columns if "/" in c})
        rh = np.stack(df["rh"].to_numpy())
        for p in GEDI_RH:
            df[f"rh{p}"] = rh[:, p].astype("float32")
        df = df.drop(columns=["rh"])
        df["coverage_beam"] = df["beam"].isin(GEDI_COVERAGE_BEAMS)
        df = _to_utm(df).reset_index()
        df.to_parquet(f)
        log.info("GEDI %s Q%s: %d shots", year, q, len(df))
    return out


def fetch_icesat2(year: int, force: bool = False) -> list[Path]:
    from sliderule import sliderule as _  # noqa: F401  (ensures client import)
    import importlib

    srmod = importlib.import_module("sliderule.sliderule")
    sr.init()
    icfg = load_config()["icesat2"]
    poly = sliderule_region()
    out = []
    for q, t0, t1 in quarters(year):
        f = data_path("raw", "lidar", "icesat2", f"{year}Q{q}.parquet")
        out.append(f)
        if f.exists() and not force:
            continue
        parms = {
            "poly": poly, "t0": t0, "t1": t1,
            "srt": 0, "cnf": 0,
            "atl08_class": icfg["atl08_class"],
            "len": icfg["segment_len_m"], "res": icfg["segment_res_m"],
            "cnt": 5, "ats": 10, "pass_invalid": True,
            "phoreal": icfg["phoreal"],
            "atl08_fields": IS2_ATL08_FIELDS,
        }
        df = _retry(lambda: srmod.run("atl03x", parms))
        if df is None or len(df) == 0:
            pd.DataFrame().to_parquet(f)
            log.info("ICESat-2 %s Q%s: no segments", year, q)
            continue
        df = df.rename(columns={c: c.split("/")[-1] for c in df.columns if "/" in c})
        m = np.stack(df["canopy_h_metrics"].to_numpy()).astype("float32")
        for i, p in enumerate(IS2_PCTS):
            if p in (25, 50, 75, 90, 95):
                df[f"h_canopy_p{p}"] = m[:, i]
        df = df.drop(columns=["canopy_h_metrics"])
        df = _to_utm(df).reset_index().rename(columns={"time_ns": "time"})
        df.to_parquet(f)
        log.info("ICESat-2 %s Q%s: %d segments", year, q, len(df))
    return out


def load_raw(sensor: str, years=None) -> gpd.GeoDataFrame:
    years = years or load_config()["years"]
    parts = []
    for y in years:
        for f in sorted(data_path("raw", "lidar", sensor).glob(f"{y}Q*.parquet")):
            d = pd.read_parquet(f)
            if len(d):
                parts.append(d)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def cli(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(prog="azuero lidar")
    p.add_argument("--sensor", choices=["gedi", "icesat2", "both"], default="both")
    p.add_argument("--years", type=int, nargs="*", default=None)
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    years = a.years or load_config()["lidar_years"]
    for y in years:
        if a.sensor in ("gedi", "both") and y >= 2019:
            fetch_gedi(y, a.force)
        if a.sensor in ("icesat2", "both"):
            fetch_icesat2(y, a.force)
