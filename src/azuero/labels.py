"""Lidar label quality control and standardization.

Produces one table of canopy-height observations from both sensors with a common schema:
sensor, time, year, decimal year, x/y (UTM), lon/lat, h (rh98-equivalent canopy height),
and sensor-specific quality covariates. Observations are assigned to grid cells for each
analysis grid. Sensor harmonization (ICESat-2 -> GEDI scale) is done in `calibration`.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .aoi import aoi_mask
from .config import data_path, load_config, results_path
from .grid import get_grid
from .io_lidar import load_raw

log = logging.getLogger(__name__)

COMMON = ["sensor", "time", "year", "t", "x", "y", "lon", "lat", "h", "night", "strong",
          "quality"]


def decimal_year(t: pd.Series) -> np.ndarray:
    t = pd.to_datetime(t)
    start = pd.to_datetime(t.dt.year.astype(str) + "-01-01")
    end = pd.to_datetime((t.dt.year + 1).astype(str) + "-01-01")
    return (t.dt.year + (t - start) / (end - start)).to_numpy()


def qc_gedi(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    n0 = len(df)
    sens_min = np.where(df["coverage_beam"], cfg["min_sensitivity_coverage_beam"],
                        cfg["min_sensitivity"])
    steps = {
        "quality_flag": df["quality_flag"] == 1,
        "degrade_flag": df["degrade_flag"] == 0,
        "surface_flag": df["surface_flag"] == 1,
        "sensitivity": (df["sensitivity"] >= sens_min) & (df["sensitivity"] <= 1),
        "detected_modes": df["num_detectedmodes"] >= 1,
        "dem_agreement": (df["elevation_lm"] - df["digital_elevation_model"]).abs() < 50,
        "urban": df["land_cover_data_urban_proportion"] < 50,
        "height_range": df["rh98"].between(0, 80),
    }
    keep = np.ones(n0, bool)
    counts = {"raw": n0}
    for k, m in steps.items():
        keep &= m.to_numpy()
        counts[k] = int(keep.sum())
    d = df[keep]
    out = pd.DataFrame({
        "sensor": "gedi",
        "time": pd.to_datetime(d["time"]),
        "x": d["x"], "y": d["y"], "lon": d["lon"], "lat": d["lat"],
        "h": d["rh98"].astype("float32"),
        "night": (d["solar_elevation"] < 0).to_numpy(),
        "strong": ~d["coverage_beam"].to_numpy(),  # GEDI power beams
        "quality": d["sensitivity"].astype("float32"),
        "rh50": d["rh50"], "rh75": d["rh75"], "rh95": d["rh95"], "rh100": d["rh100"],
        "beam": d["beam"].astype("int16"),
    })
    return out, counts


def qc_icesat2(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    n0 = len(df)
    steps = {
        "pflags": df["pflags"] == 0,
        "water": df["segment_watermask"] == 0,
        "photons": df["photon_count"] >= cfg["min_photons"],
        "ground_photons": df["ground_photon_count"] >= cfg["min_ground_photons"],
        "cloud_msw": df["msw_flag"] <= cfg.get("max_msw_flag", 1),
        "height_range": df["h_canopy"].between(0, 80),
    }
    keep = np.ones(n0, bool)
    counts = {"raw": n0}
    for k, m in steps.items():
        keep &= m.to_numpy()
        counts[k] = int(keep.sum())
    d = df[keep]
    out = pd.DataFrame({
        "sensor": "icesat2",
        "time": pd.to_datetime(d["time"]),
        "x": d["x"], "y": d["y"], "lon": d["lon"], "lat": d["lat"],
        "h": d["h_canopy"].astype("float32"),
        "night": (d["night_flag"] >= 0.5).to_numpy(),
        "strong": d["spot"].isin([1, 3, 5]).to_numpy(),
        "quality": d["photon_count"].astype("float32"),
        "rh50": d["h_canopy_p50"], "rh75": d["h_canopy_p75"], "rh95": d["h_canopy_p95"],
        "rh100": d["h_max_canopy"],
        "veg_photons": d["vegetation_photon_count"].astype("int16"),
        "ground_photons": d["ground_photon_count"].astype("int16"),
        "rgt": d["rgt"].astype("int16"), "cycle": d["cycle"].astype("int16"),
        "spot": d["spot"].astype("int8"),
        "msw": d["msw_flag"].round().astype("int8"),
    })
    return out, counts


def flag_local_outliers(df: pd.DataFrame, radius: float = 100.0, k: float = 3.0,
                        floor_m: float = 10.0, min_n: int = 4) -> np.ndarray:
    """Flag observations far from the robust local consensus of the same sensor & year.

    |h - median| > max(k * 1.4826 * MAD, floor_m), using neighbours within `radius` m
    (excluding the point itself) when at least `min_n` neighbours exist.
    """
    flag = np.zeros(len(df), bool)
    for _, idx in df.groupby(["sensor", "year"]).indices.items():
        sub = df.iloc[idx]
        xy = sub[["x", "y"]].to_numpy()
        h = sub["h"].to_numpy()
        tree = cKDTree(xy)
        nbrs = tree.query_ball_point(xy, r=radius)
        f = np.zeros(len(sub), bool)
        for i, nb in enumerate(nbrs):
            nb = [j for j in nb if j != i]
            if len(nb) < min_n:
                continue
            v = h[nb]
            med = np.median(v)
            mad = 1.4826 * np.median(np.abs(v - med))
            f[i] = abs(h[i] - med) > max(k * mad, floor_m)
        flag[idx] = f
    return flag


def build(force: bool = False) -> pd.DataFrame:
    f = data_path("labels", "labels_qc.parquet")
    if f.exists() and not force:
        return pd.read_parquet(f)
    cfg = load_config()
    g, gc = qc_gedi(load_raw("gedi"), cfg["gedi"])
    i, ic = qc_icesat2(load_raw("icesat2"), cfg["icesat2"])
    df = pd.concat([g, i], ignore_index=True)
    df["year"] = df["time"].dt.year.astype("int16")
    df["t"] = decimal_year(df["time"])
    df = df[df["year"].isin(cfg["years"])]
    # Grid cells for every analysis grid; drop observations outside the AOI.
    inside = np.ones(len(df), bool)
    for gname in ("hls30", "aef20"):
        grid = get_grid(gname)
        r, c = grid.rowcol(df["x"].to_numpy(), df["y"].to_numpy())
        ok = grid.inside(r, c)
        m = aoi_mask(grid)
        ok[ok] &= m[r[ok], c[ok]]
        inside &= ok
        df[f"row_{gname}"] = r.astype("int32")
        df[f"col_{gname}"] = c.astype("int32")
    counts_aoi = df.groupby("sensor").size().to_dict()
    df = df[inside].reset_index(drop=True)
    df["outlier"] = flag_local_outliers(df)
    df["obs_id"] = np.arange(len(df), dtype=np.int64)
    df.to_parquet(f)
    # QC accounting table (committed).
    rows = [{"sensor": "gedi", "step": k, "n": v} for k, v in gc.items()]
    rows += [{"sensor": "icesat2", "step": k, "n": v} for k, v in ic.items()]
    for s in ("gedi", "icesat2"):
        rows.append({"sensor": s, "step": "years_in_range", "n": counts_aoi.get(s, 0)})
        rows.append({"sensor": s, "step": "inside_aoi", "n": int((df.sensor == s).sum())})
        rows.append({"sensor": s, "step": "not_local_outlier",
                     "n": int(((df.sensor == s) & ~df.outlier).sum())})
    pd.DataFrame(rows).to_csv(results_path("labels", "qc_counts.csv"), index=False)
    summ = (df[~df.outlier].groupby(["sensor", "year", "night", "strong"])
            .agg(n=("h", "size"), h_mean=("h", "mean"), h_median=("h", "median"),
                 h_p90=("h", lambda v: v.quantile(0.9)))
            .reset_index())
    summ.round(3).to_csv(results_path("labels", "summary_by_year.csv"), index=False)
    log.info("labels: %s", df.groupby("sensor").size().to_dict())
    return df


def load(drop_outliers: bool = True) -> pd.DataFrame:
    df = build()
    return df[~df["outlier"]] if drop_outliers else df


def cli(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(prog="azuero labels")
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    df = build(a.force)
    print(df.groupby(["sensor", "year"]).size().unstack(0))
