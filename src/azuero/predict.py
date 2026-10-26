"""Wall-to-wall prediction of the annual canopy height product.

1. Train the final model on all labels (no hold-out) with the chosen configuration.
2. For each year, predict the point estimate and raw 5/95 % quantiles tile by tile.
3. Apply the conformal (CQR) offsets estimated from the spatial-CV out-of-fold
   predictions to get calibrated 90 % intervals, then convert them to SDs.
4. Optionally run the robust Kalman smoother over each pixel's annual series for
   temporal consistency (smoothed heights and annual change). Height SDs are the
   conformal predictive SDs; change SDs are empirical, from held-out lidar pairs.
5. Write full-precision GeoTIFFs (data/product) and compact COGs for the website.

Quality flag: 0 = lidar years (2019-2025), 1 = partial-lidar year (2018),
2 = extrapolated year without lidar (2017), +10 if the AlphaEarth embedding was missing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy.stats import norm

from .aoi import aoi_mask
from .config import data_path, load_config, resources, results_path
from .experiment import Experiment, prepare
from .features import stack_window
from .grid import get_grid
from .models import get_model
from .temporal import SmootherParams, smooth
from .uncertainty import apply_offsets, change_sd_lookup

log = logging.getLogger(__name__)


PRODUCT_SCALE = 0.01
PRODUCT_NODATA = -32768


def read_product(path, band=None) -> np.ndarray:
    """Read product band(s) as float32 metres (NaN for nodata)."""
    with rasterio.open(path) as src:
        a = src.read(band) if band else src.read()
    out = a.astype(np.float32) * PRODUCT_SCALE
    out[a == PRODUCT_NODATA] = np.nan
    return out


def model_path(name: str) -> Path:
    return data_path("models", f"{name}.joblib")


def train_final(exp: Experiment, force: bool = False):
    f = model_path(exp.name)
    if f.exists() and not force:
        return joblib.load(f)
    lab, X = prepare(exp)
    y = lab[exp.target].to_numpy(np.float32)
    w = (1.0 / lab["sigma"].to_numpy() ** 2) if exp.weights else None
    if w is not None:
        w = w / w.mean()
    cap = resources().max_train_rows
    idx = np.arange(len(lab))
    if cap and len(idx) > cap:
        idx = np.random.default_rng(exp.seed).choice(len(idx), cap, replace=False)
    q = load_config()["model"]["quantiles"]
    params = dict(exp.model_params)
    if exp.model == "lightgbm":
        params.setdefault("quantiles", (q[0], q[-1]))
    m = get_model(exp.model, **params)
    m.fit(X.iloc[idx].to_numpy(np.float32), y[idx], None if w is None else w[idx])
    bundle = {"model": m, "columns": list(X.columns), "experiment": exp.__dict__}
    joblib.dump(bundle, f)
    return bundle


def raw_path(name: str, year: int) -> Path:
    return data_path("product", name, "raw", f"pred_{year}.tif")


def predict_year(bundle: dict, exp: Experiment, year: int, force: bool = False) -> Path:
    """Point prediction and raw 5/95 % quantiles for one year (3 float32 bands)."""
    out_f = raw_path(exp.name, year)
    if out_f.exists() and not force:
        return out_f
    grid = get_grid(exp.grid)
    mask = aoi_mask(grid)
    m = bundle["model"]
    cols = bundle["columns"]
    q = load_config()["model"]["quantiles"]
    prof = grid.profile(count=3, dtype="float32", nodata=np.nan, tiled=True, blockxsize=256,
                        blockysize=256, compress="zstd", predictor=3, interleave="band")
    tmp = out_f.with_suffix(".part.tif")
    with rasterio.open(tmp, "w", **prof) as dst:
        dst.descriptions = ("pred", "q05", "q95")
        for r0, c0, h, w in grid.windows(resources().inference_tile_px):
            win = Window(c0, r0, w, h)
            sub = mask[r0:r0 + h, c0:c0 + w]
            block = np.full((3, h, w), np.nan, np.float32)
            if sub.any():
                # The time feature is held at the start of the GEDI era for earlier years:
                # before 2019 only late-2018 ICESat-2 labels exist, so mid-year 2017/2018 on
                # the GEDI standard would fall in tree leaves fitted to no comparable data.
                t_val = max(year + 0.5, load_config()["model"].get("t_min_predict", -np.inf))
                X, names = stack_window(exp.grid, [s for s in exp.sets if s != "obs"], year, win,
                                        t_value=t_val)
                Xd = pd.DataFrame(X, columns=names)
                if exp.obs_covariates:
                    Xd["is_icesat2"], Xd["night"], Xd["strong"] = 0.0, 1.0, 1.0
                Xd = Xd[cols]
                ok = sub.ravel() & np.isfinite(Xd.iloc[:, 0].to_numpy())
                if ok.any():
                    Xa = Xd.to_numpy(np.float32)[ok]
                    p = m.predict(Xa)
                    qq = m.predict_quantiles(Xa, (q[0], q[-1]))
                    for i, v in enumerate((p, qq[:, 0], qq[:, 1])):
                        flat = block[i].ravel()
                        flat[ok] = v
                        block[i] = flat.reshape(h, w)
            dst.write(block, window=win)
        log.info("predicted %s %s", exp.name, year)
    tmp.rename(out_f)
    return out_f


def finalize(name: str, grid_name: str, cqr_name: str, smooth_series: bool = True,
             params: SmootherParams | None = None, force: bool = False) -> list[Path]:
    """Calibrate intervals, smooth over years and write product GeoTIFFs per year.

    Bands: height, lower90, upper90, sd, change (vs previous year), change_sd, flag.
    """
    cfg = load_config()
    years = cfg["years"]
    lidar_years = set(cfg["lidar_years"])
    level = cfg["model"]["interval_level"]
    z = norm.ppf(0.5 + level / 2)
    grid = get_grid(grid_name)
    offsets = pd.read_csv(results_path("uncertainty", cqr_name, "cqr_offsets.csv"))
    outs = {y: data_path("product", name, f"canopy_height_{y}.tif") for y in years}
    if all(f.exists() for f in outs.values()) and not force:
        return list(outs.values())
    srcs = {y: rasterio.open(raw_path(name, y)) for y in years}
    bands = ("height", "lower90", "upper90", "sd", "change", "change_sd", "flag")
    # Stored as int16 with scale 0.01 (cm precision; nodata -32768), which halves the
    # size of float32 without losing meaningful precision. Read with `read_product`.
    prof = grid.profile(count=len(bands), dtype="int16", nodata=PRODUCT_NODATA, tiled=True,
                        blockxsize=256, blockysize=256, compress="zstd", predictor=2,
                        interleave="band")
    dsts = {y: rasterio.open(outs[y].with_suffix(".part.tif"), "w", **prof) for y in years}
    for d in dsts.values():
        d.descriptions = bands
    gedi_sigma = pd.read_csv(results_path("labels", "noise_by_class.csv")).query(
        "sensor == 'gedi'").sigma.mean()
    try:
        for r0, c0, h, w in grid.windows(512):
            win = Window(c0, r0, w, h)
            raw = np.stack([srcs[y].read(window=win) for y in years])  # (T, 3, h, w)
            T = len(years)
            P = raw[:, 0].reshape(T, -1).T
            L = raw[:, 1].reshape(T, -1).T
            U = raw[:, 2].reshape(T, -1).T
            lo, hi = np.full_like(L, np.nan), np.full_like(U, np.nan)
            for t in range(T):
                ok = np.isfinite(P[:, t])
                lo[ok, t], hi[ok, t] = apply_offsets(P[ok, t], L[ok, t], U[ok, t], offsets)
            sd_obs = (hi - lo) / (2 * z)  # predictive SD for a GEDI-equivalent observation
            any_ok = np.isfinite(P).any(1)
            H, S = P.copy(), sd_obs.copy()
            dH = np.full_like(P, np.nan)
            dS = np.full_like(P, np.nan)
            if smooth_series and any_ok.any():
                # Same inputs as evaluated in CV (experiment.smooth_panel): raw quantile SDs
                # with the tuned r_scale.
                s_raw = np.clip((U[any_ok] - L[any_ok]) / (2 * z), 0.5, None)
                h_s, _, _ = smooth(P[any_ok], s_raw, params or SmootherParams.from_config())
                H[any_ok] = h_s
                dH[any_ok, 1:] = np.diff(h_s, axis=1)
            else:
                dH[:, 1:] = np.diff(P, axis=1)
            # Height uncertainty: the conformal (CV-calibrated) predictive SD of each year's
            # prediction. The smoother's posterior SD is not used because annual prediction
            # errors at a pixel are strongly correlated across years, which it cannot see.
            # Change uncertainty: empirical 1-year change SD by height class from held-out
            # lidar pairs (uncertainty.calibrate_change_sd).
            dS[:, 1:] = change_sd_lookup(cqr_name, H[:, 1:])
            for t, y in enumerate(years):
                flag = np.where(y in lidar_years, 0 if y >= 2019 else 1, 2)
                fl = np.full(P.shape[0], flag, np.float32)
                fl[~np.isfinite(P[:, t])] += 10
                H_t = np.clip(H[:, t], 0, None)
                arr = np.stack([H_t, np.clip(H_t - z * S[:, t], 0, None), H_t + z * S[:, t],
                                S[:, t], dH[:, t], dS[:, t], fl]).reshape(len(bands), h, w)
                arr[:, ~np.isfinite(P[:, t]).reshape(h, w) & ~any_ok.reshape(h, w)] = np.nan
                q = np.where(np.isfinite(arr),
                             np.clip(np.round(arr / PRODUCT_SCALE), -32767, 32767), PRODUCT_NODATA)
                dsts[y].write(q.astype(np.int16), window=win)
    finally:
        for d in dsts.values():
            d.close()
        for s in srcs.values():
            s.close()
    for y in years:
        outs[y].with_suffix(".part.tif").rename(outs[y])
        with rasterio.open(outs[y], "r+") as d:
            d.scales = (PRODUCT_SCALE,) * len(bands)
    if resources().delete_raw_after_composite:
        for y in years:
            raw_path(name, y).unlink(missing_ok=True)
    (data_path("product", name) / "product.json").write_text(json.dumps(
        {"grid": grid_name, "years": years, "bands": bands, "interval_level": level,
         "smoothed": smooth_series, "cqr": cqr_name}, indent=2))
    return list(outs.values())


# Web distribution layers: name -> (source band, scale, dtype, nodata, aggregation factor).
# Height keeps the native grid; the SD is spatially smooth and is block-averaged to ~100 m;
# change is block-averaged too (its pixel-level 1-year SD is 6-9 m, so ~100 m is the
# finest meaningful scale); lower/upper bounds are height -/+ 1.645 SD, and change SDs come from the published
# lookup table, so neither is stored separately.
COMPACT = {
    "height": ("height", 0.5, "uint8", 255, 1),
    "sd": ("sd", 0.1, "uint8", 255, 5),
    "change": ("change", 0.1, "int8", -128, 5),
}


def compact_cogs(name: str, out_dir: Path) -> list[Path]:
    """Quantized single-band COGs for web distribution: height, SD and change per year."""
    from rasterio.shutil import copy as rcopy

    out_dir.mkdir(parents=True, exist_ok=True)
    outs = []
    for f in sorted(data_path("product", name).glob("canopy_height_*.tif")):
        year = f.stem.split("_")[-1]
        with rasterio.open(f) as src:
            names = list(src.descriptions)
            prof = src.profile
        for layer, (band, sc, dt, nod, k) in COMPACT.items():
            v = read_product(f, names.index(band) + 1)
            if not np.isfinite(v).any():
                continue
            transform = prof["transform"]
            if k > 1:
                v = block_mean(v, k, min_frac=0.3)
                transform = transform * transform.scale(k, k)
            info = np.iinfo(dt)
            lo = info.min + (1 if nod == info.min else 0)
            hi = info.max - (1 if nod == info.max else 0)
            q = np.where(np.isfinite(v), np.clip(np.round(v / sc), lo, hi), nod).astype(dt)
            p = dict(prof)
            p.update(count=1, dtype=dt, nodata=nod, compress="deflate", zlevel=9, predictor=2,
                     tiled=True, blockxsize=512, blockysize=512, interleave="band",
                     width=q.shape[1], height=q.shape[0], transform=transform)
            p.pop("scales", None)
            tmp = out_dir / f"tmp_{year}_{layer}.tif"
            with rasterio.open(tmp, "w", **p) as dst:
                dst.write(q, 1)
                dst.descriptions = (layer,)
                dst.scales = (sc,)
                dst.update_tags(scale=sc, units="m")
            suffix = "" if layer == "height" else f"_{layer}"
            dst_f = out_dir / f"azuero_canopy_height_{year}{suffix}.tif"
            rcopy(tmp, dst_f, driver="COG", compress="DEFLATE", predictor="2", level=9,
                  overview_resampling="nearest")
            tmp.unlink()
            outs.append(dst_f)
    return outs


def block_mean(a: np.ndarray, k: int, min_frac: float = 0.8) -> np.ndarray:
    """k x k block mean ignoring NaN; blocks with < min_frac valid pixels are NaN."""
    H, W = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    b = a[:H, :W].reshape(H // k, k, W // k, k)
    ok = np.isfinite(b)
    n = ok.sum(axis=(1, 3))
    s = np.where(ok, b, 0).sum(axis=(1, 3))
    with np.errstate(all="ignore"):
        out = s / n
    out[n < min_frac * k * k] = np.nan
    return out


def reference_comparison(name: str, grid_name: str) -> pd.DataFrame:
    """Agreement with independent canopy-height maps after ~1 ha block averaging
    (90 m on the 30 m grid, 100 m on the 20 m grid), using matched product years."""
    g = get_grid(grid_name)
    k = 3 if g.res == 30 else int(round(100 / g.res))
    refs = [("GAO 2012 (airborne lidar + upscaling, 1 ha)", "gao_tch", 2017),
            ("ETH 2020 (Lang et al., 10 m)", "eth_chm", 2020),
            ("Meta/WRI 1 m CHM (~2018-2020)", "meta_chm", 2019)]
    rows = []
    for label, key, year in refs:
        f = data_path("interim", "reference", f"{key}_{grid_name}.tif", mkdir=False)
        if not f.exists():
            continue
        with rasterio.open(f) as src:
            r = src.read(1)
        p = read_product(data_path("product", name, f"canopy_height_{year}.tif"), 1)
        both = np.isfinite(r) & np.isfinite(p)
        rc = block_mean(np.where(both, r, np.nan), k)
        pc = block_mean(np.where(both, p, np.nan), k)
        ok = np.isfinite(rc) & np.isfinite(pc)
        d = pc[ok] - rc[ok]
        rows.append({"reference": label, "product year": year,
                     f"cells ({int(k * g.res)} m)": int(ok.sum()),
                     "mean difference (m)": float(d.mean()),
                     "RMSD (m)": float(np.sqrt((d ** 2).mean())),
                     "r": float(np.corrcoef(pc[ok], rc[ok])[0, 1])})
    df = pd.DataFrame(rows)
    df.to_csv(results_path("product", "reference_comparison.csv"), index=False)
    return df


def cli(argv=None) -> None:
    import argparse

    from .experiment import Experiment

    p = argparse.ArgumentParser(prog="azuero product")
    p.add_argument("name", help="CV experiment whose configuration is used")
    p.add_argument("--product-name", default=None)
    p.add_argument("--no-smooth", action="store_true")
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    cfg_f = results_path("cv", a.name, "summary.json")
    e = json.loads(cfg_f.read_text())["experiment"]
    exp = Experiment(**{**e, "name": a.product_name or f"final_{a.name}"})
    bundle = train_final(exp, a.force)
    for y in load_config()["years"]:
        predict_year(bundle, exp, y, a.force)
    finalize(exp.name, exp.grid, a.name, smooth_series=not a.no_smooth, force=a.force)
    print(product_stats(exp.name).round(2).to_string(index=False))
    print(reference_comparison(exp.name, exp.grid).round(2).to_string(index=False))
