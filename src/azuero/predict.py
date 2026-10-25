"""Wall-to-wall prediction of the annual canopy height product.

1. Train the final model on all labels (no hold-out) with the chosen configuration.
2. For each year, predict the point estimate and raw 5/95 % quantiles tile by tile.
3. Apply the conformal (CQR) offsets estimated from the spatial-CV out-of-fold
   predictions to get calibrated 90 % intervals, then convert them to SDs.
4. Optionally run the robust Kalman smoother over each pixel's annual series to
   enforce temporal consistency. This yields smoothed heights, SDs, annual change
   and change SDs.
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
from .temporal import SmootherParams, change_sd, smooth
from .uncertainty import apply_offsets

log = logging.getLogger(__name__)


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
                X, names = stack_window(exp.grid, [s for s in exp.sets if s != "obs"], year, win)
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
    prof = grid.profile(count=len(bands), dtype="float32", nodata=np.nan, tiled=True,
                        blockxsize=256, blockysize=256, compress="zstd", predictor=3,
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
                # The smoother treats annual predictions as observations of the latent
                # height with the model's error variance (lidar noise removed).
                s_model = np.sqrt(np.maximum(sd_obs[any_ok] ** 2 - gedi_sigma ** 2, 1.0))
                h_s, sd_s, g_s, Ps, lag = smooth(P[any_ok], s_model, params or SmootherParams(),
                                                 return_cov=True)
                H[any_ok] = h_s
                # Report intervals for a GEDI-equivalent measurement: add lidar noise back.
                S[any_ok] = np.sqrt(sd_s ** 2 + gedi_sigma ** 2)
                dH[any_ok, 1:] = np.diff(h_s, axis=1)
                dS[any_ok] = change_sd(Ps, lag)
            else:
                dH[:, 1:] = np.diff(P, axis=1)
                dS[:, 1:] = np.sqrt(sd_obs[:, 1:] ** 2 + sd_obs[:, :-1] ** 2)
            for t, y in enumerate(years):
                flag = np.where(y in lidar_years, 0 if y >= 2019 else 1, 2)
                fl = np.full(P.shape[0], flag, np.float32)
                fl[~np.isfinite(P[:, t])] += 10
                H_t = np.clip(H[:, t], 0, None)
                arr = np.stack([H_t, np.clip(H_t - z * S[:, t], 0, None), H_t + z * S[:, t],
                                S[:, t], dH[:, t], dS[:, t], fl]).reshape(len(bands), h, w)
                arr[:, ~np.isfinite(P[:, t]).reshape(h, w) & ~any_ok.reshape(h, w)] = np.nan
                dsts[y].write(arr.astype(np.float32), window=win)
    finally:
        for d in dsts.values():
            d.close()
        for s in srcs.values():
            s.close()
    for y in years:
        outs[y].with_suffix(".part.tif").rename(outs[y])
    (data_path("product", name) / "product.json").write_text(json.dumps(
        {"grid": grid_name, "years": years, "bands": bands, "interval_level": level,
         "smoothed": smooth_series, "cqr": cqr_name}, indent=2))
    return list(outs.values())


COMPACT = {  # band -> (scale, offset, dtype, nodata)
    "height": (0.25, 0.0, "uint8", 255),
    "lower90": (0.25, 0.0, "uint8", 255),
    "upper90": (0.25, 0.0, "uint8", 255),
    "sd": (0.1, 0.0, "uint8", 255),
    "change": (0.1, 0.0, "int8", -128),
    "change_sd": (0.1, 0.0, "uint8", 255),
    "flag": (1.0, 0.0, "uint8", 255),
}


def compact_cogs(name: str, out_dir: Path) -> list[Path]:
    """Quantized multi-band COGs (uint8/int8) for web distribution, one per year."""
    from rasterio.shutil import copy as rcopy

    out_dir.mkdir(parents=True, exist_ok=True)
    outs = []
    for f in sorted(data_path("product", name).glob("canopy_height_*.tif")):
        year = f.stem.split("_")[-1]
        with rasterio.open(f) as src:
            a = src.read()
            names = src.descriptions
            prof = src.profile
        # One dtype per file is required; store the signed change band in a separate COG.
        for group, members in (("", [n for n in names if n != "change"]), ("_change", ["change"])):
            dt = COMPACT[members[0]][2]
            nod = COMPACT[members[0]][3]
            q = np.full((len(members), *a.shape[1:]), nod, dtype=dt)
            for i, n in enumerate(members):
                sc, off, _, _ = COMPACT[n]
                v = a[names.index(n)]
                info = np.iinfo(dt)
                lim_lo = info.min + (1 if nod == info.min else 0)
                lim_hi = info.max - (1 if nod == info.max else 0)
                qv = np.clip(np.round((v - off) / sc), lim_lo, lim_hi)
                q[i] = np.where(np.isfinite(v), qv, nod).astype(dt)
            p = dict(prof)
            p.update(count=len(members), dtype=dt, nodata=nod, compress="deflate", predictor=2,
                     tiled=True, blockxsize=512, blockysize=512, interleave="band")
            tmp = out_dir / f"tmp_{year}{group}.tif"
            with rasterio.open(tmp, "w", **p) as dst:
                dst.write(q)
                dst.descriptions = tuple(members)
                dst.update_tags(**{f"scale_{n}": COMPACT[n][0] for n in members})
                dst.build_overviews([2, 4, 8, 16], rasterio.enums.Resampling.nearest)
            dst_f = out_dir / f"azuero_canopy_height_{year}{group}.tif"
            rcopy(tmp, dst_f, driver="COG", compress="DEFLATE", predictor="2",
                  overview_resampling="nearest")
            tmp.unlink()
            outs.append(dst_f)
    return outs


def product_stats(name: str) -> pd.DataFrame:
    """Peninsula-wide summaries per year (committed to results/product)."""
    rows = []
    for y in load_config()["years"]:
        with rasterio.open(data_path("product", name, f"canopy_height_{y}.tif")) as src:
            h = src.read(1)
            sd = src.read(4)
            ch = src.read(5)
        ok = np.isfinite(h)
        px_ha = (src.res[0] * src.res[1]) / 1e4
        rows.append({"year": y, "mean_height": float(np.nanmean(h)),
                     "median_height": float(np.nanmedian(h)),
                     "area_ge_10m_kha": float((h[ok] >= 10).sum() * px_ha / 1e3),
                     "area_ge_20m_kha": float((h[ok] >= 20).sum() * px_ha / 1e3),
                     "mean_sd": float(np.nanmean(sd)),
                     "mean_change": float(np.nanmean(ch)) if np.isfinite(ch).any() else np.nan,
                     "area_loss_gt5m_ha": float((ch[np.isfinite(ch)] < -5).sum() * px_ha)})
    df = pd.DataFrame(rows)
    df.round(3).to_csv(results_path("product", "stats_by_year.csv"), index=False)
    return df


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
        with rasterio.open(data_path("product", name, f"canopy_height_{year}.tif")) as src:
            p = src.read(1)
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
