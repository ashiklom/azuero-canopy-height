"""Build the static project website in docs/ from committed results and the product.

Every number shown on the site is read from results/ (or computed from the product
rasters here), so the site always matches the analysis. Steps:
  1. copy assets (CSS, chart kit, map viewer)
  2. static figures (fold maps, lidar density, AlphaEarth PCA)
  3. map overlays (Web Mercator PNGs per year/layer) + manifest + compact COGs
  4. results JSON for the interactive charts
  5. index.html
"""

from __future__ import annotations

import html
import json
import logging
import shutil
from importlib import resources as ires
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT, data_path, load_config, path, results_path

log = logging.getLogger(__name__)

SITE = path("site")


def esc(s) -> str:
    return html.escape(str(s))


def f1(v, d=1):
    try:
        return f"{float(v):.{d}f}"
    except (TypeError, ValueError):
        return "–"


# ----------------------------------------------------------------------------- assets

def copy_assets():
    (SITE / "assets").mkdir(parents=True, exist_ok=True)
    src = ires.files("azuero") / "site_assets"
    for name in ("style.css", "charts.js", "map.js"):
        shutil.copyfile(src / name, SITE / "assets" / name)
    (SITE / ".nojekyll").write_text("")


# ----------------------------------------------------------------------------- figures

def static_figures():
    from .calibration import apply as calibrate
    from .cv import spatial_folds
    from .figures import fold_maps, lidar_density
    from .labels import load

    img = SITE / "img"
    img.mkdir(parents=True, exist_ok=True)
    lab = calibrate(load()).reset_index(drop=True)
    fold_maps(lab, spatial_folds(lab), img / "folds.png")
    lidar_density(lab, img / "lidar_density.png")


# ----------------------------------------------------------------------------- map

def map_layers(product: str, grid_name: str, res_m: float = 100.0):
    """Warp product bands to Web Mercator PNG overlays, with a manifest."""
    import rasterio
    from PIL import Image
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    from rasterio.warp import transform_bounds

    from .figures import DIVERGING, GREEN_SEQ, ORANGE_SEQ, colorize
    from .predict import PRODUCT_NODATA, PRODUCT_SCALE

    out = SITE / "data" / "map"
    out.mkdir(parents=True, exist_ok=True)
    years = load_config()["years"]
    spec = {
        "height": {"band": 1, "kind": "height", "vmin": 0, "vmax": 35, "label": "Canopy height",
                   "unit": "m", "colors": GREEN_SEQ},
        "sd": {"band": 4, "kind": "sd", "vmin": 2, "vmax": 9, "label": "Uncertainty (1 SD)",
               "unit": "m", "colors": ORANGE_SEQ},
        "change": {"band": 5, "kind": "change", "vmin": -3, "vmax": 3,
                   "label": "Change from previous year", "unit": "m", "colors": DIVERGING},
    }
    bounds = None
    for y in years:
        f = data_path("product", product, f"canopy_height_{y}.tif")
        with rasterio.open(f) as src:
            if bounds is None:
                b = transform_bounds(src.crs, "EPSG:3857", *src.bounds)
                w = int((b[2] - b[0]) / res_m)
                hgt = int((b[3] - b[1]) / res_m)
                tr = rasterio.transform.from_bounds(*b, w, hgt)
                ll = transform_bounds("EPSG:3857", "EPSG:4326", *b)
                bounds = [[ll[1], ll[0]], [ll[3], ll[2]]]
            with WarpedVRT(src, crs="EPSG:3857", transform=tr, width=w, height=hgt,
                           resampling=Resampling.average, src_nodata=PRODUCT_NODATA,
                           nodata=np.nan, dtype="float32") as vrt:
                for key, s in spec.items():
                    if key == "change" and y == years[0]:
                        continue
                    a = vrt.read(s["band"]) * PRODUCT_SCALE
                    img = Image.fromarray(colorize(a, s["kind"], s["vmin"], s["vmax"]))
                    # 8-bit palette (64 colours + transparency) keeps overlays small.
                    alpha = img.getchannel("A")
                    pimg = img.convert("RGB").quantize(colors=63, method=Image.Quantize.MEDIANCUT)
                    pal = (pimg.getpalette() or [])[:63 * 3]
                    pal += [0] * (63 * 3 - len(pal)) + [0, 0, 0]  # index 63 = transparent
                    pimg.putpalette(pal)
                    pimg.paste(63, mask=alpha.point(lambda v: 255 if v == 0 else 0))
                    pimg.save(out / f"{key}_{y}.png", optimize=False, transparency=63)
    layers = {}
    for key, s in spec.items():
        layers[key] = {"label": s["label"], "unit": s["unit"], "colors": s["colors"],
                       "min_label": f"{s['vmin']}", "max_label": f"{s['vmax']}+" if key != "change" else f"+{s['vmax']}",
                       "years": years[1:] if key == "change" else years}
    from .predict import COMPACT

    manifest = {"bounds": bounds, "years": years, "layers": layers, "res_m": res_m,
                "cog": {"crs_proj4": "+proj=utm +zone=17 +datum=WGS84 +units=m +no_defs",
                        "files": {k: "data/cogs/azuero_canopy_height_{year}"
                                  + ("" if k == "height" else f"_{k}") + ".tif"
                                  for k in COMPACT},
                        "scale": {k: v[1] for k, v in COMPACT.items()},
                        "nodata": {k: v[3] for k, v in COMPACT.items()}}}
    cfg = load_config()
    manifest["flags"] = {y: (0 if y >= 2019 else 1) if y in cfg["lidar_years"] else 2 for y in years}
    csd = pd.read_csv(results_path("uncertainty", cfg["model"]["final_experiment"], "change_sd.csv"))
    manifest["change_sd"] = [{"lo": max(r.lo_edge, -1e9), "hi": min(r.hi_edge, 1e9), "sd": r.change_sd}
                             for r in csd.itertuples()]
    (out / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def cogs(product: str):
    from .predict import compact_cogs

    out = SITE / "data" / "cogs"
    for f in out.glob("*.tif"):
        f.unlink()
    return compact_cogs(product, out)


# ----------------------------------------------------------------------------- results

def hist2d(obs, pred, edges):
    H, _, _ = np.histogram2d(pred, obs, bins=[edges, edges])
    return H.T.astype(int).tolist()  # rows = obs (y), cols = pred (x)


def results_json(final: str, extra: dict) -> dict:
    """Collect everything the interactive charts need."""
    cfg = load_config()
    R = {}
    summ = pd.read_csv(results_path("labels", "summary_by_year.csv"))
    cnt = summ.groupby(["year", "sensor"]).n.sum().unstack(fill_value=0)
    R["labels_by_year"] = {"years": [int(y) for y in cnt.index],
                           "gedi": cnt.get("gedi", pd.Series(0, cnt.index)).astype(int).tolist(),
                           "icesat2": cnt.get("icesat2", pd.Series(0, cnt.index)).astype(int).tolist()}
    oof = pd.read_parquet(data_path("cv", final, "oof.parquet"))
    edges = np.arange(0, 42, 2.0)
    for s in ("gedi", "icesat2"):
        d = oof[oof.sensor == s]
        R[f"scatter_{s}"] = {"edges": edges.tolist(), "z": hist2d(d.h_cal, d.pred, edges)}
    byc = pd.read_csv(results_path("cv", final, "height_by_class.csv"))
    order = ["(-0.001, 5.0]", "(5.0, 10.0]", "(10.0, 15.0]", "(15.0, 20.0]", "(20.0, 25.0]",
             "(25.0, 30.0]", "(30.0, 80.0]"]
    byc = byc.set_index("hbin").reindex(order)
    R["by_class"] = {"labels": ["0–5", "5–10", "10–15", "15–20", "20–25", "25–30", ">30"],
                     "rmse": byc.rmse.round(2).tolist(), "bias": byc.bias.round(2).tolist(),
                     "n": byc.n.astype(int).tolist()}
    for k, v in extra.items():
        R[k] = v
    return R


def build(final: str, product: str | None = None, cqr: str | None = None,
          skip_heavy: bool = False):
    from .site_page import render

    copy_assets()
    if not skip_heavy:
        static_figures()
        if product:
            map_layers(product, load_config()["primary_grid"])
            cogs(product)
    ctx = render_context(final, product, cqr)
    (SITE / "data").mkdir(parents=True, exist_ok=True)
    (SITE / "data" / "results.json").write_text(json.dumps(ctx["R"], default=float))
    (SITE / "index.html").write_text(render(ctx))
    log.info("site written to %s", SITE)


def render_context(final: str, product: str | None, cqr: str | None) -> dict:
    from .report import collect

    exps = collect()
    extra = {}
    s = json.loads(results_path("cv", final, "summary.json").read_text())
    tv = results_path("cv", final, "temporal_variants.csv")
    extra["temporal"] = pd.read_csv(tv, index_col=0).to_dict() if tv.exists() else {}
    for name in ("base_year", "base_spacetime"):
        f = results_path("cv", name, "height_by_year.csv")
        if f.exists():
            extra[f"{name}_by_year"] = pd.read_csv(f)[["year", "rmse", "bias", "r2", "n"]].to_dict("list")
    extra["space_by_year"] = pd.read_csv(results_path("cv", final, "height_by_year.csv"))[
        ["year", "rmse", "bias", "r2", "n"]].to_dict("list")
    if cqr:
        extra["reliability"] = pd.read_csv(results_path("uncertainty", cqr, "reliability.csv")).to_dict("list")
        extra["cqr"] = json.loads(results_path("uncertainty", cqr, "summary.json").read_text())
    for f, k in ((results_path("product", "stats_by_year.csv"), "product_stats"),
                 (results_path("product", "reference_comparison.csv"), "reference"),
                 (results_path("cv", "av3_compare.csv"), "av3")):
        if f.exists():
            extra[k] = pd.read_csv(f).to_dict("records")
    cd = results_path("cv", "change_direct", "summary.json")
    if cd.exists():
        extra["change_direct"] = json.loads(cd.read_text())["pair_change"]
    R = results_json(final, extra)
    import geopandas as gpd

    aoi_km2 = float(gpd.read_file(results_path("aoi.geojson")).to_crs(
        load_config()["crs"]).area.sum() / 1e6)
    return {"final": final, "summary": s, "experiments": exps, "R": R, "cfg": load_config(),
            "product": product, "aoi_km2": aoi_km2}


def cli(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(prog="azuero site")
    p.add_argument("--final", required=True, help="CV experiment shown as the final model")
    p.add_argument("--product", default=None)
    p.add_argument("--cqr", default=None)
    p.add_argument("--skip-heavy", action="store_true")
    a = p.parse_args(argv)
    build(a.final, a.product, a.cqr or a.final, a.skip_heavy)
