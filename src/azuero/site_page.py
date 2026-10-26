"""HTML for the project website. All numbers come from the build context (results/)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import results_path
from .site import esc, f1


def _tbl(df: pd.DataFrame, highlight: str | None = None, key: str | None = None) -> str:
    head = "".join(f"<th>{esc(c)}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cls = ' class="highlight"' if highlight is not None and key and r[key] == highlight else ""
        cells = []
        for c in df.columns:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append(f"<td>{'–' if not np.isfinite(v) else f'{v:.2f}'}</td>")
            else:
                cells.append(f"<td>{esc(v)}</td>")
        rows.append(f"<tr{cls}>{''.join(cells)}</tr>")
    return (f'<div class="tablewrap"><table class="data"><thead><tr>{head}</tr></thead>'
            f"<tbody>{''.join(rows)}</tbody></table></div>")


EXPERIMENT_LABELS = {
    "base_aef_hls30": "AlphaEarth (30 m)",
    "cal_none": "AlphaEarth, no sensor calibration",
    "cal_offset": "AlphaEarth, offset calibration",
    "cal_deming": "AlphaEarth, Deming calibration",
    "noweights": "AlphaEarth, unweighted",
    "obscov": "AlphaEarth + observation covariates",
    "aef_dem": "AlphaEarth + terrain",
    "aef_time": "AlphaEarth + time trend",
    "gedi_only": "AlphaEarth, GEDI labels only",
    "mlp": "AlphaEarth, MLP instead of LightGBM",
    "ctx": "AlphaEarth + 90 m context",
    "ctx_dem": "AlphaEarth + context + terrain",
    "ctx_dem_hls_s1": "AlphaEarth + context + terrain + HLS + S1",
    "ctx_dem_hls": "AlphaEarth + context + terrain + HLS",
    "ctx_dem_s1": "AlphaEarth + context + terrain + S1",
    "base_aef20": "AlphaEarth (20 m native grid)",
    "ctx_aef20": "AlphaEarth + context (20 m native grid)",
    "aef_av3": "AlphaEarth + AVIRIS-3",
    "base_year": "AlphaEarth, leave-one-year-out",
    "base_spacetime": "AlphaEarth, space × time hold-out",
    "obscov_time": "AlphaEarth + time + observation covariates",
    "obscov_time_dem": "AlphaEarth + time + terrain + observation covariates (30 m)",
    "v1_hls30": "Final features on the 30 m HLS grid",
    "v1_aef20": "Final: AlphaEarth + time + terrain + observation covariates (20 m)",
    "v1_hls_s1": "Final + HLS + Sentinel-1 (20 m)",
    "base_block5": "AlphaEarth, 5 km blocks",
    "base_block25": "AlphaEarth, 25 km blocks",
}


def render(ctx: dict) -> str:
    s = ctx["summary"]
    R = ctx["R"]
    cfg = ctx["cfg"]
    final = ctx["final"]
    exps: pd.DataFrame = ctx["experiments"]
    hs = {r["sensor"]: r for r in s["height_by_sensor"]}
    g = hs.get("gedi", {})
    var = s.get("variants", {})
    cqr = R.get("cqr", {})
    nlab = sum(R["labels_by_year"]["gedi"]) + sum(R["labels_by_year"]["icesat2"])
    years = cfg["years"]

    # ---- tables
    qc = pd.read_csv(results_path("labels", "qc_counts.csv"))
    noise = pd.read_csv(results_path("labels", "noise_by_class.csv"))
    cal = pd.read_csv(results_path("labels", "calibration.csv")).query("pairs == 'stable_pairs'")
    screen = pd.read_csv(results_path("labels", "pair_screening.csv"), index_col=0)["value"]
    thr = pd.read_csv(results_path("labels", "aef_stability_threshold.csv")).iloc[0]

    noise_t = noise.assign(
        sensor=noise.sensor.map({"gedi": "GEDI", "icesat2": "ICESat-2"}),
        illumination=np.where(noise.night, "night", "day"),
        beam=np.where(noise.strong, "strong/power", "weak/coverage"))[
        ["sensor", "illumination", "beam", "n_pairs", "sigma"]].rename(
        columns={"n_pairs": "stable repeat pairs", "sigma": "noise σ (m)"})
    cal_t = cal.assign(illumination=np.where(cal.night, "night", "day"),
                       beam=np.where(cal.strong, "strong", "weak"))[
        ["illumination", "beam", "n", "median_diff_raw", "r_raw", "a", "b"]].rename(
        columns={"n": "pairs", "median_diff_raw": "median ICESat-2 − GEDI (m)",
                 "r_raw": "r", "a": "Deming a", "b": "Deming b"})

    ex = exps.copy()
    ex["label"] = ex["name"].map(EXPERIMENT_LABELS).fillna(ex["name"])
    space = ex[(ex.design == "space") & ex.block_km.isna()]
    abl = space[["label", "name", "gedi_rmse", "gedi_bias", "gedi_r2", "icesat2_rmse",
                 "block_r"]].sort_values("gedi_rmse")
    abl_t = abl.rename(columns={"label": "configuration", "gedi_rmse": "RMSE vs GEDI (m)",
                                "gedi_bias": "bias vs GEDI (m)", "gedi_r2": "R² vs GEDI",
                                "icesat2_rmse": "RMSE vs ICESat-2 (m)",
                                "block_r": "block change r"})
    abl_html = _tbl(abl_t.drop(columns=["name"]), EXPERIMENT_LABELS.get(final, final),
                    "configuration")
    des = ex[ex.name.isin(["base_aef_hls30", "base_block5", "base_block25", "base_year",
                           "base_spacetime"])][["label", "all_rmse", "gedi_rmse", "gedi_r2"]]
    des_html = _tbl(des.rename(columns={"label": "design", "all_rmse": "RMSE all labels (m)",
                                        "gedi_rmse": "RMSE vs GEDI (m)", "gedi_r2": "R² vs GEDI"}))

    def pc(v, k):
        return f1(v.get(k), 2) if v else "–"
    vkey = "panel_smooth|gedi_night_power"
    smv = var.get(vkey, {})
    cd = R.get("change_direct", {}).get("all", {})
    labels_v = {"oof": "Annual model, conditioned on each observation's sensor/date",
                "panel_raw": "Annual map predictions (standardized), unsmoothed",
                "panel_smooth": "Annual map predictions, Kalman-smoothed (the product)"}
    labels_s = {"all": "all labels", "gedi": "GEDI", "gedi_night_power": "GEDI night, power beams"}
    change_rows = []
    for vk, vv in var.items():
        v, sub = vk.split("|")
        change_rows.append((f"{labels_v.get(v, v)} — {labels_s.get(sub, sub)}", vv.get("height", {}),
                            vv.get("pair_change", {}), vv.get("block_change", {})))
    change_rows.append(("Direct change model trained on lidar pairs — all labels", {}, cd, {}))
    ch_html = "".join(
        f"<tr><td>{esc(n)}</td><td>{pc(h, 'rmse')}</td><td>{pc(p, 'rate_r')}</td><td>{pc(p, 'rate_rmse')}</td>"
        f"<td>{pc(p, 'drop_recall')}</td><td>{pc(p, 'drop_precision')}</td>"
        f"<td>{pc(b, 'block_r')}</td><td>{pc(b, 'block_rmse')}</td></tr>"
        for n, h, p, b in change_rows)
    allv = var.get("panel_smooth|all", {})
    n_pairs = allv.get("pair_change", {}).get("n_pairs", 0)
    n_blocks = allv.get("block_change", {}).get("block_n", 0)
    diag = pd.read_csv(results_path("labels", "screening_diagnostic.csv"))
    diag_html = _tbl(diag.rename(columns={"median_abs_diff_m": "median |ICESat-2 − GEDI| (m)",
                                          "mean_abs_diff_m": "mean |ICESat-2 − GEDI| (m)"}))
    scr_all = pd.read_csv(results_path("labels", "pair_screening_allchecks.csv"), index_col=0)["value"]
    csd = pd.read_csv(results_path("uncertainty", final, "change_sd.csv"))
    csd_t = pd.DataFrame({"canopy height (m)": [f"{a:g}–{b:g}".replace("-inf–", "< ").replace("–inf", "+")
                                                for a, b in zip(csd.lo_edge, csd.hi_edge)],
                          "held-out pairs": csd.n_pairs, "1-year change SD (m)": csd.change_sd})
    csd_html = _tbl(csd_t)
    tun = results_path("cv", final, "smoother_tuning.csv")
    sm_cfg = cfg["model"].get("smoother", {})
    ssm_rows = []
    for nm, lab in (("ssm_monthly", "Monthly SSM (v1 annual + HLS + S1)"),
                    ("ssm_3monthly", "3-monthly SSM (v1 annual + HLS + S1)"),
                    ("ssm_annual_only", "Monthly SSM, v1 annual predictions only")):
        f = results_path("cv", nm, "summary_subsets.json")
        if f.exists():
            for sub, r in json.loads(f.read_text()).items():
                if sub == "gedi_night_power":
                    ssm_rows.append({"model": lab, "RMSE (m)": r["rmse"], "bias (m)": r["bias"],
                                     "pair rate r": r["pair_rate_r"], "drop recall": r["drop_recall"],
                                     "block change r": r["block_r"]})
    if smv:
        ssm_rows.insert(0, {"model": "v1: Kalman-smoothed annual predictions",
                            "RMSE (m)": smv["height"]["rmse"], "bias (m)": smv["height"]["bias"],
                            "pair rate r": smv["pair_change"].get("rate_r"),
                            "drop recall": smv["pair_change"].get("drop_recall"),
                            "block change r": smv["block_change"].get("block_r")})
    ssm_html = _tbl(pd.DataFrame(ssm_rows)) if len(ssm_rows) > 1 else ""
    ssm_tune = R.get("ssm_tuning_best")
    noise_floor = float(np.sqrt(2) * noise.query("sensor == 'gedi'").sigma.mean())

    av3 = pd.DataFrame(R.get("av3", []))
    av3_html = ""
    if len(av3):
        t = av3[av3.sensor == "all"][["subset", "model", "n", "rmse", "bias", "r2"]]
        av3_html = _tbl(t.rename(columns={"rmse": "RMSE (m)", "bias": "bias (m)", "r2": "R²"}))
    ref = pd.DataFrame(R.get("reference", []))
    ref_html = _tbl(ref) if len(ref) else "<p class='muted'>Not available.</p>"
    stats = pd.DataFrame(R.get("product_stats", []))

    data_json = json.dumps(R, default=float)
    final_label = EXPERIMENT_LABELS.get(final, final)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Azuero canopy height 2017–2025</title>
<link rel="stylesheet" href="assets/style.css">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js" defer></script>
<script src="https://unpkg.com/geotiff@2.1.3/dist-browser/geotiff.js" defer></script>
<script src="https://unpkg.com/proj4@2.12.1/dist/proj4.js" defer></script>
<script src="assets/charts.js" defer></script>
<script src="assets/map.js" defer></script>
</head>
<body>
<header class="site"><div class="inner">
  <span class="brand">Azuero canopy height</span>
  <nav>
    <a href="#map-section">Map</a><a href="#data">Data</a><a href="#methods">Methods</a>
    <a href="#results">Results</a><a href="#change">Change</a><a href="#uncertainty">Uncertainty</a>
    <a href="#limitations">Limitations</a><a href="#downloads">Downloads</a>
  </nav>
  <button class="theme" id="theme-btn" type="button">Dark mode</button>
</div></header>
<main>

<section id="top">
<h1>Annual canopy height for the Azuero Peninsula, Panama, 2017–2025</h1>
<p class="lede">An annual, gridded {int(ctx['cfg']['grids'][ctx['cfg']['primary_grid']]['res'])} m
canopy-height product with per-pixel uncertainty for the Azuero Peninsula
({f1(ctx['aoi_km2'], 0)} km²). Spaceborne lidar from GEDI and ICESat-2 supplies the labels:
{nlab:,} quality-screened observations, harmonized to the GEDI rh98 scale. The
<a href="https://deepmind.google/discover/blog/alphaearth-foundations-helps-map-our-planet-in-unprecedented-detail/">AlphaEarth
Foundations</a> annual satellite embeddings, plus supporting predictors, map those labels
across the peninsula. Accuracy comes from cross-validation that holds out whole
contiguous regions and whole years. It is a lower bound for the final product, which
uses all the data.</p>
<div class="tiles">
  <div class="tile"><div class="label">Height RMSE vs held-out GEDI</div><div class="value">{f1(g.get('rmse'))} m</div><div class="note">spatial-region CV, n = {int(g.get('n', 0)):,}; bias {f1(g.get('bias'), 2)} m</div></div>
  <div class="tile"><div class="label">R² vs held-out GEDI</div><div class="value">{f1(g.get('r2'), 2)}</div><div class="note">the GEDI noise floor alone is σ ≈ {f1(noise.query("sensor == 'gedi'").sigma.mean())} m</div></div>
  <div class="tile"><div class="label">90 % interval coverage</div><div class="value">{f1(100 * cqr.get('coverage_crossfit', np.nan), 1)} %</div><div class="note">conformal, cross-fitted over folds</div></div>
  <div class="tile"><div class="label">Year-to-year change, 2 km blocks</div><div class="value">r = {f1(smv.get('block_change', {}).get('block_r'), 2)}</div><div class="note">smoothed map vs held-out GEDI night power-beam shots</div></div>
</div>
<p class="small muted">Final configuration: {esc(final_label)}. Built from commit-tracked results in
<code>results/</code>; code at <a href="https://github.com/ashiklom/azuero-canopy-height">github.com/ashiklom/azuero-canopy-height</a>.</p>
</section>

<section id="map-section">
<h2>Explore the product</h2>
<p>Pick a year and a layer. Click anywhere to read the 20 m pixel values, which are read
directly from the product's Cloud-Optimized GeoTIFFs. Change is the smoothed height
minus the previous year's smoothed height. Uncertainty is the 1 SD of a GEDI-equivalent
measurement at the cell (conformally calibrated).</p>
<div class="map-controls">
  <label for="map-layer">Layer</label><select id="map-layer"></select>
  <label for="map-year">Year</label><input type="range" id="map-year" step="1"><strong id="map-year-label"></strong>
  <label for="map-opacity">Opacity</label><input type="range" id="map-opacity" min="0" max="1" step="0.05" value="0.85">
  <span class="scale-legend" id="map-legend"></span>
</div>
<div id="map" role="application" aria-label="Canopy height map"></div>
<p class="small muted">Overlays are 100 m averages for display. The downloadable rasters are at full resolution.</p>
</section>

<section id="data">
<h2>Data</h2>
<h3>Lidar labels</h3>
<p><strong>GEDI L2A v002</strong> (2019–2023, then from 2024 after the ISS storage period).
Shots are retrieved with <a href="https://slideruleearth.io">SlideRule Earth</a>
(<code>gedi02ap</code>) with the full relative-height profile and quality fields. We keep
shots with <code>quality_flag = 1</code>, <code>degrade_flag = 0</code>, the surface flag
set, sensitivity ≥ 0.95 (≥ 0.98 for coverage beams), at least one detected mode,
agreement with the reference DEM within 50 m, and urban fraction below 50 %. The
target is rh98.</p>
<p><strong>ICESat-2</strong> (from October 2018). Instead of the standard 100 m ATL08
segments, SlideRule's <code>atl03x</code> endpoint runs the PhoREAL canopy algorithm on
custom <strong>30 m along-track extents</strong> (<code>len = res = 30 m</code>). This
matches the analysis grid. It uses the ATL08 photon classes (ground, canopy,
top-of-canopy) with a 1 m vertical bin, relative heights, and the extent centre for
geolocation. We require at least 10 photons including at least 3 ground photons, no
processing flags and no water. The 98th-percentile canopy relief (<code>h_canopy</code>)
is the target. Comparisons with GEDI showed that the ATL08 multiple-scattering (cloud)
flag does <em>not</em> degrade agreement, so it is not used as a filter.</p>
<p>Observations far from the robust local consensus are flagged and excluded from
training. The test is |h − median| &gt; max(3·MAD, 10 m) among same-sensor,
same-year neighbours within 100 m.</p>
<div id="fig-labels"></div>
<img class="figure" src="img/lidar_density.png" alt="Maps of lidar observation density for GEDI and ICESat-2 over the Azuero Peninsula">
<h3>Quality screening</h3>
{_tbl(qc.rename(columns={'n': 'observations remaining'}))}

<h3>Predictors</h3>
<ul>
<li><strong>AlphaEarth Foundations annual embeddings</strong> (Google / Google DeepMind,
CC-BY 4.0). These are 64-dimensional unit vectors per 10 m pixel and year, 2017–2025,
summarizing Sentinel-1/2, Landsat and other inputs. We read the
<a href="https://source.coop/tge-labs/aef">Source Cooperative copy</a> directly. The
copy is stored bottom-up, so rows are flipped on read. The <code>small</code> profile
reads the 20 m overview level. Other grids are derived by exact area-weighted
averaging. Earth Engine was not available.</li>
<li><strong>Context embedding</strong>: the AlphaEarth vector averaged over a ~90 m
neighbourhood. It adds spatial context and absorbs GEDI's ~10 m geolocation error.</li>
<li><strong>Terrain</strong> (Copernicus GLO-30): elevation, slope, northness/eastness,
topographic position at 150 m and 1 km, and 1 km relief.</li>
<li><strong>Harmonized Landsat Sentinel-2 (HLS v2.0)</strong>. Fmask-screened monthly
medians of red, NIR, SWIR1, SWIR2, NDVI, NBR and NDMI are summarized per year
(p10/p50/p90, dry-season medians). The monthly composites also feed the stability
screening and the state-space model.</li>
<li><strong>Sentinel-1 RTC</strong> γ⁰ VV/VH monthly medians and annual summaries.
NASA's OPERA RTC-S1 was the first choice, but ASF downloads were refused for the
account used here. Data come from the Microsoft Planetary Computer
<code>sentinel-1-rtc</code> collection instead.</li>
<li><strong>AVIRIS-3 imaging spectroscopy</strong> from NASA's
<a href="https://www.earthdata.nasa.gov/data/projects/avuelo">AVUELO</a> campaign
(8 and 11 February 2025): 94 orthorectified L2A reflectance scenes at 3.4 m over
Azuero. They are aggregated to the grid, binned to 30 nm, and reduced to 10 principal
components plus NDVI, NDRE, PRI, NDWI and the cellulose absorption index. They are
tested as an experiment because they cover only part of the peninsula on one date.</li>
</ul>
<p><strong>What the final model uses.</strong> Cross-validation (below) settled on the
following features:</p>
<ul>
<li>the AlphaEarth embedding of the observation's year;</li>
<li>the observation's decimal year;</li>
<li>terrain;</li>
<li>three observation covariates: ICESat-2 vs GEDI, night vs day, and strong vs weak
beam.</li>
</ul>
<p>The map is predicted on a single measurement standard: a GEDI power-beam shot at
night, mid-year. Context embeddings, HLS, Sentinel-1 and AVIRIS-3 features did not
improve held-out accuracy. They are reported as ablations. HLS and Sentinel-1 still feed
the stability screening and the version-2 state-space model.</p>
<h3>Evaluated but not used</h3>
<ul>
<li><strong>LVIS</strong> and other airborne lidar: none over Azuero in NASA Earthdata.</li>
<li><strong>UAVSAR</strong>: two L-band lines (Cerro Hoya, May 2026), after the product
period.</li>
<li><strong>EMIT</strong>: 109 granules, mostly cloudy, at 60 m. Not used because
AVIRIS-3 covers the same need at much finer resolution.</li>
<li><strong>In-situ plots</strong>: no openly available Azuero plot inventory with tree
heights was found. The PRORENA Los Santos plantation trial and STRI plots are
described in publications, but no data deposits were found.</li>
<li><strong>Reference maps</strong> are used only for comparison:
<a href="https://zenodo.org/records/4624240">Global Airborne Observatory</a> 2012 1 ha
canopy height (airborne lidar + upscaling), ETH 2020 10 m (Lang et al.), and the
Meta/WRI 1 m CHM (Tolan et al.). Hansen GFC v1.13 loss years are used for stability
screening.</li>
</ul>
</section>

<section id="methods">
<h2>Methods</h2>
<h3>Area and grids</h3>
<p>The area of interest is the union of the Herrera and Los Santos provinces with the
Mariato district and mainland Montijo (Veraguas), from geoBoundaries ADM2, without
islands. All analysis grids are in UTM 17N, with pixel edges on integer multiples of
their resolution. <code>hls30</code> (30 m) coincides with the HLS/MGRS grid, and
<code>aef20</code> (20 m) with AlphaEarth's native pixel edges at its first overview.
A native 10 m grid (<code>aef10</code>) is configured for the <code>full</code> profile.</p>

<h3>Sensor harmonization with stability screening</h3>
<p>GEDI and ICESat-2 measure canopy height differently: full-waveform rh98 versus
photon-counting relief. ICESat-2 heights are mapped onto the GEDI scale using
near-coincident pairs (within 20 m and 1 year). A pair is only useful for
calibration if the canopy did not change between the two acquisitions, so each pair is
screened with independent evidence over the years it spans:</p>
<ul>
<li>the minimum cosine similarity between consecutive AlphaEarth embeddings, flagged
below {f1(thr.threshold, 3)}. That is the 2nd percentile in stable reference forest
(GFC tree cover ≥ 90 %, no loss, ETH height ≥ 20 m);</li>
<li>Hansen GFC loss within the span;</li>
<li>annual HLS NBR changes &gt; 0.10 and Sentinel-1 VH changes &gt; 1.5 dB.</li>
</ul>
<p>The product's calibration uses the AlphaEarth and GFC checks. Of
{int(screen.get('n_pairs', 0)):,} cross-sensor pairs,
{f1(100 * screen.get('flag_aef', 0), 1)} % were flagged by AlphaEarth and
{f1(100 * screen.get('flag_gfc', 0), 1)} % by GFC loss, and
{f1(100 * screen.get('stable', 0), 1)} % were kept. Adding the HLS and Sentinel-1 checks
flags more pairs ({f1(100 * scr_all.get('flag_hls', 0), 1)} % and
{f1(100 * scr_all.get('flag_s1', 0), 1)} %), keeping {f1(100 * scr_all.get('stable', 0), 1)} %.
But the flagged pairs do <em>not</em> disagree more than unflagged pairs; only the few
GFC-loss pairs do. Within a year, real canopy change is rare compared with measurement
noise and geolocation mismatch. The +0.3–0.4 m shift that the stricter screen produces
in the calibration is selection noise, so the product keeps the simpler screen:</p>
{diag_html}
<p> Measurement noise per sensor and
class comes from stable same-sensor repeat pairs, using Var(h₁ − h₂) = 2σ²:</p>
{_tbl(noise_t)}
<p>Night acquisitions are much less noisy for ICESat-2, and daytime solar background
inflates its canopy heights. Three calibrations were compared per class:</p>
<ul>
<li>an additive offset;</li>
<li>quantile mapping of collocated distributions;</li>
<li>Deming (errors-in-variables) regression with the variance ratio set from the noise
estimates.</li>
</ul>
<p>Cross-validation selected quantile mapping (see the table below). Deming slopes are
poorly constrained for the noisy daytime classes:</p>
{_tbl(cal_t)}

<h3>Model</h3>
<p>One row per lidar observation, with features from that observation's calendar year
at its grid cell. LightGBM gradient boosting (600 trees, 127 leaves) predicts the
harmonized height. The observation covariates absorb sensor- and illumination-specific
differences that the calibration leaves behind. The time feature captures a trend that
the embeddings miss: models that never saw a year are biased by +1.0 m in 2020 and
−0.6 m in 2023–2025. For years before 2019 the time feature is held at 2019.3, because the
only earlier labels are late-2018 ICESat-2; extrapolating it produced an artificial
coastal "loss" from 2018 to 2019. Those years differ only through their embeddings. Separate quantile models give raw 5 % and 95 % quantiles. Up to
400 000 rows are sampled per fit in the <code>small</code> profile. A scikit-learn MLP
was also tested. Larger models are implemented in <code>azuero.models.registry</code>
but were not run on this VM: a spatial-context U-Net on embedding/HLS patches, a
temporal transformer over monthly HLS/S1 sequences, and a deep ensemble with
heteroscedastic heads. They run automatically under the <code>full</code> profile with
a GPU.</p>

<h3>Cross-validation</h3>
<p><strong>Spatial-region CV.</strong> The peninsula is tiled into {cfg['cv']['block_km']} km
blocks. Occupied blocks are clustered into {cfg['cv']['n_folds'] * cfg['cv']['regions_per_fold']}
contiguous regions, and each of the {cfg['cv']['n_folds']} folds holds out
{cfg['cv']['regions_per_fold']} non-adjacent regions. Training labels within
{cfg['cv']['buffer_km']} km of any held-out label are dropped. Block sizes of 5 and
25 km were run as sensitivity checks. <strong>Leave-one-year-out</strong> holds out all
lidar from one year. <strong>Space × time</strong> holds out a region-fold in one year
and trains on neither the region nor the year.</p>
<img class="figure" src="img/folds.png" alt="Five small maps, one per fold, showing held-out lidar observations in blue">

<h3>Accuracy metrics</h3>
<p><strong>Height</strong>: RMSE, MAE, bias (prediction − observation) and R² against
held-out lidar. GEDI-only metrics are the headline, because GEDI is the reference scale
and its test labels do not depend on the calibration choice.
<strong>Change</strong> (the growth-rate metric) is scored three ways:</p>
<ul>
<li><em>pair change</em>: same-sensor observations within 20 m and ≥ 0.75 yr apart,
comparing observed and predicted rates in m/yr;</li>
<li><em>block change</em>: per 2 km block and consecutive-year pair, the change in mean
observed height versus the change in mean predicted height <em>at the same
observations</em>. Sample matching cancels footprint-placement differences;</li>
<li><em>disturbance detection</em>: recall and precision for observed drops &gt; 10 m.</li>
</ul>
<p>Change is judged correct if the difference is right, even if the absolute baseline
is biased.</p>

<h3>Uncertainty</h3>
<p><strong>Height.</strong> Raw quantiles are calibrated with Mondrian conformalized
quantile regression. The offsets are computed per bin of predicted height from
out-of-fold residuals and cross-fitted, so reported coverage is out of sample. The
interval refers to a GEDI-equivalent rh98 measurement at the cell, so it includes lidar
measurement noise. The product's SD is this calibrated SD. The Kalman smoother's own
posterior SD is <em>not</em> used: it assumes independent annual errors, while a
pixel's errors are strongly correlated across years, so it would be overconfident.</p>
<p><strong>Change.</strong> The SD of a 1-year change is estimated empirically from
held-out lidar pairs 0.75–1.5 years apart, scored on the smoothed map predictions.
Each pair's own label noise (class-specific σ) is subtracted:</p>
{csd_html}

<h3>Temporal consistency</h3>
<p>Annual predictions are combined per pixel with a robust Kalman filter and
Rauch–Tung–Striebel smoother:</p>
<ul>
<li>the state is a local linear trend (height, growth rate);</li>
<li>observation variances come from the calibrated intervals, minus GEDI noise;</li>
<li>a large negative innovation (&gt; 2.5σ) inflates the process variance for that step,
so clearing appears as a jump instead of being smoothed away.</li>
</ul>
<p>For evaluation, the smoother runs on out-of-fold <em>panel</em> predictions: every
year at every held-out cell, predicted by that fold's model on the product's measurement
standard. Its parameters were tuned on folds 0–2 and checked on folds 3–4:
q<sub>h</sub> = {sm_cfg.get('q_h')}, q<sub>g</sub> = {sm_cfg.get('q_g')}, jump threshold
{sm_cfg.get('jump_z')}σ, observation-variance scale {sm_cfg.get('r_scale')}. The objective
was nearly flat across the grid.</p>

<h3>Resource profiles</h3>
<p>Everything specific to this small VM (4 CPUs, 15 GB RAM, ~30 GB disk) is a separate
toggle in <code>config.yaml</code>. The <code>small</code>/<code>full</code> profile sets
them all at once. The toggles:</p>
<ul>
<li>deleting raw downloads after compositing;</li>
<li>the AlphaEarth read level (20 m overview vs native 10 m);</li>
<li>one-year feature stacks;</li>
<li>workers and threads (LightGBM is limited to 2 threads to avoid OpenMP
oversubscription);</li>
<li>training subsample size;</li>
<li>HLS cloud and band limits;</li>
<li>Sentinel-1 scene caps;</li>
<li>GPU and large-model switches.</li>
</ul>
</section>

<section id="results">
<h2>Results: height</h2>
<div class="grid2"><div id="fig-scatter-gedi"></div><div id="fig-scatter-is2"></div></div>
<div id="fig-byclass"></div>
<p>Like every regression-based canopy-height map, the model compresses the extremes:
it overestimates short canopies and underestimates the tallest (&gt; 30 m)
canopies. Part of this is regression dilution from label noise (σ ≈ 3 m for GEDI). The
bias by class shows how much.</p>
<h3>Ablations (spatial-region CV)</h3>
{abl_html}
<p class="small muted">Rows are sorted by RMSE against held-out GEDI. The final
configuration is in bold. Block change r is the correlation of year-to-year changes in
2 km block means (raw annual predictions).</p>
<h3>CV design and temporal transfer</h3>
{des_html}
<div id="fig-byyear"></div>
<p>Leave-one-year-out shows how well the model predicts a year it has never seen. This
is also how 2017, which has no lidar at all, is mapped. 2018 has only three months of
ICESat-2.</p>
{('<h3>AVIRIS-3 (AVUELO) experiment</h3>' + av3_html + '<p>Both models are scored on the same held-out 2025 lidar. The comparison is reported separately inside and outside the February 2025 flight coverage.</p>') if av3_html else ''}
<h3>Comparison with other canopy-height maps</h3>
{ref_html}
</section>

<section id="change">
<h2>Results: change</h2>
<p>Annual height change is a much harder target than height. Two GEDI shots within
20 m of each other already differ by a noise SD of about {f1(noise_floor)} m, while
typical regrowth is 0.5–2 m/yr. Pair-level skill is therefore limited by the labels as
much as by the model. Block-level change averages much of that noise out.</p>
<p>The rows that describe the <strong>map</strong> are the standardized ("panel")
predictions. The first rows score the annual model with each observation's own
sensor, illumination and date. They show higher pair-level change skill, but part of that
comes from changes in measurement conditions between the two observations, not from
canopy change. The fairest test of the product is the smoothed panel on GEDI night
power-beam shots.</p>
<div class="tablewrap"><table class="data"><thead><tr><th>predictions — labels</th><th>height RMSE (m)</th><th>pair rate r</th><th>pair rate RMSE (m/yr)</th><th>drop recall</th><th>drop precision</th><th>block change r</th><th>block change RMSE (m)</th></tr></thead>
<tbody>{ch_html}</tbody></table></div>
<p class="small muted">On all labels there are {n_pairs:,} held-out lidar pairs and
{n_blocks:,} block-year pairs. The direct change model (embedding differences →
Δheight) is scored on the same all-label pairs. It reaches the best pair-level rate
correlation, but it is not used for the map because its changes would not be consistent
with the annual heights. Large drops (&gt; 10 m) are rarely recovered by any method:
annual predictions understate clearings.</p>
<div id="fig-area"></div>
<div id="fig-stats"></div>
</section>

<section id="uncertainty">
<h2>Results: uncertainty</h2>
<div id="fig-reliability"></div>
<p>The raw LightGBM 90 % quantile intervals covered {f1(100 * cqr.get('coverage_raw', np.nan), 1)} %
of held-out labels. After conformal calibration the cross-fitted coverage is
{f1(100 * cqr.get('coverage_crossfit', np.nan), 1)} %, with a mean width of
{f1(cqr.get('width_crossfit'))} m. Coverage is lower for the tallest observed canopies,
which is the regression-to-the-mean effect seen in the height results.</p>
</section>

<section id="v2">
<h2>Version 2: monthly state-space model</h2>
<p>After version 1, a per-pixel <strong>monthly</strong> state-space model was built
(<code>azuero.ssm</code>). Its state is canopy height, growth rate, and two slowly
varying bias terms for the optical and radar observation operators. It takes four kinds
of observations:</p>
<ul>
<li><em>lidar</em> (GEDI / ICESat-2, calibrated): identity observation operator, with
the class-specific label noise;</li>
<li>the <em>v1 annual prediction</em>: placed at July, with its model-error variance;</li>
<li><em>HLS</em> monthly composites: mapped to height by a LightGBM operator trained on
lidar from the same month;</li>
<li><em>Sentinel-1</em> monthly composites: mapped the same way.</li>
</ul>
<p>The operators' errors are split into a persistent per-pixel bias, estimated from
residual pairs at the same place in different months, and month-to-month noise. That
way the filter does not treat 108 correlated monthly errors as independent evidence.
Disturbances use the same robust jump mechanism as version 1. In cross-validation, the
held-out cells' lidar is withheld and the smoothed monthly height is scored at each
held-out observation's month.</p>
{ssm_html or '<p class="muted">Results pending.</p>'}
<p class="small muted">All rows are scored on held-out GEDI night power-beam shots,
with no held-out lidar used by any model.</p>
</section>

<section id="limitations">
<h2>Limitations</h2>
<ul>
<li>The cross-validated accuracy is a lower bound for mapping unseen regions. It is not
an accuracy for each pixel in each year.</li>
<li>There is no independent airborne lidar or field data for Azuero, so "truth" is
spaceborne lidar with σ ≈ 2–4 m noise and ~10 m geolocation error.</li>
<li>AlphaEarth embeddings are produced by a model whose training may have included GEDI.
The spatial hold-outs still test generalization to unseen places.</li>
<li>2017 has no lidar and 2018 has only October–December ICESat-2. Their quality
flags (2 and 1) mark that, and their time feature is held at the 2019 value.</li>
<li>Pixel-level annual change is very uncertain (1 SD ≈ 6–9 m for one year). Use block
or landscape aggregates, or multi-year differences, for growth rates. Clearings are
under-detected.</li>
<li>The 20 m pixels are finer than a GEDI footprint (25 m) or an ICESat-2 segment
(30 m along track). The effective resolution is closer to the lidar footprint.</li>
</ul>
</section>

<section id="downloads">
<h2>Downloads and reproducibility</h2>
<p>The yearly products are Cloud-Optimized GeoTIFFs in UTM 17N (EPSG:32617). Three
files per year, with the scale factor stored in each file:</p>
<ul>
<li><code>azuero_canopy_height_YEAR.tif</code>: smoothed canopy height (rh98-equivalent)
on the 20 m AlphaEarth grid, uint8 × 0.5 m, nodata 255;</li>
<li><code>azuero_canopy_height_YEAR_sd.tif</code>: calibrated 1 SD of height, averaged to
100 m, uint8 × 0.1 m. The 90 % interval is height ± 1.645 SD;</li>
<li><code>azuero_canopy_height_YEAR_change.tif</code>: change from the previous year
(smoothed), int8 × 0.1 m, nodata −128. Its 1 SD is given in the change-uncertainty table
above.</li>
</ul>
<p>Quality flag by year: 0 = lidar year (2019–2025), 1 = partial lidar year (2018),
2 = no lidar (2017). The full-precision rasters (int16 × 0.01 m, all bands) are
regenerated with <code>azuero product v1_aef20</code>.</p>
<ul>{''.join(f'<li>{y}: <a href="data/cogs/azuero_canopy_height_{y}.tif">height</a> · <a href="data/cogs/azuero_canopy_height_{y}_sd.tif">SD</a>' + (f' · <a href="data/cogs/azuero_canopy_height_{y}_change.tif">change</a>' if y != years[0] else '') + '</li>' for y in years)}</ul>
<pre>uv venv &amp;&amp; uv pip install -e '.[dev]'
azuero aoi &amp;&amp; azuero lidar &amp;&amp; azuero alphaearth &amp;&amp; azuero aefctx &amp;&amp; azuero dem
azuero hls &amp;&amp; azuero s1 &amp;&amp; azuero airborne &amp;&amp; azuero reference
azuero labels &amp;&amp; azuero calibrate
azuero cv v1_aef20 --grid aef20 --sets aef time dem --obs-covariates   # final config CV
azuero product v1_aef20                       # train on all data, predict, calibrate, smooth
azuero ssm-cv v1_aef20 --name ssm_monthly     # version 2
azuero site --final v1_aef20 --product final_v1_aef20</pre>
</section>

<footer>
<p>Data credits: The AlphaEarth Foundations Satellite Embedding dataset is produced by
Google and Google DeepMind (CC-BY 4.0). GEDI, ICESat-2, HLS, AVIRIS-3/AVUELO: NASA (via
NASA Earthdata and SlideRule Earth). Sentinel-1 RTC: ESA Copernicus via Microsoft
Planetary Computer. Copernicus DEM GLO-30 © DLR/Airbus. Hansen GFC v1.13 (UMD).
Global Airborne Observatory Panama map (Asner et al., CC-BY 4.0). ETH Global Canopy
Height 2020 (Lang et al.). Meta/WRI CHM (Tolan et al.). Boundaries: geoBoundaries
(STRI).</p>
</footer>
</main>
<script>window.RESULTS = {data_json};</script>
<script>
document.addEventListener("DOMContentLoaded", () => {{
  const R = window.RESULTS, V = window.Viz;
  V.themeToggle(document.getElementById("theme-btn"));
  V.columns(document.getElementById("fig-labels"), {{
    title: "Quality-screened lidar observations per year", subtitle: "inside the AOI, outliers removed",
    categories: R.labels_by_year.years.map(String),
    series: [{{name: "GEDI", values: R.labels_by_year.gedi}}, {{name: "ICESat-2 (30 m)", values: R.labels_by_year.icesat2}}],
    xTitle: "Year", yTitle: "Observations", digits: 0 }});
  for (const [id, key, name] of [["fig-scatter-gedi", "scatter_gedi", "GEDI"], ["fig-scatter-is2", "scatter_icesat2", "ICESat-2 (on GEDI scale)"]]) {{
    V.heatmap(document.getElementById(id), {{ title: `Held-out ${{name}} vs. predicted`, subtitle: "spatial-region CV; line = 1:1",
      xEdges: R[key].edges, yEdges: R[key].edges, z: R[key].z, xTitle: "Predicted (m)", yTitle: "Observed (m)", diag: true }});
  }}
  V.lines(document.getElementById("fig-byclass"), {{
    title: "Error by observed height class", subtitle: "all held-out labels",
    x: R.by_class.labels.map((_, i) => i), xTicks: R.by_class.labels.map((_, i) => i),
    xFmt: (i) => R.by_class.labels[i], xUnit: " m",
    series: [{{name: "RMSE (m)", values: R.by_class.rmse}}, {{name: "Bias, predicted − observed (m)", values: R.by_class.bias}}],
    xTitle: "Observed canopy height (m)", yTitle: "m", hline: 0 }});
  const yrs = R.space_by_year.year;
  const pick = (o) => o ? yrs.map((y) => {{ const i = o.year.indexOf(y); return i < 0 ? null : o.rmse[i]; }}) : yrs.map(() => null);
  const ser = [{{name: "Spatial-region CV", values: R.space_by_year.rmse}}];
  if (R.base_year_by_year) ser.push({{name: "Leave-one-year-out", values: pick(R.base_year_by_year)}});
  if (R.base_spacetime_by_year) ser.push({{name: "Space × time", values: pick(R.base_spacetime_by_year)}});
  V.lines(document.getElementById("fig-byyear"), {{ title: "RMSE by year and CV design", subtitle: "all held-out labels",
    x: yrs, xTicks: yrs, series: ser, xTitle: "Year", yTitle: "RMSE (m)", xFmt: (v) => String(v) }});
  if (R.reliability) V.lines(document.getElementById("fig-reliability"), {{
    title: "Interval reliability", subtitle: "empirical vs nominal coverage (Gaussian intervals from calibrated SDs); line = ideal",
    x: R.reliability.nominal, series: [{{name: "Empirical coverage", values: R.reliability.empirical}}],
    xTitle: "Nominal coverage", yTitle: "Empirical coverage", diag: true, xMin: 0.4, xMax: 1, yMin: 0.4, yMax: 1 }});
  if (R.product_stats) V.lines(document.getElementById("fig-area"), {{
    title: "Area with tall canopy", subtitle: "final product; thousand hectares",
    x: R.product_stats.map((d) => d.year), xTicks: R.product_stats.map((d) => d.year), xFmt: (v) => String(v),
    series: [{{name: "Canopy ≥ 10 m", values: R.product_stats.map((d) => d.area_ge_10m_kha)}}, {{name: "Canopy ≥ 20 m", values: R.product_stats.map((d) => d.area_ge_20m_kha)}}],
    xTitle: "Year", yTitle: "Area (kha)" }});
  if (R.product_stats) V.lines(document.getElementById("fig-stats"), {{
    title: "Mean canopy height across the peninsula", subtitle: "final product, smoothed",
    x: R.product_stats.map((d) => d.year), xTicks: R.product_stats.map((d) => d.year), xFmt: (v) => String(v),
    series: [{{name: "Mean height (m)", values: R.product_stats.map((d) => d.mean_height)}}],
    xTitle: "Year", yTitle: "Mean canopy height (m)" }});
}});
</script>
</body>
</html>
"""
