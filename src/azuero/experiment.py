"""Cross-validation experiments: fit on training folds, predict held-out lidar, score.

An experiment is defined by the grid, feature sets, model, CV design and label options.
Out-of-fold (OOF) predictions are saved to `data/cv/<name>/oof.parquet`. The metric
tables go to `results/cv/<name>/` and are committed, because the website is built
from them.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from . import cv as cvmod
from .calibration import apply as calibrate
from .config import data_path, load_config, resources, results_dir, results_path
from .features import training_table
from .labels import load as load_labels
from .metrics import (block_change, block_change_metrics, change_pairs, grouped,
                      height_metrics, interval_metrics, pair_change_metrics)
from .models import get_model

log = logging.getLogger(__name__)

HEIGHT_BINS = [0, 5, 10, 15, 20, 25, 30, 80]


@dataclass
class Experiment:
    name: str
    grid: str = "hls30"
    sets: list = field(default_factory=lambda: ["aef"])
    model: str = "lightgbm"
    design: str = "space"            # space | year | spacetime
    target: str = "h_cal"            # calibrated height (h for raw)
    calibration: str = "qmap"        # none | offset | qmap | deming
    sensors: list = field(default_factory=lambda: ["gedi", "icesat2"])
    weights: bool = True             # inverse label-noise variance weights
    obs_covariates: bool = False     # add sensor/night/strong as features
    block_km: float | None = None
    model_params: dict = field(default_factory=dict)
    seed: int = 0
    panel: bool = True               # predict all years at held-out cells (for smoothing)


def prepare(exp: Experiment) -> tuple[pd.DataFrame, pd.DataFrame]:
    lab = calibrate(load_labels(), exp.calibration)
    lab = lab[lab.sensor.isin(exp.sensors)].reset_index(drop=True)
    X = training_table(lab, exp.grid, [s for s in exp.sets if s != "obs"])
    if exp.obs_covariates:
        X["is_icesat2"] = (lab.sensor == "icesat2").astype(np.float32).to_numpy()
        X["night"] = lab.night.astype(np.float32).to_numpy()
        X["strong"] = lab.strong.astype(np.float32).to_numpy()
    # Require a valid embedding / primary predictor.
    key = "A00" if "aef" in exp.sets else X.columns[0]
    ok = X[key].notna().to_numpy()
    return lab[ok].reset_index(drop=True), X[ok].reset_index(drop=True)


def folds_for(exp: Experiment, lab: pd.DataFrame):
    kw = {"block_km": exp.block_km} if exp.block_km else {}
    if exp.design == "space":
        return cvmod.spatial_folds(lab, **kw)
    if exp.design == "year":
        return cvmod.year_folds(lab)
    if exp.design == "spacetime":
        return cvmod.space_time_folds(lab, **kw)
    raise ValueError(exp.design)


def run(exp: Experiment, force: bool = False) -> dict:
    out_dir = results_dir("cv", exp.name)
    oof_f = data_path("cv", exp.name, "oof.parquet")
    if oof_f.exists() and (out_dir / "summary.json").exists() and not force:
        return json.loads((out_dir / "summary.json").read_text())
    t0 = time.time()
    lab, X = prepare(exp)
    y = lab[exp.target].to_numpy(np.float32)
    w = (1.0 / lab["sigma"].to_numpy() ** 2) if exp.weights else None
    if w is not None:
        w = w / w.mean()
    folds = folds_for(exp, lab)
    q = load_config()["model"]["quantiles"]
    pred = np.full(len(lab), np.nan)
    lo = np.full(len(lab), np.nan)
    hi = np.full(len(lab), np.nan)
    fold_name = np.full(len(lab), "", dtype=object)
    rng = np.random.default_rng(exp.seed)
    cap = resources().max_train_rows
    importances = []
    panels = []
    years_all = load_config()["years"]
    for f in folds:
        tr = np.nonzero(f.train)[0]
        if cap and tr.size > cap:
            tr = rng.choice(tr, cap, replace=False)
        params = dict(exp.model_params)
        if exp.model == "lightgbm":
            params.setdefault("quantiles", (q[0], q[-1]))
        m = get_model(exp.model, **params)
        m.fit(X.iloc[tr].to_numpy(np.float32), y[tr], None if w is None else w[tr])
        te = np.nonzero(f.test)[0]
        Xte = X.iloc[te].to_numpy(np.float32)
        pred[te] = m.predict(Xte)
        qq = m.predict_quantiles(Xte, (q[0], q[-1]))
        lo[te], hi[te] = qq[:, 0], qq[:, -1]
        fold_name[te] = f.name
        if exp.panel and exp.design == "space":
            panels.append(panel_predict(m, exp, lab.iloc[te], X.columns, years_all, f.name, q))
        if hasattr(m, "feature_importance"):
            importances.append(m.feature_importance(list(X.columns)))
        log.info("%s fold %s: n_train=%d n_test=%d rmse=%.2f", exp.name, f.name, tr.size,
                 te.size, np.sqrt(np.nanmean((pred[te] - y[te]) ** 2)))
    oof = lab.assign(pred=pred, lo=lo, hi=hi, fold=fold_name)
    oof = oof[np.isfinite(oof.pred)]
    oof.to_parquet(oof_f)
    summary = score(oof, out_dir, exp.target)
    if panels:
        panel = pd.concat(panels, ignore_index=True)
        panel.to_parquet(data_path("cv", exp.name, "panel.parquet"))
        summary["variants"] = score_temporal_variants(oof, panel, out_dir, exp.grid)
    summary["experiment"] = asdict(exp)
    summary["minutes"] = (time.time() - t0) / 60
    summary["n_folds"] = len(folds)
    if importances:
        imp = pd.DataFrame(importances).mean().sort_values(ascending=False)
        imp.rename("gain_share").to_csv(out_dir / "feature_importance.csv")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


def panel_predict(model, exp: Experiment, test_lab: pd.DataFrame, columns, years, fold, q
                  ) -> pd.DataFrame:
    """Predictions for every year at the unique held-out cells of a fold."""
    rc, cc = f"row_{exp.grid}", f"col_{exp.grid}"
    cells = test_lab[[rc, cc]].drop_duplicates().reset_index(drop=True)
    frames = []
    for y in years:
        d = cells.copy()
        d["year"] = y
        d["t"] = y + 0.5
        d["obs_id"] = -1
        frames.append(d)
    panel = pd.concat(frames, ignore_index=True)
    Xp = training_table(panel, exp.grid, [s for s in exp.sets if s != "obs"], cache=False)
    if exp.obs_covariates:  # predict on the GEDI power-beam, night scale
        Xp["is_icesat2"] = 0.0
        Xp["night"] = 1.0
        Xp["strong"] = 1.0
    Xp = Xp[list(columns)]
    ok = Xp.iloc[:, 0].notna().to_numpy()
    pred = np.full(len(panel), np.nan)
    lo = np.full(len(panel), np.nan)
    hi = np.full(len(panel), np.nan)
    Xa = Xp[ok].to_numpy(np.float32)
    pred[ok] = model.predict(Xa)
    qq = model.predict_quantiles(Xa, (q[0], q[-1]))
    lo[ok], hi[ok] = qq[:, 0], qq[:, -1]
    return panel.assign(pred=pred, lo=lo, hi=hi, fold=fold)[
        [rc, cc, "year", "pred", "lo", "hi", "fold"]].rename(columns={rc: "row", cc: "col"})


def smooth_panel(panel: pd.DataFrame, years, params=None) -> pd.DataFrame:
    """Apply the robust Kalman smoother to each cell's annual series."""
    from scipy.stats import norm

    from .temporal import SmootherParams, smooth

    level = load_config()["model"]["interval_level"]
    z = norm.ppf(0.5 + level / 2)
    wide = panel.pivot_table(index=["row", "col"], columns="year", values=["pred", "lo", "hi"])
    P = wide["pred"].reindex(columns=years).to_numpy()
    S = ((wide["hi"] - wide["lo"]) / (2 * z)).reindex(columns=years).to_numpy()
    h, sd, g = smooth(P, np.clip(S, 0.5, None), params or SmootherParams.from_config())
    out = pd.DataFrame(h, index=wide.index, columns=years).stack().rename("pred_smooth")
    return out.reset_index().rename(columns={"level_2": "year"})


SUBSETS = {
    "all": lambda d: d,
    "gedi": lambda d: d[d.sensor == "gedi"],
    # The product is predicted on this measurement standard, so it is the fairest test
    # of mapped heights and changes.
    "gedi_night_power": lambda d: d[(d.sensor == "gedi") & d.night & d.strong],
}


def score_temporal_variants(oof, panel, out_dir, grid) -> dict:
    """Score three prediction variants on three label subsets.

    oof          out-of-fold predictions conditioned on each observation's own sensor,
                 illumination, beam and date (best for "what would this sensor see")
    panel_raw    standardized annual predictions (GEDI power beam, night, mid-year) at
                 held-out cells, i.e. what the map shows before temporal smoothing
    panel_smooth the same after the robust Kalman smoother (the product)
    Change metrics on `oof` partly reflect changing measurement conditions between the
    two observations; the panel variants isolate canopy change.
    """
    years = load_config()["years"]
    sm = smooth_panel(panel, years)
    keys = [f"row_{grid}", f"col_{grid}", "year"]
    m = oof.merge(panel[["row", "col", "year", "pred"]].rename(
        columns={"row": keys[0], "col": keys[1], "pred": "pred_panel"}), on=keys, how="left")
    m = m.merge(sm.rename(columns={"row": keys[0], "col": keys[1]}), on=keys, how="left")
    res, rows = {}, []
    for name, col in (("oof", "pred"), ("panel_raw", "pred_panel"), ("panel_smooth", "pred_smooth")):
        for sub, fn in SUBSETS.items():
            d = fn(m).assign(pred=lambda x: x[col])
            d = d[np.isfinite(d.pred)]
            pairs = change_pairs(d)
            r = {"height": height_metrics(d.h_cal, d.pred),
                 "pair_change": pair_change_metrics(pairs) if len(pairs) else {},
                 "block_change": block_change_metrics(block_change(d))}
            res[f"{name}|{sub}"] = r
            rows.append({"variant": name, "subset": sub, **r["height"],
                         **{f"pair_{a}": b for a, b in r["pair_change"].items()},
                         **r["block_change"]})
    pd.DataFrame(rows).to_csv(out_dir / "temporal_variants.csv", index=False)
    return res


def rescore_temporal(name: str, grid: str) -> dict:
    """Recompute temporal variants from saved OOF/panel predictions."""
    oof = pd.read_parquet(data_path("cv", name, "oof.parquet"))
    panel = pd.read_parquet(data_path("cv", name, "panel.parquet"))
    out_dir = results_dir("cv", name)
    v = score_temporal_variants(oof, panel, out_dir, grid)
    sf = out_dir / "summary.json"
    s = json.loads(sf.read_text())
    s["variants"] = v
    sf.write_text(json.dumps(s, indent=2, default=float))
    return v


def score(oof: pd.DataFrame, out_dir, target: str = "h_cal") -> dict:
    """Height, interval and change metrics for a set of OOF predictions."""
    oof = oof.copy()
    if target != "h_cal":
        oof["h_cal"] = oof[target]
    s = {"height": height_metrics(oof.h_cal, oof.pred),
         "interval": interval_metrics(oof.h_cal, oof.lo, oof.hi)}
    s["height_by_sensor"] = grouped(oof, "sensor").to_dict("records")
    oof["hbin"] = pd.cut(oof.h_cal, HEIGHT_BINS, include_lowest=True).astype(str)
    grouped(oof, "sensor").to_csv(out_dir / "height_by_sensor.csv", index=False)
    grouped(oof, "year").to_csv(out_dir / "height_by_year.csv", index=False)
    grouped(oof, "hbin").to_csv(out_dir / "height_by_class.csv", index=False)
    grouped(oof, "fold").to_csv(out_dir / "height_by_fold.csv", index=False)
    # Interval coverage by height class.
    cov = [{"hbin": k, **interval_metrics(g.h_cal, g.lo, g.hi)} for k, g in oof.groupby("hbin")]
    pd.DataFrame(cov).to_csv(out_dir / "interval_by_class.csv", index=False)
    # Change.
    pairs = change_pairs(oof)
    pc = {}
    if len(pairs):
        for sname, g in [("all", pairs)] + list(pairs.groupby("sensor")):
            pc[sname] = pair_change_metrics(g)
    s["pair_change"] = pc
    bc = block_change(oof)
    s["block_change"] = block_change_metrics(bc)
    bc.to_csv(out_dir / "block_change.csv", index=False)
    if len(pairs):
        pairs.to_parquet(data_path("cv", out_dir.name, "pairs.parquet"))
    # Scatter sample for the website.
    oof.sample(min(len(oof), 5000), random_state=0)[
        ["sensor", "year", "h_cal", "pred", "lo", "hi"]].round(2).to_csv(
        out_dir / "scatter_sample.csv", index=False)
    return s


def cli(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(prog="azuero cv")
    p.add_argument("name")
    p.add_argument("--grid", default="hls30")
    p.add_argument("--sets", nargs="*", default=["aef"])
    p.add_argument("--model", default="lightgbm")
    p.add_argument("--design", default="space")
    p.add_argument("--target", default="h_cal")
    p.add_argument("--sensors", nargs="*", default=["gedi", "icesat2"])
    p.add_argument("--no-weights", action="store_true")
    p.add_argument("--obs-covariates", action="store_true")
    p.add_argument("--block-km", type=float, default=None)
    p.add_argument("--calibration", default="qmap")
    p.add_argument("--no-panel", action="store_true")
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    exp = Experiment(a.name, a.grid, a.sets, a.model, a.design, a.target, a.calibration,
                     a.sensors, not a.no_weights, a.obs_covariates, a.block_km,
                     panel=not a.no_panel)
    s = run(exp, a.force)
    print(json.dumps({k: s[k] for k in ("height", "interval", "pair_change", "block_change")},
                     indent=1, default=float))


def tune_smoother(name: str, grid: str, tune_folds=("space0", "space1", "space2")) -> pd.DataFrame:
    """Grid-search smoother parameters on some spatial folds, report on the others.

    Objective (on GEDI night/power-beam labels): block change r + pair rate r +
    drop recall - 0.1 * height RMSE, which balances height and change accuracy.
    """
    import itertools

    from .temporal import SmootherParams

    years = load_config()["years"]
    oof = pd.read_parquet(data_path("cv", name, "oof.parquet"))
    panel = pd.read_parquet(data_path("cv", name, "panel.parquet"))
    oof = SUBSETS["gedi_night_power"](oof)
    keys = [f"row_{grid}", f"col_{grid}", "year"]
    cells = oof[keys[:2]].drop_duplicates().rename(columns={keys[0]: "row", keys[1]: "col"})
    panel = panel.merge(cells, on=["row", "col"])
    rows = []
    grid_ = itertools.product([0.25 ** 2, 0.5 ** 2, 1.0 ** 2], [0.1 ** 2, 0.3 ** 2],
                              [2.0, 2.5, 3.0], [0.5, 1.0])
    for qh, qg, jz, rs in grid_:
        p = SmootherParams(q_h=qh, q_g=qg, jump_z=jz, r_scale=rs)
        sm = smooth_panel(panel, years, p).rename(columns={"row": keys[0], "col": keys[1]})
        m = oof.merge(sm, on=keys, how="left")
        for part, sel in (("tune", m.fold.isin(tune_folds)), ("test", ~m.fold.isin(tune_folds))):
            d = m[sel].assign(pred=lambda x: x.pred_smooth)
            d = d[np.isfinite(d.pred)]
            hm = height_metrics(d.h_cal, d.pred)
            pc = pair_change_metrics(change_pairs(d))
            bc = block_change_metrics(block_change(d))
            rows.append({"q_h": qh, "q_g": qg, "jump_z": jz, "r_scale": rs, "part": part,
                         "rmse": hm["rmse"], "bias": hm["bias"], "pair_rate_r": pc.get("rate_r"),
                         "drop_recall": pc.get("drop_recall"), "block_r": bc.get("block_r")})
    df = pd.DataFrame(rows)
    df["objective"] = df.block_r + df.pair_rate_r + df.drop_recall - 0.1 * df.rmse
    df.to_csv(results_dir("cv", name) / "smoother_tuning.csv", index=False)
    return df
