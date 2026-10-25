"""Cross-validation of the monthly (or 3-monthly) state-space model.

For each spatial fold of the v1 experiment:
  1. fit the optical and radar observation operators on the fold's training lidar only;
  2. at the fold's held-out cells, assemble observations for every month 2017-2025:
       - the v1 annual prediction for that cell/year (from the fold's v1 model),
         placed at July with its model-error variance,
       - optical and radar operator predictions from the monthly composites;
     the held-out lidar is NOT used;
  3. run the Kalman filter/smoother;
  4. score the smoothed height at each held-out observation's month with the same
     height and change metrics as v1.
Variants: monthly vs 3-monthly steps, and "annual only" (the v1 predictions alone run
through the monthly model) to isolate what HLS/S1 add.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..calibration import apply as calibrate
from ..config import data_path, load_config, results_dir, results_path
from ..cv import spatial_folds
from ..labels import load as load_labels
from ..metrics import (block_change, block_change_metrics, change_pairs, height_metrics,
                       pair_change_metrics)
from ..sample import sample_cells
from ..uncertainty import apply_offsets
from . import operators
from .kalman import SSMParams, run as kf_run

log = logging.getLogger(__name__)


def static_features(rows, cols) -> np.ndarray:
    return sample_cells(data_path("interim", "dem", "terrain_hls30.tif"), rows, cols)


def assemble(cells: pd.DataFrame, panel: pd.DataFrame, ops: dict, years, step_months: int,
             cqr_offsets: pd.DataFrame, gedi_sigma: float, use: tuple):
    """Observation arrays (n_cells, T) for the SSM."""
    n = len(cells)
    T = 12 * len(years) // step_months
    z = norm.ppf(0.95)
    obs = {}
    if "annual" in use:
        A = np.full((n, T), np.nan, np.float32)
        V = np.full((n, T), np.nan, np.float32)
        key = cells.set_index(["row", "col"]).index
        for yi, y in enumerate(years):
            p = panel[panel.year == y].set_index(["row", "col"]).reindex(key)
            pred, lo, hi = (p[c].to_numpy() for c in ("pred", "lo", "hi"))
            ok = np.isfinite(pred)
            lo2, hi2 = np.full_like(lo, np.nan), np.full_like(hi, np.nan)
            lo2[ok], hi2[ok] = apply_offsets(pred[ok], lo[ok], hi[ok], cqr_offsets)
            sd = (hi2 - lo2) / (2 * z)
            t_idx = (yi * 12 + 6) // step_months
            A[:, t_idx] = pred
            V[:, t_idx] = np.maximum(sd ** 2 - gedi_sigma ** 2, 1.0)
        obs["annual"] = (A, V)
    rows, cols = cells.row.to_numpy(), cells.col.to_numpy()
    static = static_features(rows, cols)
    for fam in ("optical", "radar"):
        if fam not in use or fam not in ops:
            continue
        feats = operators.sample_all_months(fam, years, rows, cols)       # (n, 12Y, nv)
        pred = ops[fam].predict(feats, static[:, None, :])                 # (n, 12Y)
        if step_months > 1:
            pred = np.nanmean(pred.reshape(n, T, step_months), axis=2)
            var = ops[fam].var_noise / np.maximum(
                np.isfinite(feats[..., 0]).reshape(n, T, step_months).sum(2), 1)
        else:
            var = np.full_like(pred, ops[fam].var_noise)
        var = np.where(np.isfinite(pred), var, np.nan)
        obs[fam] = (pred.astype(np.float32), var.astype(np.float32))
    return obs, T


def score_fold(held: pd.DataFrame, cells: pd.DataFrame, H: np.ndarray, years, step_months):
    pos = pd.Series(np.arange(len(cells)), index=pd.MultiIndex.from_frame(cells[["row", "col"]]))
    ci = pos.reindex(pd.MultiIndex.from_arrays([held.row_hls30, held.col_hls30])).to_numpy()
    m = held.time.dt.month.to_numpy() - 1
    ti = ((held.year.to_numpy() - years[0]) * 12 + m) // step_months
    ok = np.isfinite(ci)
    pred = np.full(len(held), np.nan)
    pred[ok] = H[ci[ok].astype(int), ti[ok]]
    return pred


def run(v1: str, name: str = "ssm", step_months: int = 1, use=("annual", "optical", "radar"),
        params: SSMParams | None = None, chunk: int = 15000, max_cells: int | None = None,
        seed: int = 0) -> dict:
    cfg = load_config()
    years = cfg["years"]
    params = params or SSMParams()
    lab = calibrate(load_labels()).reset_index(drop=True)
    oof = pd.read_parquet(data_path("cv", v1, "oof.parquet"))
    panel = pd.read_parquet(data_path("cv", v1, "panel.parquet"))
    cqr = pd.read_csv(results_path("uncertainty", v1, "cqr_offsets.csv"))
    gedi_sigma = pd.read_csv(results_path("labels", "noise_by_class.csv")).query(
        "sensor == 'gedi'").sigma.mean()
    folds = spatial_folds(lab)
    out = []
    op_info = []
    rng = np.random.default_rng(seed)
    for f in folds:
        held = lab[f.test & lab.obs_id.isin(oof.obs_id)]
        train = lab[f.train].reset_index(drop=True)
        ops = {}
        stat_tr = static_features(train.row_hls30.to_numpy(), train.col_hls30.to_numpy())
        for fam in ("optical", "radar"):
            if fam in use:
                ops[fam] = operators.fit(train, fam, stat_tr, train.sigma.to_numpy())
                op_info.append({"fold": f.name, "family": fam, "noise_sd": np.sqrt(ops[fam].var_noise),
                                "bias_sd": ops[fam].bias_sd})
                params.b0_sd[fam] = ops[fam].bias_sd
        cells = panel[panel.fold == f.name][["row", "col"]].drop_duplicates().reset_index(drop=True)
        if max_cells and len(cells) > max_cells:
            keep = set(map(tuple, held[["row_hls30", "col_hls30"]].drop_duplicates().sample(
                max_cells, random_state=seed).to_numpy()))
            cells = cells[[t in keep for t in map(tuple, cells.to_numpy())]].reset_index(drop=True)
        fp = panel[panel.fold == f.name]
        H = np.full((len(cells), 12 * len(years) // step_months), np.nan, np.float32)
        for c0 in range(0, len(cells), chunk):
            cc = cells.iloc[c0:c0 + chunk].reset_index(drop=True)
            obs, T = assemble(cc, fp, ops, years, step_months, cqr, gedi_sigma, use)
            xs, _ = kf_run(obs, T, step_months / 12, params)
            H[c0:c0 + chunk] = xs[:, :, 0]
        pred = score_fold(held, cells, H, years, step_months)
        out.append(held.assign(pred=pred, fold=f.name))
        log.info("%s %s: cells=%d held=%d rmse=%.2f", name, f.name, len(cells), len(held),
                 np.sqrt(np.nanmean((pred - held.h_cal) ** 2)))
    res = pd.concat(out, ignore_index=True)
    res = res[np.isfinite(res.pred)]
    res.to_parquet(data_path("cv", name, "oof.parquet"))
    s = {"height": height_metrics(res.h_cal, res.pred),
         "height_by_sensor": [{"sensor": k, **height_metrics(g.h_cal, g.pred)}
                              for k, g in res.groupby("sensor")]}
    pairs = change_pairs(res)
    s["pair_change"] = {"all": pair_change_metrics(pairs)}
    s["block_change"] = block_change_metrics(block_change(res))
    s["operators"] = op_info
    s["config"] = {"v1": v1, "step_months": step_months, "use": list(use)}
    d = results_dir("cv", name)
    (d / "summary.json").write_text(json.dumps(s, indent=2, default=float))
    return s


def cli(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(prog="azuero ssm-cv")
    p.add_argument("v1")
    p.add_argument("--name", default="ssm")
    p.add_argument("--step-months", type=int, default=1)
    p.add_argument("--use", nargs="*", default=["annual", "optical", "radar"])
    p.add_argument("--max-cells", type=int, default=None)
    a = p.parse_args(argv)
    s = run(a.v1, a.name, a.step_months, tuple(a.use), max_cells=a.max_cells)
    print(json.dumps({k: s[k] for k in ("height", "pair_change", "block_change")}, indent=1,
                     default=float))
