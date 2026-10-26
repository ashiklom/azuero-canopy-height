"""Tune the state-space model's hyperparameters with the same fold split as v1.

For each spatial fold, operators are fit and observations assembled once (on a random
subsample of held-out cells). Then every parameter combination is run through the
Kalman smoother, which is cheap. Parameters are selected on folds 0-2 with the v1
smoother objective (block change r + pair rate r + drop recall - 0.1 * RMSE, on GEDI
night/power-beam labels) and reported on folds 3-4.
"""

from __future__ import annotations

import itertools
import json
import logging
from dataclasses import replace

import numpy as np
import pandas as pd

from ..calibration import apply as calibrate
from ..config import data_path, load_config, results_dir, results_path
from ..cv import spatial_folds
from ..experiment import SUBSETS
from ..labels import load as load_labels
from ..metrics import (block_change, block_change_metrics, change_pairs, height_metrics,
                       pair_change_metrics)
from . import kalman, operators
from .cv import assemble, monthly_features, score_fold
from .kalman import SSMParams

log = logging.getLogger(__name__)

GRID = {
    "jump_types": [("lidar", "annual")],
    "jump_z": [3.0],
    "q_h": [1.0 ** 2],
    "r_opt": [0.25, 0.5, 1.0, 2.0, 4.0],
}


def objective(d: pd.DataFrame) -> dict:
    d = SUBSETS["gedi_night_power"](d)
    d = d[np.isfinite(d.pred)]
    hm = height_metrics(d.h_cal, d.pred)
    pc = pair_change_metrics(change_pairs(d))
    bc = block_change_metrics(block_change(d))
    r = {"rmse": hm["rmse"], "bias": hm["bias"], "pair_rate_r": pc.get("rate_r"),
         "drop_recall": pc.get("drop_recall"), "block_r": bc.get("block_r")}
    r["objective"] = (r["block_r"] or 0) + (r["pair_rate_r"] or 0) + (r["drop_recall"] or 0) \
        - 0.1 * r["rmse"]
    return r


def run(v1: str = "v1_aef20", name: str = "ssm_tuning", step_months: int = 1,
        cells_per_fold: int = 60000, seed: int = 0,
        tune_folds=("space0", "space1", "space2")) -> pd.DataFrame:
    years = load_config()["years"]
    grid = json.loads(results_path("cv", v1, "summary.json").read_text())["experiment"]["grid"]
    lab = calibrate(load_labels()).reset_index(drop=True)
    oof = pd.read_parquet(data_path("cv", v1, "oof.parquet"))
    panel = pd.read_parquet(data_path("cv", v1, "panel.parquet"))
    cqr = pd.read_csv(results_path("uncertainty", v1, "cqr_offsets.csv"))
    gedi_sigma = pd.read_csv(results_path("labels", "noise_by_class.csv")).query(
        "sensor == 'gedi'").sigma.mean()
    rng = np.random.default_rng(seed)
    use = ("annual", "optical", "radar")
    combos = list(itertools.product(*GRID.values()))
    preds = {c: [] for c in combos}
    for f in spatial_folds(lab):
        held = lab[f.test & lab.obs_id.isin(oof.obs_id)]
        train = lab[f.train].reset_index(drop=True)
        from .cv import static_features
        st = static_features(train.row_hls30.to_numpy(), train.col_hls30.to_numpy())
        ops = {fam: operators.fit(train, fam, st, train.sigma.to_numpy()) for fam in ("optical", "radar")}
        cells = panel[panel.fold == f.name][["row", "col"]].drop_duplicates().reset_index(drop=True)
        if len(cells) > cells_per_fold:
            cells = cells.iloc[np.sort(rng.choice(len(cells), cells_per_fold, replace=False))].reset_index(drop=True)
        feats = monthly_features(cells, years, use, grid)
        obs, T = assemble(cells, panel[panel.fold == f.name], ops, years, step_months, cqr,
                          gedi_sigma, use, grid, feats)
        # Operators are anchored per pixel to the v1 level, so the bias states only need to
        # absorb slow drift.
        base = SSMParams(b0_sd={k: 1.0 for k in ops})
        keep = held.set_index([f"row_{grid}", f"col_{grid}"]).index.isin(
            pd.MultiIndex.from_frame(cells[["row", "col"]]))
        held = held[keep]
        for c in combos:
            jt, jz, qh, ro = c
            p = replace(base, jump_types=jt, jump_z=jz, q_h=qh,
                        r_scale={"lidar": 1.0, "annual": 1.0, "optical": ro, "radar": ro})
            xs, _ = kalman.run(obs, T, step_months / 12, p)
            pr = score_fold(held, cells, xs[:, :, 0], years, step_months, grid)
            preds[c].append(held.assign(pred=pr, fold=f.name))
        log.info("tuning fold %s done (%d cells)", f.name, len(cells))
    rows = []
    for c, parts in preds.items():
        d = pd.concat(parts, ignore_index=True)
        for part, sel in (("tune", d.fold.isin(tune_folds)), ("test", ~d.fold.isin(tune_folds))):
            rows.append({"jump_types": "+".join(c[0]), "jump_z": c[1], "q_h": c[2], "r_opt": c[3],
                         "part": part, **objective(d[sel])})
    out = pd.DataFrame(rows)
    out.to_csv(results_dir("cv", name) / "ssm_tuning.csv", index=False)
    return out


def cli(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(prog="azuero ssm-tune")
    p.add_argument("--v1", default="v1_aef20")
    p.add_argument("--name", default="ssm_tuning")
    p.add_argument("--step-months", type=int, default=1)
    p.add_argument("--cells", type=int, default=60000)
    a = p.parse_args(argv)
    df = run(a.v1, a.name, a.step_months, a.cells)
    t = df[df.part == "tune"].sort_values("objective", ascending=False)
    print(t.head(8).round(3).to_string(index=False))
