"""Direct change model (approach c): predict height change between two dates directly.

Training examples are same-sensor lidar pairs at (nearly) the same location (within 20 m)
observed at least 0.75 years apart. Features are the AlphaEarth embeddings of both years
(and their difference) plus the time gap. The target is h2 - h1. A model that differences
two annual predictions inherits the errors of both. A direct model sees only paired
information and can learn that most places do not change.

It is evaluated on the same spatial folds as the annual models: a pair is held out when
its first observation is in the held-out region. The scores are directly comparable to
the `pair_change` metrics of the annual models (see `metrics.pair_change_metrics`).
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from .calibration import apply as calibrate
from .config import data_path, resources, results_dir
from .cv import spatial_folds
from .features import training_table
from .labels import load as load_labels
from .metrics import change_pairs, pair_change_metrics
from .models import get_model

log = logging.getLogger(__name__)


def build_pairs(lab: pd.DataFrame) -> pd.DataFrame:
    d = lab.assign(pred=np.nan)
    p = change_pairs(d)
    return p


def run(name: str = "change_direct", grid: str = "hls30", sets=("aef",),
        calibration: str = "qmap") -> dict:
    lab = calibrate(load_labels(), calibration).reset_index(drop=True)
    X = training_table(lab, grid, list(sets))
    ok = X.iloc[:, 0].notna().to_numpy()
    lab, X = lab[ok].reset_index(drop=True), X[ok].reset_index(drop=True)
    pos = pd.Series(np.arange(len(lab)), index=lab.obs_id.to_numpy())
    pairs = build_pairs(lab)
    i1 = pos.loc[pairs.obs1].to_numpy()
    i2 = pos.loc[pairs.obs2].to_numpy()
    A1, A2 = X.to_numpy(np.float32)[i1], X.to_numpy(np.float32)[i2]
    F = np.hstack([A1, A2 - A1, pairs[["dt"]].to_numpy(np.float32)])
    y = (pairs.h2 - pairs.h1).to_numpy(np.float32)
    folds = spatial_folds(lab)
    pred = np.full(len(pairs), np.nan)
    cap = resources().max_train_rows
    rng = np.random.default_rng(0)
    for f in folds:
        te = f.test[i1]
        tr = f.train[i1] & f.train[i2]
        tri = np.nonzero(tr)[0]
        if cap and tri.size > cap:
            tri = rng.choice(tri, cap, replace=False)
        m = get_model("lightgbm", quantiles=(0.05, 0.95), n_estimators=400)
        m.fit(F[tri], y[tri])
        pred[te] = m.predict(F[te])
        log.info("%s %s: train=%d test=%d", name, f.name, tri.size, te.sum())
    pairs["d_pred"] = pred
    pairs["rate_pred"] = pred / pairs.dt
    pairs = pairs[np.isfinite(pairs.d_pred)]
    out = {"all": pair_change_metrics(pairs)}
    for s, g in pairs.groupby("sensor"):
        out[s] = pair_change_metrics(g)
    d = results_dir("cv", name)
    (d / "summary.json").write_text(json.dumps({"pair_change": out}, indent=2, default=float))
    pairs.to_parquet(data_path("cv", name, "pairs.parquet"))
    return out


def cli(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(prog="azuero change-model")
    p.add_argument("--name", default="change_direct")
    p.add_argument("--sets", nargs="*", default=["aef"])
    a = p.parse_args(argv)
    print(json.dumps(run(a.name, sets=a.sets), indent=1, default=float))
