"""Tabulate all CV experiments into results/cv/experiments.csv."""

from __future__ import annotations

import json

import pandas as pd

from .config import results_path


def collect() -> pd.DataFrame:
    rows = []
    for f in sorted(results_path("cv", mkdir=False).glob("*/summary.json")):
        s = json.loads(f.read_text())
        e = s.get("experiment", {})
        r = {"name": f.parent.name, "grid": e.get("grid"), "sets": "+".join(e.get("sets", [])),
             "model": e.get("model"), "design": e.get("design"),
             "calibration": e.get("calibration", "qmap"), "weights": e.get("weights"),
             "sensors": "+".join(e.get("sensors", [])), "obs_covariates": e.get("obs_covariates"),
             "block_km": e.get("block_km")}
        if "height" in s:
            r.update({f"all_{k}": v for k, v in s["height"].items() if k in ("n", "rmse", "mae", "bias", "r2")})
        for g in s.get("height_by_sensor", []):
            r.update({f"{g['sensor']}_{k}": g.get(k) for k in ("rmse", "bias", "r2")})
        if "interval" in s:
            r["coverage_raw"] = s["interval"]["coverage"]
        pc = s.get("pair_change", {}).get("all", {})
        r.update({"pair_rate_r": pc.get("rate_r"), "pair_rate_rmse": pc.get("rate_rmse"),
                  "drop_recall": pc.get("drop_recall"), "drop_precision": pc.get("drop_precision")})
        bc = s.get("block_change", {})
        r.update({"block_r": bc.get("block_r"), "block_rmse": bc.get("block_rmse")})
        for v, vv in s.get("variants", {}).items():
            r[f"{v}_gedi_free_rmse"] = vv["height"].get("rmse")
            r[f"{v}_pair_rate_r"] = vv["pair_change"].get("rate_r")
            r[f"{v}_block_r"] = vv["block_change"].get("block_r")
        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(results_path("cv", "experiments.csv"), index=False)
    return df


def cli(argv=None) -> None:
    df = collect()
    cols = ["name", "grid", "sets", "design", "calibration", "all_rmse", "all_r2", "gedi_rmse",
            "gedi_r2", "gedi_bias", "icesat2_rmse", "pair_rate_r", "block_r"]
    print(df[[c for c in cols if c in df]].round(3).to_string(index=False))
