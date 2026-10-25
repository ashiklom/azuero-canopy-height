"""Cross-validation designs.

Spatial: the AOI frame is tiled into square blocks (`block_km`). Blocks with labels are
grouped into contiguous regions: k-means on block centres seeds the regions, so each
region is a compact patch of several adjacent blocks. Each fold holds out
`regions_per_fold` regions that are not adjacent to each other. Training labels within
`buffer_km` of any held-out label are dropped for that fold, which reduces leakage from
spatial autocorrelation.

Temporal: leave-one-year-out holds out all labels of a calendar year.

Spatio-temporal: the spatial folds crossed with years. The model never sees the test
region in any year, nor any data from the test year.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

from .config import load_config


@dataclass
class Fold:
    name: str
    train: np.ndarray  # boolean masks over the label table
    test: np.ndarray


def block_ids(x: np.ndarray, y: np.ndarray, block_km: float) -> tuple[np.ndarray, np.ndarray]:
    b = block_km * 1000.0
    bx = np.floor(x / b).astype(np.int64)
    by = np.floor(y / b).astype(np.int64)
    return bx, by


def spatial_regions(labels: pd.DataFrame, block_km: float, n_regions: int, seed: int
                    ) -> np.ndarray:
    """Region id per label, from k-means clustering of occupied block centres."""
    bx, by = block_ids(labels.x.to_numpy(), labels.y.to_numpy(), block_km)
    blocks = pd.DataFrame({"bx": bx, "by": by})
    ub = blocks.drop_duplicates().reset_index(drop=True)
    km = KMeans(n_clusters=n_regions, n_init=20, random_state=seed)
    reg = km.fit_predict(ub[["bx", "by"]].to_numpy(float))
    ub["region"] = reg
    return blocks.merge(ub, on=["bx", "by"], how="left")["region"].to_numpy()


def assign_regions_to_folds(labels: pd.DataFrame, region: np.ndarray, n_folds: int,
                            seed: int) -> np.ndarray:
    """Spread regions over folds, balancing label counts and avoiding adjacent regions
    in the same fold where possible."""
    rng = np.random.default_rng(seed)
    regs = pd.DataFrame({"region": region, "x": labels.x.to_numpy(), "y": labels.y.to_numpy()})
    stats = regs.groupby("region").agg(n=("x", "size"), cx=("x", "mean"), cy=("y", "mean"))
    order = stats.sort_values("n", ascending=False).index.to_numpy()
    fold_of = {}
    load = np.zeros(n_folds)
    members = {f: [] for f in range(n_folds)}
    for r in order:
        c = stats.loc[r, ["cx", "cy"]].to_numpy()
        # Penalize folds that already contain a nearby region.
        pen = np.array([min([np.hypot(*(c - stats.loc[o, ["cx", "cy"]].to_numpy()))
                             for o in members[f]] or [np.inf]) for f in range(n_folds)])
        score = load / load.sum() if load.sum() else np.zeros(n_folds)
        score = score + np.where(pen < 15000, 1.0, 0.0) + rng.random(n_folds) * 1e-6
        f = int(np.argmin(score))
        fold_of[r] = f
        members[f].append(r)
        load[f] += stats.loc[r, "n"]
    return np.array([fold_of[r] for r in region])


def buffered_train(xy_train: np.ndarray, xy_test: np.ndarray, buffer_m: float) -> np.ndarray:
    if buffer_m <= 0 or len(xy_test) == 0:
        return np.ones(len(xy_train), bool)
    tree = cKDTree(xy_test)
    d, _ = tree.query(xy_train, k=1, distance_upper_bound=buffer_m)
    return ~np.isfinite(d)


def spatial_folds(labels: pd.DataFrame, block_km: float | None = None,
                  n_folds: int | None = None, regions_per_fold: int | None = None,
                  buffer_km: float | None = None, seed: int | None = None) -> list[Fold]:
    c = load_config()["cv"]
    block_km = block_km or c["block_km"]
    n_folds = n_folds or c["n_folds"]
    regions_per_fold = regions_per_fold or c["regions_per_fold"]
    buffer_km = c["buffer_km"] if buffer_km is None else buffer_km
    seed = c["seed"] if seed is None else seed
    region = spatial_regions(labels, block_km, n_folds * regions_per_fold, seed)
    fold = assign_regions_to_folds(labels, region, n_folds, seed)
    xy = labels[["x", "y"]].to_numpy()
    out = []
    for f in range(n_folds):
        test = fold == f
        train = ~test
        keep = buffered_train(xy[train], xy[test], buffer_km * 1000)
        tr = np.zeros(len(labels), bool)
        tr[np.nonzero(train)[0][keep]] = True
        out.append(Fold(f"space{f}", tr, test))
    return out


def fold_assignment(labels: pd.DataFrame, **kw) -> np.ndarray:
    """Spatial fold id per label (for maps)."""
    c = load_config()["cv"]
    block_km = kw.get("block_km") or c["block_km"]
    n_folds = kw.get("n_folds") or c["n_folds"]
    rpf = kw.get("regions_per_fold") or c["regions_per_fold"]
    seed = c["seed"]
    region = spatial_regions(labels, block_km, n_folds * rpf, seed)
    return assign_regions_to_folds(labels, region, n_folds, seed)


def year_folds(labels: pd.DataFrame, years=None) -> list[Fold]:
    years = years or sorted(labels.year.unique())
    return [Fold(f"year{y}", (labels.year != y).to_numpy(), (labels.year == y).to_numpy())
            for y in years]


def space_time_folds(labels: pd.DataFrame, years=None, **kw) -> list[Fold]:
    """Each spatial fold x each held-out year: train excludes the test region (all years,
    with buffer) and the test year (everywhere)."""
    sp = spatial_folds(labels, **kw)
    years = years or sorted(labels.year.unique())
    out = []
    yr = labels.year.to_numpy()
    for f in sp:
        for y in years:
            test = f.test & (yr == y)
            if test.sum() < 50:
                continue
            out.append(Fold(f"{f.name}_year{y}", f.train & (yr != y), test))
    return out
