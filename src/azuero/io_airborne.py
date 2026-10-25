"""AVIRIS-3 imaging spectroscopy from the AVUELO campaign (Feb 2025).

Each orthocorrected L2A reflectance scene (3.4 m, 284 bands) is streamed: downloaded,
reduced, and deleted (per `delete_raw_after_composite`). The reduction:
* average the 3.4 m pixels whose centers fall in each analysis-grid cell (cells need at
  least half of their nominal pixel count to be valid);
* bin the spectrum to ~30 nm windows, skipping the 1330-1460 nm and 1780-1990 nm water
  absorption regions and >2450 nm.
Covered cells are stored sparsely (row, col, binned spectrum) per grid as parquet, since
the flights cover a small fraction of the AOI. Derived features (PCs and indices) are
computed by `features`. EMIT (60 m) and UAVSAR (two lines flown May 2026, after the
product period) were evaluated for availability but are not processed.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import data_path, resources
from .earthdata import login, session
from .grid import get_grid

log = logging.getLogger(__name__)

SHORT_NAME = "AVUELO_AV3_L2A_Reflectance_2479"
BAD = [(1330, 1460), (1780, 1990), (2450, 3000)]
BIN_NM = 30.0


def bin_edges() -> np.ndarray:
    edges, lo = [], 400.0
    while lo < 2450:
        hi = lo + BIN_NM
        if not any(a < hi and lo < b for a, b in BAD):
            edges.append((lo, hi))
        lo = hi
    return np.array(edges)


def bin_names() -> list[str]:
    return [f"av3_{int(a)}_{int(b)}" for a, b in bin_edges()]


def granules():
    import earthaccess

    from .aoi import aoi_bounds_wgs84

    login()
    gs = earthaccess.search_data(short_name=SHORT_NAME, bounding_box=aoi_bounds_wgs84(),
                                 count=-1)
    return [g for g in gs if "RFL_ORT" in g["umm"]["GranuleUR"]]


def _reduce_scene(path: Path, grid_names) -> dict[str, pd.DataFrame]:
    import netCDF4

    d = netCDF4.Dataset(path)
    e = d["easting"][:].astype(np.float64)
    n = d["northing"][:].astype(np.float64)
    wl = d["reflectance/wavelength"][:].astype(np.float64)
    refl = d["reflectance/reflectance"]
    edges = bin_edges()
    band_bin = np.full(wl.size, -1)
    for i, (a, b) in enumerate(edges):
        band_bin[(wl >= a) & (wl < b)] = i
    nb = len(edges)
    px = abs(e[1] - e[0])
    out = {}
    cells = {}
    for g in grid_names:
        grid = get_grid(g)
        rr, cc = grid.rowcol(e[None, :], n[:, None])
        rr, cc = np.broadcast_arrays(rr, cc)
        inside = grid.inside(rr, cc)
        lin = np.where(inside, rr * grid.width + cc, -1).ravel()
        uniq, inv = np.unique(lin, return_inverse=True)
        cells[g] = (grid, uniq, inv, np.zeros((nb, uniq.size)), np.zeros((nb, uniq.size)))
    valid_any = None
    for b0 in range(0, wl.size, 10):
        block = refl[b0:b0 + 10, :, :]
        block = np.ma.filled(block.astype(np.float32), np.nan)
        for k in range(block.shape[0]):
            bi = band_bin[b0 + k]
            band = block[k].ravel()
            ok = np.isfinite(band) & (band > -1)
            if valid_any is None and b0 + k == 50:
                valid_any = ok.copy()
            if bi < 0:
                continue
            for g, (grid, uniq, inv, s, c) in cells.items():
                s[bi] += np.bincount(inv, weights=np.where(ok, band, 0), minlength=uniq.size)
                c[bi] += np.bincount(inv, weights=ok.astype(np.float64), minlength=uniq.size)
    d.close()
    for g, (grid, uniq, inv, s, c) in cells.items():
        nominal = (grid.res / px) ** 2
        nvalid = np.bincount(inv, weights=valid_any.astype(np.float64), minlength=uniq.size)
        keep = (uniq >= 0) & (nvalid >= 0.5 * nominal)
        with np.errstate(all="ignore"):
            spec = (s / c)[:, keep].T.astype(np.float32)
        df = pd.DataFrame(spec, columns=bin_names())
        df.insert(0, "col", (uniq[keep] % grid.width).astype(np.int32))
        df.insert(0, "row", (uniq[keep] // grid.width).astype(np.int32))
        df["n_px"] = nvalid[keep].astype(np.float32)
        out[g] = df
    return out


def build(grid_names=("hls30", "aef20"), force: bool = False) -> dict[str, Path]:
    finals = {g: data_path("interim", "airborne", f"av3_{g}.parquet") for g in grid_names}
    if all(f.exists() for f in finals.values()) and not force:
        return finals
    sess = session()
    res = resources()
    scene_dir = data_path("interim", "airborne", "scenes")
    raw_dir = data_path("raw", "av3") if not res.delete_raw_after_composite else data_path("tmp", "av3")
    for gr in granules():
        gid = gr["umm"]["GranuleUR"].split(".", 1)[1].replace(".nc", "")
        done = [scene_dir / f"{gid}_{g}.parquet" for g in grid_names]
        if all(f.exists() for f in done) and not force:
            continue
        url = gr.data_links(access="external")[0]
        local = raw_dir / url.rsplit("/", 1)[-1]
        if not local.exists():
            with sess.get(url, stream=True, timeout=900) as r:
                r.raise_for_status()
                tmp = local.with_suffix(".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 22):
                        f.write(chunk)
                tmp.rename(local)
        try:
            dfs = _reduce_scene(local, grid_names)
        finally:
            if res.delete_raw_after_composite:
                local.unlink(missing_ok=True)
        for g, df in dfs.items():
            df["scene"] = gid
            df.to_parquet(scene_dir / f"{gid}_{g}.parquet")
        log.info("AV3 %s: %s cells", gid, {g: len(df) for g, df in dfs.items()})
    for g in grid_names:
        parts = [pd.read_parquet(f) for f in sorted(scene_dir.glob(f"*_{g}.parquet"))]
        df = pd.concat(parts, ignore_index=True)
        # Overlapping flight lines: keep the observation with the most valid pixels.
        df = df.sort_values("n_px", ascending=False).drop_duplicates(["row", "col"])
        df.to_parquet(finals[g])
    return finals


def cli(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(prog="azuero airborne")
    p.add_argument("--grids", nargs="*", default=["hls30", "aef20"])
    p.add_argument("--force", action="store_true")
    a = p.parse_args(argv)
    build(tuple(a.grids), a.force)
