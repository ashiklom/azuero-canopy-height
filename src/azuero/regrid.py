"""Exact area-weighted regridding between aligned axis-parallel grids.

Grids in this project share a CRS and have pixel edges on integer multiples of their
resolution, so resampling (e.g. AlphaEarth 20 m -> HLS 30 m) is a separable operation:
out = Wy @ img @ Wx.T, where W holds the fractional overlap of target and source pixels.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def overlap_matrix(t0: float, tres: float, tn: int, s0: float, sres: float, sn: int
                   ) -> sp.csr_matrix:
    """Row-normalized overlap weights (tn x sn) between ascending 1-D pixel edges.

    Target pixel i spans [t0 + i*tres, t0 + (i+1)*tres); source likewise.
    """
    rows, cols, vals = [], [], []
    for i in range(tn):
        lo, hi = t0 + i * tres, t0 + (i + 1) * tres
        j0 = max(int(np.floor((lo - s0) / sres)), 0)
        j1 = min(int(np.ceil((hi - s0) / sres)), sn)
        for j in range(j0, j1):
            ov = min(hi, s0 + (j + 1) * sres) - max(lo, s0 + j * sres)
            if ov > 1e-9:
                rows.append(i)
                cols.append(j)
                vals.append(ov / tres)
    return sp.csr_matrix((vals, (rows, cols)), shape=(tn, sn), dtype=np.float32)


def grid_weights(src_grid, dst_grid) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    # Rows run north->south: use distance below the source top edge as the coordinate.
    wy = overlap_matrix(src_grid.ymax - dst_grid.ymax, dst_grid.res, dst_grid.height,
                        0.0, src_grid.res, src_grid.height)
    wx = overlap_matrix(dst_grid.xmin - src_grid.xmin, dst_grid.res, dst_grid.width,
                        0.0, src_grid.res, src_grid.width)
    return wy, wx


def regrid_mean(img: np.ndarray, src_grid, dst_grid, valid: np.ndarray | None = None,
                min_cover: float = 0.5) -> np.ndarray:
    """Area-weighted mean of a (bands, H, W) array from src_grid to dst_grid.

    Invalid source pixels (NaN, or valid == False) are excluded and the weights are
    renormalized; target pixels with less than `min_cover` valid area are NaN.
    """
    if img.ndim == 2:
        img = img[None]
    wy, wx = grid_weights(src_grid, dst_grid)
    ok = np.isfinite(img[0]) if valid is None else valid
    cover = np.asarray(wy @ ok.astype(np.float32) @ wx.T)
    out = np.full((img.shape[0], dst_grid.height, dst_grid.width), np.nan, np.float32)
    good = cover >= min_cover
    for b in range(img.shape[0]):
        band = np.where(ok, img[b], 0).astype(np.float32)
        acc = np.asarray(wy @ band @ wx.T)
        out[b][good] = acc[good] / cover[good]
    return out
