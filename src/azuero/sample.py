"""Fast sampling of grid-aligned rasters at many (row, col) cells.

Points are grouped by row strips so that each strip is read once (all bands, full
width), which is efficient for tiled, compressed GeoTIFFs on local disk.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


def sample_cells(path: str | Path, rows: np.ndarray, cols: np.ndarray,
                 bands: list[int] | None = None, strip: int = 256,
                 out_dtype=np.float32) -> np.ndarray:
    """Return (n, nbands) values; nodata becomes NaN for float outputs."""
    rows = np.asarray(rows, np.int64)
    cols = np.asarray(cols, np.int64)
    with rasterio.open(path) as src:
        idx = bands or list(range(1, src.count + 1))
        out = np.full((rows.size, len(idx)), np.nan, out_dtype)
        nod = src.nodata
        order = np.argsort(rows, kind="stable")
        rs = rows[order]
        for r0 in range(0, src.height, strip):
            lo, hi = np.searchsorted(rs, [r0, r0 + strip])
            if lo == hi:
                continue
            sel = order[lo:hi]
            h = min(strip, src.height - r0)
            a = src.read(idx, window=Window(0, r0, src.width, h))
            vals = a[:, rows[sel] - r0, cols[sel]].T.astype(out_dtype)
            if nod is not None and np.issubdtype(out_dtype, np.floating):
                vals[a[:, rows[sel] - r0, cols[sel]].T == nod] = np.nan
            out[sel] = vals
    return out


def band_names(path: str | Path) -> list[str]:
    with rasterio.open(path) as src:
        return list(src.descriptions)
