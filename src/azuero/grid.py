"""Named analysis grids.

All grids share the AOI CRS (UTM 17N) and have pixel edges at integer multiples of the
resolution, which makes `hls30` coincide with the HLS/MGRS 30 m grid and `aef20`/`aef10`
coincide with the AlphaEarth embedding pixel grid (tile edges at 500000 +/- k*81920 m).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from affine import Affine

from .config import load_config, resources


@dataclass(frozen=True)
class Grid:
    name: str
    res: float
    crs: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> int:
        return int(round((self.xmax - self.xmin) / self.res))

    @property
    def height(self) -> int:
        return int(round((self.ymax - self.ymin) / self.res))

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def transform(self) -> Affine:
        return Affine(self.res, 0, self.xmin, 0, -self.res, self.ymax)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (self.xmin, self.ymin, self.xmax, self.ymax)

    def rowcol(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """Row/col indices (may fall outside the grid) for projected coordinates."""
        col = np.floor((np.asarray(x) - self.xmin) / self.res).astype(np.int64)
        row = np.floor((self.ymax - np.asarray(y)) / self.res).astype(np.int64)
        return row, col

    def xy(self, row, col) -> tuple[np.ndarray, np.ndarray]:
        """Pixel-center coordinates."""
        x = self.xmin + (np.asarray(col) + 0.5) * self.res
        y = self.ymax - (np.asarray(row) + 0.5) * self.res
        return x, y

    def inside(self, row, col) -> np.ndarray:
        row, col = np.asarray(row), np.asarray(col)
        return (row >= 0) & (row < self.height) & (col >= 0) & (col < self.width)

    def windows(self, tile_px: int):
        """Yield (row_off, col_off, nrows, ncols) tiles covering the grid."""
        for r in range(0, self.height, tile_px):
            for c in range(0, self.width, tile_px):
                yield r, c, min(tile_px, self.height - r), min(tile_px, self.width - c)

    def profile(self, **kw) -> dict:
        p = dict(driver="GTiff", width=self.width, height=self.height, crs=self.crs,
                 transform=self.transform)
        p.update(kw)
        return p


def snap_bounds(bounds, res: float, pad: float = 0.0):
    xmin, ymin, xmax, ymax = bounds
    return (math.floor((xmin - pad) / res) * res, math.floor((ymin - pad) / res) * res,
            math.ceil((xmax + pad) / res) * res, math.ceil((ymax + pad) / res) * res)


@lru_cache(maxsize=None)
def get_grid(name: str | None = None) -> Grid:
    from .aoi import aoi_bounds_utm

    cfg = load_config()
    name = name or cfg["primary_grid"]
    gcfg = cfg["grids"][name]
    allowed = gcfg.get("profiles")
    prof = resources().profile
    if allowed and prof not in allowed:
        raise RuntimeError(f"grid {name!r} is only enabled for profiles {allowed}; "
                           f"active profile is {prof!r}")
    res = float(gcfg["res"])
    # Pad by 60 m (common multiple of 10/20/30) so all grids nest within a 60 m frame.
    xmin, ymin, xmax, ymax = snap_bounds(aoi_bounds_utm(), 60.0, pad=60.0)
    return Grid(name, res, cfg["crs"], xmin, ymin, xmax, ymax)
