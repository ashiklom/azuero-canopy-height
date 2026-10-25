import numpy as np

from azuero.grid import Grid
from azuero.regrid import regrid_mean


def test_20_to_30_exact():
    src = Grid("s", 20.0, "EPSG:32617", 0, 0, 120, 120)
    dst = Grid("d", 30.0, "EPSG:32617", 0, 0, 120, 120)
    img = np.arange(36, dtype=np.float32).reshape(6, 6)
    out = regrid_mean(img, src, dst)[0]
    # top-left 30 m pixel = 20m (0,0) full, (0,1),(1,0) half, (1,1) quarter
    w = np.array([[1, .5], [.5, .25]])
    exp = (img[:2, :2] * w).sum() / w.sum()
    assert np.isclose(out[0, 0], exp)
    assert np.isclose(np.nanmean(out), img.mean())


def test_nan_handling():
    src = Grid("s", 10.0, "EPSG:32617", 0, 0, 60, 60)
    dst = Grid("d", 30.0, "EPSG:32617", 0, 0, 60, 60)
    img = np.ones((6, 6), np.float32)
    img[:3, :3] = np.nan
    out = regrid_mean(img, src, dst)[0]
    assert np.isnan(out[0, 0]) and np.all(out.ravel()[1:] == 1)
