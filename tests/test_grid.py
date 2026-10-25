import numpy as np

from azuero.grid import get_grid


def test_grids_nest_and_align():
    g30, g20 = get_grid("hls30"), get_grid("aef20")
    assert g30.bounds == g20.bounds
    for g in (g30, g20):
        assert g.xmin % g.res == 0 and g.ymax % g.res == 0
    # AlphaEarth tile edges (500000 +/- k*81920) are on the 20 m grid.
    assert (500000 - g20.xmin) % 20 == 0 and (819200 - g20.ymax) % 20 == 0


def test_rowcol_roundtrip():
    g = get_grid("hls30")
    rows = np.array([0, 10, g.height - 1])
    cols = np.array([0, 5, g.width - 1])
    x, y = g.xy(rows, cols)
    r2, c2 = g.rowcol(x, y)
    assert (r2 == rows).all() and (c2 == cols).all()
