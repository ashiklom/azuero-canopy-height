import numpy as np
import rasterio
from rasterio.transform import from_origin

from azuero.sample import sample_cells


def test_sample_cells(tmp_path):
    a = np.arange(3 * 600 * 50, dtype=np.int16).reshape(3, 600, 50)
    a[1, 300, 7] = -1
    f = tmp_path / "x.tif"
    with rasterio.open(f, "w", driver="GTiff", width=50, height=600, count=3, dtype="int16",
                       transform=from_origin(0, 0, 1, 1), nodata=-1) as dst:
        dst.write(a)
    rows = np.array([599, 0, 300, 257])
    cols = np.array([3, 49, 7, 0])
    v = sample_cells(f, rows, cols, strip=256)
    assert v[0, 0] == a[0, 599, 3] and v[1, 2] == a[2, 0, 49] and v[3, 1] == a[1, 257, 0]
    assert np.isnan(v[2, 1])
