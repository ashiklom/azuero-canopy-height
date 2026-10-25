"""Area of interest: the Azuero Peninsula.

The AOI is the union of the configured geoBoundaries ADM2 districts (Herrera, Los Santos,
Mariato, Montijo), reduced to its largest connected part to drop islands. The resulting
polygon is committed to `results/aoi.geojson` so downstream steps do not need the raw
boundary download.
"""

from __future__ import annotations

from functools import lru_cache

import geopandas as gpd
import numpy as np
import requests
from shapely.geometry import MultiPolygon, Polygon

from .config import data_path, load_config, results_path

AOI_FILE = "aoi.geojson"


def _download_boundaries() -> gpd.GeoDataFrame:
    cfg = load_config()["aoi"]
    dst = data_path("raw", "boundaries", "PAN_ADM2.geojson")
    if not dst.exists():
        r = requests.get(cfg["boundaries_url"], timeout=300)
        r.raise_for_status()
        dst.write_bytes(r.content)
    return gpd.read_file(dst)


def build_aoi() -> gpd.GeoDataFrame:
    cfg = load_config()
    acfg = cfg["aoi"]
    adm2 = _download_boundaries()
    sel = adm2[adm2["shapeName"].isin(acfg["districts"])]
    missing = set(acfg["districts"]) - set(sel["shapeName"])
    if missing:
        raise ValueError(f"districts not found: {sorted(missing)}")
    geom = sel.to_crs(cfg["crs"]).union_all().buffer(0)
    if acfg.get("keep_largest_part") and isinstance(geom, MultiPolygon):
        geom = max(geom.geoms, key=lambda g: g.area)
    # Fill internal slivers between districts.
    if isinstance(geom, Polygon):
        geom = Polygon(geom.exterior)
    if acfg.get("buffer_m"):
        geom = geom.buffer(acfg["buffer_m"])
    geom = geom.simplify(15)
    out = gpd.GeoDataFrame({"name": ["Azuero Peninsula"],
                            "area_km2": [geom.area / 1e6]}, geometry=[geom], crs=cfg["crs"])
    out.to_crs(4326).to_file(results_path(AOI_FILE), driver="GeoJSON")
    return out


@lru_cache(maxsize=1)
def aoi_utm() -> gpd.GeoDataFrame:
    f = results_path(AOI_FILE)
    if not f.exists():
        return build_aoi()
    return gpd.read_file(f).to_crs(load_config()["crs"])


def aoi_geometry():
    return aoi_utm().geometry.iloc[0]


def aoi_bounds_utm() -> tuple[float, float, float, float]:
    return tuple(aoi_utm().total_bounds)


def aoi_bounds_wgs84(pad_deg: float = 0.0) -> tuple[float, float, float, float]:
    b = aoi_utm().to_crs(4326).total_bounds
    return (b[0] - pad_deg, b[1] - pad_deg, b[2] + pad_deg, b[3] + pad_deg)


def aoi_mask(grid) -> np.ndarray:
    """Boolean mask of grid pixels whose centers fall inside the AOI (cached)."""
    from rasterio.features import rasterize

    f = data_path("interim", "masks", f"aoi_{grid.name}.npy")
    if f.exists():
        return np.load(f)
    m = rasterize([(aoi_geometry(), 1)], out_shape=grid.shape, transform=grid.transform,
                  fill=0, dtype="uint8").astype(bool)
    np.save(f, m)
    return m


def cli(argv=None) -> None:
    """Rebuild the AOI polygon and grid masks."""
    from .grid import get_grid

    aoi = build_aoi()
    print(f"AOI area: {aoi.area_km2.iloc[0]:.1f} km2")
    for name in ("hls30", "aef20"):
        g = get_grid(name)
        m = aoi_mask(g)
        print(f"{name}: shape={g.shape} AOI pixels={int(m.sum()):,}")
