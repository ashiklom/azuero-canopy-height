"""Static figures for the website (matplotlib, palette shared with the site CSS)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .aoi import aoi_utm  # noqa: E402

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
SLOT1 = "#2a78d6"
SLOT2 = "#eb6834"
DEEMPH = "#c3c2b7"

# Sequential single-hue ramps (light -> dark) and a diverging red-gray-blue ramp.
GREEN_SEQ = ["#e8f3e0", "#c4e2b0", "#9bcf84", "#6fb65d", "#45983c", "#257a25", "#0f5c17", "#063f0e"]
ORANGE_SEQ = ["#fde7da", "#f9c6a9", "#f4a27a", "#eb7d4f", "#d95926", "#b2441a", "#8a3412", "#5f230b"]
DIVERGING = ["#a32b2b", "#d24c4b", "#e9908f", "#f0efec", "#86b6ef", "#3987e5", "#184f95"]

CMAPS = {
    "height": LinearSegmentedColormap.from_list("height", GREEN_SEQ),
    "sd": LinearSegmentedColormap.from_list("sd", ORANGE_SEQ),
    "change": LinearSegmentedColormap.from_list("change", DIVERGING),
}


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def fold_maps(labels: pd.DataFrame, folds, path, sample: int = 60000):
    """Small multiples: each panel shows one fold's held-out labels (blue), training
    labels (gray) and buffered-out labels (light)."""
    aoi = aoi_utm()
    rng = np.random.default_rng(0)
    idx = rng.choice(len(labels), min(sample, len(labels)), replace=False)
    x, y = labels.x.to_numpy()[idx], labels.y.to_numpy()[idx]
    n = len(folds)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4), facecolor=SURFACE)
    for ax, f in zip(np.atleast_1d(axes), folds):
        _style(ax)
        aoi.boundary.plot(ax=ax, color=MUTED, lw=0.6)
        tr, te = f.train[idx], f.test[idx]
        buf = ~tr & ~te
        ax.scatter(x[tr], y[tr], s=0.3, c=DEEMPH, lw=0, rasterized=True)
        ax.scatter(x[buf], y[buf], s=0.3, c="#e9e8e2", lw=0, rasterized=True)
        ax.scatter(x[te], y[te], s=0.4, c=SLOT1, lw=0, rasterized=True)
        ax.set_title(f"Fold {f.name.replace('space', '')}: held out", color=INK2, fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlabel("")
        ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def lidar_density(labels: pd.DataFrame, path, cell_m: float = 1000.0):
    """Number of quality-screened lidar observations per km2, by sensor (two panels)."""
    aoi = aoi_utm()
    xmin, ymin, xmax, ymax = aoi.total_bounds
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), facecolor=SURFACE)
    cmap = LinearSegmentedColormap.from_list(
        "blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
    for ax, (s, title) in zip(axes, (("gedi", "GEDI L2A shots"), ("icesat2", "ICESat-2 30 m segments"))):
        _style(ax)
        d = labels[labels.sensor == s]
        hb = ax.hexbin(d.x, d.y, gridsize=60, extent=(xmin, xmax, ymin, ymax), cmap=cmap,
                       mincnt=1, bins="log", linewidths=0)
        aoi.boundary.plot(ax=ax, color=INK2, lw=0.6)
        ax.set_title(f"{title} (n = {len(d):,})", color=INK2, fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlabel("")
        ax.set_ylabel("")
        cb = fig.colorbar(hb, ax=ax, shrink=0.7)
        cb.outline.set_visible(False)
        cb.ax.tick_params(colors=MUTED, labelsize=8)
        cb.set_label("observations per cell (log)", color=MUTED, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def colorize(arr: np.ndarray, kind: str, vmin: float, vmax: float) -> np.ndarray:
    """RGBA uint8 image of an array with NaN transparent."""
    cmap = CMAPS[kind]
    norm = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
    rgba = (cmap(norm) * 255).astype(np.uint8)
    rgba[~np.isfinite(arr)] = 0
    return rgba
