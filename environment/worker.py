# environment/worker.py
"""
Shared worker/icon renderer for Robomason plots
-----------------------------------------------
•  Zone supervisors = filled squares (“zones”)
•  Everyone else    = colour-coded 'x' marker
The exact colours, square size and marker map live in construction_config.
"""

import numpy as np
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from configs.construction_config import (
    WORKER_COLORS,       # {wid: colour}
    WORKER_SQUARE_SIZE,  # metres
    worker_marker_styles # {wid: "x" | "zone" | whatever}
)
from plotting.utils import z_level   # single ground-Z constant
# ───────────────────────────────────────────────────────────────────────────
#  low-level helpers
# ───────────────────────────────────────────────────────────────────────────
def _square_xy(ax, wx, wy, colour, *, size):
    rect = Rectangle(
        (wx, wy - size),    # shift down so square is centred on the coord
        size, size,
        facecolor=colour,
        edgecolor=colour,
        alpha=0.7,
        linewidth=0
    )
    ax.add_patch(rect);   return rect


def _square_xyz(ax, wx, wy, wz, colour, *, size):
    verts = [[
        (wx,        wy,        wz),
        (wx + size, wy,        wz),
        (wx + size, wy - size, wz),
        (wx,        wy - size, wz),
        (wx,        wy,        wz)
    ]]
    poly = Poly3DCollection(
        verts,
        facecolors=colour,
        edgecolors=colour,
        alpha=0.7,
        linewidths=0
    )
    ax.add_collection3d(poly);   return poly


# ───────────────────────────────────────────────────────────────────────────
#  public API
# ───────────────────────────────────────────────────────────────────────────
def draw_worker(ax, wid, wx, wy, wz=None, *, view="3d"):
    """
    Place *one* worker icon on `ax` and return the artist.
    """
    colour = WORKER_COLORS.get(wid, "black")
    marker = worker_marker_styles.get(wid, "x")

    if marker == "zone":
        size = WORKER_SQUARE_SIZE
        if view == "3d":
            return _square_xyz(ax, wx, wy, wz or z_level, colour, size=size)
        else:
            return _square_xy(ax,  wx, wy,            colour, size=size)

    # fallback: plain 'x'
    if view == "3d":
        return ax.scatter([wx], [wy], [wz or z_level],
                          marker="x", s=60, c=colour, depthshade=False)
    elif view == "top":
        return ax.scatter([wx], [wy], marker="x", s=60, c=colour)
    elif view == "front":
        return ax.scatter([wx], [wz or z_level], marker="x", s=60, c=colour)
    else:  # side
        return ax.scatter([wy], [wz or z_level], marker="x", s=60, c=colour)


def legend_handles():
    """
    Return two lists (handles, labels) ready for `ax.legend(...)`.
    """
    handles, labels = [], []
    for wid, marker in worker_marker_styles.items():
        colour = WORKER_COLORS.get(wid, "black")
        if marker == "zone":
            handles.append(Patch(facecolor=colour, edgecolor=colour, alpha=0.7))
        else:
            handles.append(Line2D([0], [0], marker="x", color="none",
                                  markeredgecolor=colour, markerfacecolor=colour,
                                  markersize=8, linewidth=0))
        labels.append(f"Worker {wid}")
    return handles, labels
