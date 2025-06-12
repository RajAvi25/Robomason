"""
Robomason – shared plotting environment
---------------------------------------
•  One authoritative description of the ground plane and the four site boxes
•  Draw helpers for both 3-D and 2-D views (top / front / side)
•  No more duplicated code in live plotting *or* post-run analysis
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# colours that already exist in your config
from configs.construction_config import SITE_COLOR, SITE_ALPHA

# ----------------------------------------------------------------------
# hard-coded scene geometry  (taken verbatim from plotting/plotting.py)
# ----------------------------------------------------------------------
X_LIMITS   = (-0.35, 0.81)
Y_LIMITS   = (-0.25, 0.75)
Z_LEVEL    = -0.155            # floor height
Z_LIMITS   = (Z_LEVEL, 0.49)

SITE_DIMS = [                  # (width, height)  in metres
    (29.5/100, 21/100),
    (21/100,  29.5/100),
    (21/100,  29.5/100),
    (45/100,  29.5/100),
]

SITE_POS = [                   # box centres (x, y)
    (-0.1522, -0.0485),
    ( 0.5802,  0.2528),
    ( 0.5802,  0.5669),
    ( 0.2420, -0.0485),
]

GROUND_COLOR = "peru"
GROUND_ALPHA = 0.30


class Environment:  # pylint: disable=too-few-public-methods
    """Draw helpers for Robomason ground + site boxes."""

    # .............................. PUBLIC ................................
    def draw_ground(self, ax, *, is3d: bool = False) -> None:
        """Ground rectangle (3-D plane or 2-D patch)."""
        if is3d:
            xx, yy = np.meshgrid(
                np.linspace(*X_LIMITS, 10),
                np.linspace(*Y_LIMITS, 10),
            )
            zz = np.full_like(xx, Z_LEVEL)
            ax.plot_surface(
                xx, yy, zz,
                color=GROUND_COLOR,
                alpha=GROUND_ALPHA,
                rstride=100, cstride=100,
            )
        else:
            ax.add_patch(
                Rectangle(
                    (X_LIMITS[0], Y_LIMITS[0]),
                    X_LIMITS[1] - X_LIMITS[0],
                    Y_LIMITS[1] - Y_LIMITS[0],
                    facecolor=GROUND_COLOR,
                    alpha=GROUND_ALPHA,
                )
            )

    def draw_sites(self, ax, *, view: str = "3d") -> None:
        """Four blue site boxes – either prisms (3-D) or rectangles (2-D)."""
        for (cx, cy), (w, h) in zip(SITE_POS, SITE_DIMS):
            if view == "3d":
                # 5-point closed rectangle at z = Z_LEVEL
                xs = [cx - w/2, cx + w/2, cx + w/2, cx - w/2, cx - w/2]
                ys = [cy - h/2, cy - h/2, cy + h/2, cy + h/2, cy - h/2]
                zs = [Z_LEVEL] * 5
                poly = Poly3DCollection(
                    [list(zip(xs, ys, zs))],
                    facecolors=SITE_COLOR,
                    edgecolors="k",
                    alpha=SITE_ALPHA,
                )
                ax.add_collection3d(poly)
            else:
                ax.add_patch(
                    Rectangle(
                        (cx - w/2, cy - h/2),
                        w, h,
                        facecolor=SITE_COLOR,
                        edgecolor="k",
                        alpha=SITE_ALPHA,
                    )
                )

    # convenience (optional – makes analysis cleaner)
    def draw_environment(self, ax, *, view: str = "3d") -> None:
        """Ground + sites together (keeps axes limits consistent)."""
        is3d = (view == "3d")
        self.draw_ground(ax, is3d=is3d)
        self.draw_sites(ax, view="3d" if is3d else "2d")

        if is3d:
            ax.set_xlim(*X_LIMITS); ax.set_ylim(*Y_LIMITS); ax.set_zlim(*Z_LIMITS)
        elif view == "top":
            ax.set_xlim(*X_LIMITS); ax.set_ylim(*Y_LIMITS); ax.set_aspect("equal")
        elif view == "front":
            ax.set_xlim(*X_LIMITS); ax.set_ylim(*Z_LIMITS); ax.set_aspect("equal")
        else:  # side
            ax.set_xlim(*Y_LIMITS); ax.set_ylim(*Z_LIMITS); ax.set_aspect("equal")
