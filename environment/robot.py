# environment/robot.py
"""
Authoritative helpers for the UR-5:
  • forward_kinematics(joints)  → (7 × 3) XYZ array
  • draw_robot(ax, joints, view=…, **kw)  → stick model

Only this file knows the kinematics.  Everybody else just *calls* it.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from configs.system_config import DH_params, TRANSLATION   # site offsets

# ------------------------------------------------------------------ FK
def _dh(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """Single Denavit–Hartenberg transform."""
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array(
        [[ct, -st * ca,  st * sa, a * ct],
         [st,  ct * ca, -ct * sa, a * st],
         [0 ,       sa,       ca,       d],
         [0 ,        0,        0,       1]]
    )

# Ground–anchor that was previously injected by the plotting helpers
_GROUND_ANCHOR = np.array([0.0, 0.34301, -0.20000])       # metres

def forward_kinematics(joints: np.ndarray | list[float]) -> np.ndarray:
    """
    Compute the UR-5 chain **once** in the world frame.

    Returns
    -------
    pts : (7, 3) ndarray
        [ ground-anchor ,
          joint-0 , … , joint-5 / TCP ]
    """
    q = np.asarray(joints, dtype=float).ravel()
    if q.size != 6:
        raise ValueError("forward_kinematics expects 6 joint angles (got %d)" % q.size)

    pts = [_GROUND_ANCHOR]           # first point: same as the old helpers
    T   = np.eye(4)

    for qi, p in zip(q, DH_params):
        T = T @ _dh(p["a"], p["alpha"], p["d"], p["theta"] + qi)
        # shift *once* with the global site translation (no double offsets!)
        pts.append(T[:3, 3] + TRANSLATION)

    return np.vstack(pts)            # shape (7, 3)

# ---------------------------------------------------------------- draw
def draw_robot(ax: plt.Axes, joints, *, view: str = "3d", **kwargs):
    """
    Stick model in one line.

    Parameters
    ----------
    ax     : Matplotlib 2-D or 3-D axes
    joints : iterable of 6 joint angles [rad]
    view   : "3d", "top" (X-Y), "front" (X-Z) or "side" (Y-Z)
    **kwargs : forwarded to `ax.plot` (colour, linewidth, …)
    """
    style = {"color": "k", "marker": "o", "lw": 2} | kwargs
    P     = forward_kinematics(joints)

    if view == "3d":
        ax.plot(P[:, 0], P[:, 1], P[:, 2], **style)
    elif view == "top":            # X–Y
        ax.plot(P[:, 0], P[:, 1], **style)
    elif view == "front":          # X–Z
        ax.plot(P[:, 0], P[:, 2], **style)
    elif view == "side":           # Y–Z
        ax.plot(P[:, 1], P[:, 2], **style)
    else:
        raise ValueError("view must be '3d', 'top', 'front' or 'side'")
