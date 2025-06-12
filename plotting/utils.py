#/plotting/utils.py
import numpy as np
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from configs.construction_config import *
from configs.system_config import *

from environment import Environment   
from environment.robot import forward_kinematics, draw_robot as _draw_robot

_ENV = Environment()

SCALING_FACTOR = 0.75

# z_level = -0.155
# x_limits = (-0.35, 0.81)
# y_limits = (-0.25, 0.75)

def draw_ground(ax, *,is3d=False):
    """Proxy → Environment.draw_ground (centralised geometry)."""
    _ENV.draw_ground(ax, is3d=is3d)

def draw_sites(ax, *, view='3d'):
    """Proxy → Environment.draw_sites."""
    _ENV.draw_sites(ax, view=view)

def draw_robot(ax, joints, *, view="3d", **kwargs):
    """
    Thin wrapper so existing calls to plotting.utils.draw_robot(...)
    continue to work.  Delegates to environment.robot.draw_robot.
    """
    _draw_robot(ax, joints, view=view, **kwargs)

def process_incoming_data(data):
    coords = np.array(data.get("coordinates", [0, 0, 0]), dtype=float) + TRANSLATION
    joints = data.get("joints", None)
    return coords, joints

# -------------------------
# TRAJECTORY SEGMENTATION FUNCTION
# -------------------------
def normalize_element_name(element):
    """
    Convert an element string (which might use underscores) into the format expected
    by ELEMENT_COLORS. For example, "wall_2" becomes "Wall 2".
    """
    if element is None:
        return ""
    candidate = element.replace("_", " ").strip()
    # Look for an exact match (ignoring case) in ELEMENT_COLORS keys.
    for key in ELEMENT_COLORS:
        if candidate.lower() == key.lower():
            return key
    return candidate.title()

def group_trajectory_points(points):
    """
    Group points (each a dict with 'coords', 'element', and 'state') into segments.
    Each segment is a tuple: (list_xs, list_ys, list_zs, linestyle, color, linewidth, alpha)
    """
    segments = []
    seg_xs, seg_ys, seg_zs = [], [], []
    prev_element = None
    prev_state = None

    for point in points:
        if point.get('element') is None or point.get('state') is None:
            continue
        current_element = normalize_element_name(point.get('element'))
        current_state = point.get('state').strip()
        if current_element.lower() == "scanning site":
            current_state = "-"

        if prev_element is not None and (current_element != prev_element or current_state != prev_state):
            col = ELEMENT_COLORS.get(prev_element, "black")
            if prev_state == "-":
                ls = SCANNING_STYLE["linestyle"]
                lw = SCANNING_STYLE["linewidth"]
            else:
                if prev_state in STATE_STYLES_ASSEMBLY:
                    ls = STATE_STYLES_ASSEMBLY[prev_state]["linestyle"]
                    lw = STATE_STYLES_ASSEMBLY[prev_state]["linewidth"]
                else:
                    ls = "-"
                    lw = 2.5
            segments.append((seg_xs.copy(), seg_ys.copy(), seg_zs.copy(), ls, col, lw, 1.0))
            seg_xs.clear()
            seg_ys.clear()
            seg_zs.clear()

        seg_xs.append(point['coords'][0])
        seg_ys.append(point['coords'][1])
        seg_zs.append(point['coords'][2])
        prev_element = current_element
        prev_state = current_state

    if seg_xs:
        col = ELEMENT_COLORS.get(prev_element, "black")
        if prev_state == "-":
            ls = SCANNING_STYLE["linestyle"]
            lw = SCANNING_STYLE["linewidth"]
        else:
            if prev_state in STATE_STYLES_ASSEMBLY:
                ls = STATE_STYLES_ASSEMBLY[prev_state]["linestyle"]
                lw = STATE_STYLES_ASSEMBLY[prev_state]["linewidth"]
            else:
                ls = "-"
                lw = 2.5
        segments.append((seg_xs.copy(), seg_ys.copy(), seg_zs.copy(), ls, col, lw, 1.0))
    return segments