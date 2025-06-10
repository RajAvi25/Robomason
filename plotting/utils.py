#/plotting/utils.py
import numpy as np
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from configs.construction_config import *
from configs.system_config import *

# CONFIG
SCALING_FACTOR = 0.75

z_level = -0.155
x_limits = (-0.35, 0.81)
y_limits = (-0.25, 0.75)

def transformation_matrix(a, alpha, d, theta):
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),               np.cos(alpha),               d],
        [0,              0,                           0,                           1]
    ])

def forward_kinematics(joint_angles):
    """Compute all joint positions in 3D (with a constant translation from config)."""
    positions = [np.array([0, 0, DH_params[0]['d']]) + TRANSLATION]
    T = np.eye(4)
    for i, params in enumerate(DH_params):
        theta = params['theta'] + joint_angles[i]
        T_joint = transformation_matrix(params['a'], params['alpha'], params['d'], theta)
        T = T @ T_joint
        x, y, z = T[0, 3], T[1, 3], T[2, 3]
        positions.append(np.array([x, y, z]) + TRANSLATION)
    return positions

def draw_ground(ax, is3d=False):
    if is3d:
        xx, yy = np.meshgrid(np.linspace(x_limits[0], x_limits[1], 10),
                             np.linspace(y_limits[0], y_limits[1], 10))
        zz = np.full_like(xx, z_level)
        ax.plot_surface(xx, yy, zz, color='peru', alpha=0.3, rstride=100, cstride=100)
    else:
        rect = Rectangle((x_limits[0], y_limits[0]),
                         x_limits[1]-x_limits[0],
                         y_limits[1]-y_limits[0],
                         facecolor='peru', alpha=0.3)
        ax.add_patch(rect)

def draw_sites(ax, view='3d'):
    site_dims = [
        (29.5/100, 21/100),
        (21/100, 29.5/100),
        (21/100, 29.5/100),
        (45/100, 29.5/100)
    ]
    box_positions = [
        (-0.1522, -0.0485),
        (0.5802, 0.2528),
        (0.5802, 0.5669),
        (0.242, -0.0485)
    ]
    for (x_center, y_center), dims in zip(box_positions, site_dims):
        if view == '3d':
            w, h = dims
            rect_x = [x_center - w/2, x_center + w/2, x_center + w/2, x_center - w/2, x_center - w/2]
            rect_y = [y_center - h/2, y_center - h/2, y_center + h/2, y_center + h/2, y_center - h/2]
            rect_z = [z_level] * 5
            verts = [list(zip(rect_x, rect_y, rect_z))]
            poly = Poly3DCollection(verts, facecolors=SITE_COLOR, alpha=SITE_ALPHA, edgecolors='k')
            ax.add_collection3d(poly)
        else:
            w, h = dims
            x_left = x_center - w/2
            y_bottom = y_center - h/2
            rect = Rectangle((x_left, y_bottom), w, h,
                             facecolor=SITE_COLOR, alpha=SITE_ALPHA, edgecolor='k')
            ax.add_patch(rect)

def draw_robot(ax, joints, view='3d'):
    joints = np.array(joints)
    if joints.ndim == 1:
        joints = forward_kinematics(joints)
        ground_anchor = np.array([0, 0.34301, -0.2])
        joints = [ground_anchor] + joints
        joints = np.array(joints)
    if view == '3d':
        ax.plot(joints[:,0], joints[:,1], joints[:,2], 'ko-', lw=2)
    elif view == 'top':
        ax.plot(joints[:,0], joints[:,1], 'ko-', lw=2)
    elif view == 'front':
        ax.plot(joints[:,0], joints[:,2], 'ko-', lw=2)
    elif view == 'side':
        ax.plot(joints[:,1], joints[:,2], 'ko-', lw=2)

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