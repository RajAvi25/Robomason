# plotting/helpers.py
import numpy as np
from matplotlib.patches import Rectangle
from .config import x_limits, y_limits, z_level, DH_params

def rdp_3d(points, epsilon):
    """Perform Ramer-Douglas-Peucker simplification on 3D points."""
    if len(points) < 3:
        return points

    start, end = points[0], points[-1]
    se = end - start
    cross_products = np.cross(points[1:-1] - start, se)
    dists = np.linalg.norm(cross_products, axis=1) / np.linalg.norm(se)
    max_dist_idx = np.argmax(dists)
    max_dist = dists[max_dist_idx]

    if max_dist > epsilon:
        idx = max_dist_idx + 1
        left_segment = rdp_3d(points[:idx+1], epsilon)
        right_segment = rdp_3d(points[idx:], epsilon)
        return np.vstack((left_segment[:-1], right_segment))
    else:
        return np.array([start, end])

def transformation_matrix(a, alpha, d, theta):
    """Return transformation matrix for given DH parameters."""
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,             np.sin(alpha),                np.cos(alpha),               d],
        [0,             0,                            0,                           1]
    ])

def forward_kinematics(joint_angles):
    """Compute forward kinematics to obtain joint positions in 3D."""
    # Translation to world frame
    translation = np.array([0.02099, 0.34301, -0.30002])
    positions = [np.array([0, 0, DH_params[0]['d']]) + translation]
    T = np.eye(4)
    for i, params in enumerate(DH_params):
        theta = params['theta'] + joint_angles[i]
        T_joint = transformation_matrix(params['a'], params['alpha'], params['d'], theta)
        T = T @ T_joint
        pos = T[:3, 3] + translation
        positions.append(pos)
    return positions

def show_robot_3d(line_obj, joint_angles):
    """Update a 3D robot line using forward kinematics."""
    joint_positions = forward_kinematics(joint_angles)
    ground_anchor = np.array([0, 0.34301, -0.2])
    joint_positions = [ground_anchor] + joint_positions
    xs = [p[0] for p in joint_positions]
    ys = [p[1] for p in joint_positions]
    zs = [p[2] for p in joint_positions]
    line_obj.set_data(xs, ys)
    line_obj.set_3d_properties(zs)

def show_robot_2d(ax_line, joint_angles, view='top'):
    """Update a 2D robot line using forward kinematics for a given view."""
    joint_positions = forward_kinematics(joint_angles)
    if view == 'top':
        xs = [p[0] for p in joint_positions]
        ys = [p[1] for p in joint_positions]
        ax_line.set_xdata(xs)
        ax_line.set_ydata(ys)
    elif view == 'front':
        xs = [p[0] for p in joint_positions]
        zs = [p[2] for p in joint_positions]
        ax_line.set_xdata(xs)
        ax_line.set_ydata(zs)
    elif view == 'side':
        ys = [p[1] for p in joint_positions]
        zs = [p[2] for p in joint_positions]
        ax_line.set_xdata(ys)
        ax_line.set_ydata(zs)

def add_environment_2d_top(ax):
    """Add a ground rectangle and sample environment boxes in a top view."""
    ground_width = x_limits[1] - x_limits[0]
    ground_height = y_limits[1] - y_limits[0]
    ground_rect = Rectangle((x_limits[0], y_limits[0]), ground_width, ground_height,
                             facecolor='peru', alpha=0.3)
    ax.add_patch(ground_rect)

    # Define sample boxes
    boxes = [
        {"center": (-0.1522, -0.0485), "dims": (29.5/100, 21/100)},
        {"center": (0.5802, 0.2528),   "dims": (21/100, 29.5/100)},
        {"center": (0.5802, 0.5669),   "dims": (21/100, 29.5/100)},
        {"center": (0.242, -0.0485),   "dims": (45/100, 29.5/100)},
    ]
    for box in boxes:
        cx, cy = box["center"]
        w, h = box["dims"]
        x_left = cx - w/2
        y_bottom = cy - h/2
        rect = Rectangle((x_left, y_bottom), w, h,
                         facecolor='darkblue', alpha=0.5, edgecolor='k')
        ax.add_patch(rect)
