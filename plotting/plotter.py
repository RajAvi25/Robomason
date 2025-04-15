"""
trajectory_module.py

A module that contains functionality for real-time trajectory visualization,
including forward kinematics, plotting workers using multiprocessing, and a
Qt-based GUI.
 
Dependencies:
    - zmq, msgpack, numpy, multiprocessing, PyQt5, matplotlib
    - configuration files (construction_config.py and system_config.py) that define
      constants like TRANSLATION, SITE_COLOR, ELEMENT_COLORS, etc.
      
Usage (as a script):
    python trajectory_module.py

Usage (as a module):
    from trajectory_module import forward_kinematics, draw_robot, MainWindow, run_app
"""

import sys
import zmq
import msgpack
import numpy as np
import multiprocessing as mp
from io import BytesIO
import time

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, QGridLayout,
    QLabel, QWidget, QScrollArea, QSizePolicy
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QPixmap, QImage

import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend safe for multiprocessing
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D

# Import configuration constants from your config scripts.
from construction_config import *
from system_config import *

# -------------------------
# CONFIGURATION / GLOBALS
# -------------------------
SCALING_FACTOR = 2.0  

# Axes limits for plotting.
z_level = -0.155
x_limits = (-0.35, 0.81)
y_limits = (-0.25, 0.75)
z_limits = (z_level, 0.49)

ZMQ_ADDRESS = "tcp://127.0.0.1:5555"
SUBSCRIBE_TOPIC = ""
PACKET_SKIP = 10

# -------------------------
# FORWARD KINEMATICS HELPERS
# -------------------------
# UR5 DH parameters
DH_params = [
    {'theta': 0, 'a': 0, 'd': 0.1625, 'alpha': np.pi / 2},
    {'theta': 0, 'a': -0.425, 'd': 0, 'alpha': 0},
    {'theta': 0, 'a': -0.3922, 'd': 0, 'alpha': 0},
    {'theta': 0, 'a': 0, 'd': 0.1333, 'alpha': np.pi / 2},
    {'theta': 0, 'a': 0, 'd': 0.0997, 'alpha': -np.pi / 2},
    {'theta': 0, 'a': 0, 'd': 0.0996, 'alpha': 0}
]

def transformation_matrix(a, alpha, d, theta):
    """Return the transformation matrix for given DH parameters."""
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),               np.cos(alpha),               d],
        [0,              0,                           0,                           1]
    ])

def forward_kinematics(joint_angles):
    """
    Compute all joint positions in 3D space given the robot joint angles.
    A constant translation (TRANSLATION from config) is applied.
    """
    positions = [np.array([0, 0, DH_params[0]['d']]) + TRANSLATION]
    T = np.eye(4)
    for i, params in enumerate(DH_params):
        theta = params['theta'] + joint_angles[i]
        T_joint = transformation_matrix(params['a'], params['alpha'], params['d'], theta)
        T = T @ T_joint
        x, y, z = T[0, 3], T[1, 3], T[2, 3]
        positions.append(np.array([x, y, z]) + TRANSLATION)
    return positions

# -------------------------
# PLOTTING HELPERS
# -------------------------
def draw_ground(ax, is3d=False):
    """Draw a ground plane on the given axis."""
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
    """Draw predefined site boxes on the axis."""
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
    """Draw the robot based on its joint positions in the specified view."""
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
    """
    Process incoming data packets to extract coordinates and joint information.
    A translation (TRANSLATION from config) is added to the coordinates.
    """
    coords = np.array(data.get("coordinates", [0, 0, 0]), dtype=float) + TRANSLATION
    joints = data.get("joints", None)
    return coords, joints

# -------------------------
# TRAJECTORY SEGMENTATION
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

# -------------------------
# PLOTTING WORKER (for multiprocessing)
# -------------------------
def plot_view_worker(input_queue, output_queue, view):
    """
    Worker process function that receives data from input_queue, updates the plot,
    and sends the new image bytes through output_queue.
    """
    fig = Figure(figsize=(4, 3), dpi=100 * SCALING_FACTOR)
    
    # Set up the axis based on the view.
    if view == '3d':
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_zlim(z_limits)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        fig.subplots_adjust(top=0.25)
        draw_ground(ax, is3d=True)
        draw_sites(ax, view='3d')
    else:
        ax = fig.add_subplot(111)
        if view == 'top':
            ax.set_xlim(x_limits)
            ax.set_ylim(y_limits)
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            draw_ground(ax, is3d=False)
            draw_sites(ax, view='2d')
        elif view == 'front':
            ax.set_xlim(x_limits)
            ax.set_ylim(z_limits)
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Z [m]")
        elif view == 'side':
            ax.set_xlim(y_limits)
            ax.set_ylim(z_limits)
            ax.set_xlabel("Y [m]")
            ax.set_ylabel("Z [m]")
    
    trajectory_data = []
    robot_joints = None
    spotted_workers = {}

    # Initial plot output.
    canvas = FigureCanvasAgg(fig)
    fig.tight_layout()
    canvas.draw()
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    output_queue.put(buf.read())
    buf.close()

    # Main update loop.
    while True:
        try:
            data = input_queue.get(timeout=0.1)
            coords, joints = process_incoming_data(data)
            trajectory_data.append({
                'coords': coords,
                'element': data.get('element', None),
                'state': data.get('state', None)
            })
            if joints is not None:
                temp = np.array(joints)
                if temp.ndim == 1:
                    robot_joints = forward_kinematics(temp)
                    ground_anchor = np.array([0, 0.34301, -0.2])
                    robot_joints = [ground_anchor] + robot_joints
                else:
                    robot_joints = joints

            if data.get('worker spotted', False):
                worker_id = data.get('worker id')
                if worker_id is not None and worker_id in WORKER_COLORS:
                    worker_coords = data.get('worker coordinates', [0, 0, 0])
                    spotted_workers[worker_id] = (worker_coords[0], worker_coords[1], z_level + 0.05)

            # Clear and redraw the axis.
            ax.cla()
            if view == '3d':
                ax.set_xlim(x_limits)
                ax.set_ylim(y_limits)
                ax.set_zlim(z_limits)
                ax.set_xlabel("X [m]")
                ax.set_ylabel("Y [m]")
                ax.set_zlabel("Z [m]")
                draw_ground(ax, is3d=True)
                draw_sites(ax, view='3d')
            else:
                if view == 'top':
                    ax.set_xlim(x_limits)
                    ax.set_ylim(y_limits)
                    ax.set_xlabel("X [m]")
                    ax.set_ylabel("Y [m]")
                    draw_ground(ax, is3d=False)
                    draw_sites(ax, view='2d')
                elif view == 'front':
                    ax.set_xlim(x_limits)
                    ax.set_ylim(z_limits)
                    ax.set_xlabel("X [m]")
                    ax.set_ylabel("Z [m]")
                elif view == 'side':
                    ax.set_xlim(y_limits)
                    ax.set_ylim(z_limits)
                    ax.set_xlabel("Y [m]")
                    ax.set_ylabel("Z [m]")
            
            segments = group_trajectory_points(trajectory_data)
            for xs, ys, zs, ls, col, lw, alpha in segments:
                if view == '3d':
                    ax.plot(xs, ys, zs, linestyle=ls, color=col, linewidth=lw, alpha=alpha)
                elif view == 'top':
                    ax.plot(xs, ys, linestyle=ls, color=col, linewidth=lw, alpha=alpha)
                elif view == 'front':
                    ax.plot(xs, zs, linestyle=ls, color=col, linewidth=lw, alpha=alpha)
                elif view == 'side':
                    ax.plot(ys, zs, linestyle=ls, color=col, linewidth=lw, alpha=alpha)
            
            if robot_joints is not None:
                draw_robot(ax, robot_joints, view=view)
                
            # Draw worker markers.
            if view == '3d':
                for wid, (wx, wy, wz) in spotted_workers.items():
                    marker = worker_marker_styles.get(wid, 'x')
                    color = WORKER_COLORS.get(wid, 'black')
                    if marker == 'zone':
                        size = WORKER_SQUARE_SIZE
                        sq_xs = [wx, wx + size, wx + size, wx, wx]
                        sq_ys = [wy, wy, wy - size, wy - size, wy]
                        sq_zs = [wz] * 5
                        verts = [list(zip(sq_xs, sq_ys, sq_zs))]
                        poly = Poly3DCollection(verts, facecolors=color, alpha=0.7, edgecolors=color)
                        ax.add_collection3d(poly)
                    else:
                        ax.scatter(wx, wy, wz, color=color, marker=marker, s=50, depthshade=False)
            else:
                if view == 'top':
                    for wid, (wx, wy, wz) in spotted_workers.items():
                        marker = worker_marker_styles.get(wid, 'x')
                        color = WORKER_COLORS.get(wid, 'black')
                        if marker == 'zone':
                            rect = Rectangle((wx, wy), WORKER_SQUARE_SIZE, WORKER_SQUARE_SIZE,
                                             facecolor=color, alpha=0.7, edgecolor=color)
                            ax.add_patch(rect)
                        else:
                            ax.scatter(wx, wy, color=color, marker=marker, s=50)
                elif view == 'front':
                    for wid, (wx, wy, wz) in spotted_workers.items():
                        marker = worker_marker_styles.get(wid, 'x')
                        color = WORKER_COLORS.get(wid, 'black')
                        if marker == 'zone':
                            rect = Rectangle((wx, wz), WORKER_SQUARE_SIZE, WORKER_SQUARE_SIZE,
                                             facecolor=color, alpha=0.7, edgecolor=color)
                            ax.add_patch(rect)
                        else:
                            ax.scatter(wx, wz, color=color, marker=marker, s=50)
                elif view == 'side':
                    for wid, (wx, wy, wz) in spotted_workers.items():
                        marker = worker_marker_styles.get(wid, 'x')
                        color = WORKER_COLORS.get(wid, 'black')
                        if marker == 'zone':
                            rect = Rectangle((wy, wz), WORKER_SQUARE_SIZE, WORKER_SQUARE_SIZE,
                                             facecolor=color, alpha=0.7, edgecolor=color)
                            ax.add_patch(rect)
                        else:
                            ax.scatter(wy, wz, color=color, marker=marker, s=50)
            
            # Render and output new plot.
            canvas = FigureCanvasAgg(fig)
            fig.tight_layout()
            canvas.draw()
            buf = BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            output_queue.put(buf.read())
            buf.close()
        except Exception as e:
            print(f"Error in worker for view '{view}':", e)
            continue

# -------------------------
# MAIN WINDOW (GUI)
# -------------------------
class MainWindow(QMainWindow):
    def __init__(self, plot_output_queues, data_dispatcher):
        super().__init__()
        self.setWindowTitle("Real-time Trajectory Views")
        self.resize(int(1400 * SCALING_FACTOR), int(800 * SCALING_FACTOR))
        self.plot_output_queues = plot_output_queues

        # Set up central widget and main layout.
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Plot widget: grid of views.
        self.plot_widget = QWidget()
        grid = QGridLayout(self.plot_widget)
        self.labels = {}
        for pos, view in zip([(0, 0), (0, 1), (1, 0), (1, 1)],
                             ['3d', 'top', 'front', 'side']):
            lbl = QLabel()
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            lbl.setStyleSheet("background-color: lightgray;")
            grid.addWidget(lbl, pos[0], pos[1])
            self.labels[view] = lbl

        # Legend widget.
        self.legend_area = QScrollArea()
        self.legend_area.setWidgetResizable(True)
        self.legend_widget = QWidget()
        self.legend_area.setWidget(self.legend_widget)
        v_leg_layout = QVBoxLayout(self.legend_widget)
        title = QLabel("<b>Legend</b>")
        title.setStyleSheet("font-size: 16px; margin-bottom: 10px;")
        v_leg_layout.addWidget(title)

        default_states = ALLOWED_STATES_ASSEMBLY  # e.g. ["search", "pick", "swing", "place", "swing_back"]
        for element, color in sorted(ELEMENT_COLORS.items()):
            if element.lower() == "scanning site":
                h_layout = QHBoxLayout()
                fig = Figure(figsize=(1 * SCALING_FACTOR, 0.2 * SCALING_FACTOR))
                ax = fig.add_axes([0, 0, 1, 1])
                style = SCANNING_STYLE
                line = Line2D([0, 1], [0.5, 0.5],
                              color=color,
                              linestyle=style["linestyle"],
                              linewidth=style["linewidth"],
                              alpha=0.75)
                ax.add_line(line)
                ax.axis("off")
                canvas = FigureCanvasAgg(fig)
                canvas.draw()
                buf = BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)
                qt_img = QImage.fromData(buf.getvalue())
                buf.close()
                pixmap = QPixmap.fromImage(qt_img).scaled(
                    int(50 * SCALING_FACTOR),
                    int(20 * SCALING_FACTOR),
                    Qt.KeepAspectRatio
                )
                img_label = QLabel()
                img_label.setPixmap(pixmap)
                lbl = QLabel(f"{element} (scanning)")
                lbl.setStyleSheet("font-size: 14px; margin-left: 5px;")
                h_layout.addWidget(img_label)
                h_layout.addWidget(lbl)
                h_layout.addStretch()
                v_leg_layout.addLayout(h_layout)
            else:
                for state in default_states:
                    h_layout = QHBoxLayout()
                    fig = Figure(figsize=(1 * SCALING_FACTOR, 0.2 * SCALING_FACTOR))
                    ax = fig.add_axes([0, 0, 1, 1])
                    style = STATE_STYLES_ASSEMBLY[state]
                    line = Line2D([0, 1], [0.5, 0.5],
                                  color=color,
                                  linestyle=style["linestyle"],
                                  linewidth=style["linewidth"],
                                  alpha=0.75)
                    ax.add_line(line)
                    ax.axis("off")
                    canvas = FigureCanvasAgg(fig)
                    canvas.draw()
                    buf = BytesIO()
                    fig.savefig(buf, format="png")
                    buf.seek(0)
                    qt_img = QImage.fromData(buf.getvalue())
                    buf.close()
                    pixmap = QPixmap.fromImage(qt_img).scaled(
                        int(50 * SCALING_FACTOR),
                        int(20 * SCALING_FACTOR),
                        Qt.KeepAspectRatio
                    )
                    img_label = QLabel()
                    img_label.setPixmap(pixmap)
                    lbl = QLabel(f"{element} ({state})")
                    lbl.setStyleSheet("font-size: 14px; margin-left: 5px;")
                    h_layout.addWidget(img_label)
                    h_layout.addWidget(lbl)
                    h_layout.addStretch()
                    v_leg_layout.addLayout(h_layout)
            sep_layout = QHBoxLayout()
            fig_sep = Figure(figsize=(1 * SCALING_FACTOR, 0.2 * SCALING_FACTOR))
            ax_sep = fig_sep.add_axes([0, 0, 1, 1])
            sep_line = Line2D([0, 1], [0.5, 0.5],
                              color="gray",
                              linewidth=2,
                              linestyle="-",
                              alpha=0.5)
            ax_sep.add_line(sep_line)
            ax_sep.axis("off")
            canvas_sep = FigureCanvasAgg(fig_sep)
            canvas_sep.draw()
            buf_sep = BytesIO()
            fig_sep.savefig(buf_sep, format="png")
            buf_sep.seek(0)
            qt_img_sep = QImage.fromData(buf_sep.getvalue())
            buf_sep.close()
            pixmap_sep = QPixmap.fromImage(qt_img_sep).scaled(
                int(50 * SCALING_FACTOR),
                int(20 * SCALING_FACTOR),
                Qt.KeepAspectRatio
            )
            h_layout_sep = QHBoxLayout()
            sep_label = QLabel()
            sep_label.setPixmap(pixmap_sep)
            h_layout_sep.addWidget(sep_label)
            h_layout_sep.addWidget(QLabel("________________________________"))
            h_layout_sep.addStretch()
            v_leg_layout.addLayout(h_layout_sep)
        v_leg_layout.addStretch()

        main_layout.addWidget(self.plot_widget, 4)
        main_layout.addWidget(self.legend_area, 1)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(100)
        self.data_dispatcher = data_dispatcher

    def update_plots(self):
        """Update all plot views by checking their output queues."""
        for view, queue in self.plot_output_queues.items():
            while not queue.empty():
                try:
                    img_bytes = queue.get_nowait()
                    qt_img = QImage.fromData(img_bytes)
                    pixmap = QPixmap.fromImage(qt_img)
                    self.labels[view].setPixmap(pixmap)
                except Exception as e:
                    print(f"Error updating view {view}:", e)
        self.data_dispatcher()

# -------------------------
# DATA DISPATCHER FUNCTION
# -------------------------
def data_dispatcher_func(zmq_data_queue, plot_input_queues):
    """
    Dispatch any new data from the ZMQ receiver into all plotting input queues.
    """
    while not zmq_data_queue.empty():
        try:
            data = zmq_data_queue.get_nowait()
            for q in plot_input_queues.values():
                q.put(data)
        except Exception as e:
            print("Error in dispatcher:", e)
            continue

# -------------------------
# ZMQ RECEIVER FUNCTION
# -------------------------
def zmq_receiver(data_queue):
    """
    Receives data packets via ZMQ and sends (every PACKET_SKIP-th packet) the unpacked
    data to the given multiprocessing queue.
    """
    packet_counter = 0
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(ZMQ_ADDRESS)
    socket.setsockopt_string(zmq.SUBSCRIBE, SUBSCRIBE_TOPIC)
    socket.setsockopt(zmq.RCVTIMEO, 1000)
    while True:
        try:
            packed_data = socket.recv(flags=0)
            packet_counter += 1
            if packet_counter % PACKET_SKIP == 0:
                data = msgpack.unpackb(packed_data, raw=False)
                data_queue.put(data)
        except zmq.Again:
            continue
        except zmq.ZMQError as e:
            print("ZMQ error:", e)
            break

# -------------------------
# ENTRY POINT
# -------------------------
def run_app():
    """
    Set up the multiprocessing components, start the ZMQ receiver, create the GUI,
    and start the Qt event loop.
    """
    mp.set_start_method('spawn')
    zmq_data_queue = mp.Queue()

    receiver_process = mp.Process(target=zmq_receiver, args=(zmq_data_queue,))
    receiver_process.start()

    views = ['3d', 'top', 'front', 'side']
    plot_input_queues = {v: mp.Queue() for v in views}
    plot_output_queues = {v: mp.Queue() for v in views}

    workers = []
    for v in views:
        p = mp.Process(target=plot_view_worker,
                       args=(plot_input_queues[v], plot_output_queues[v], v))
        p.start()
        workers.append(p)

    def dispatcher():
        data_dispatcher_func(zmq_data_queue, plot_input_queues)

    app = QApplication(sys.argv)
    window = MainWindow(plot_output_queues, dispatcher)
    window.show()

    def cleanup():
        receiver_process.terminate()
        for p in workers:
            p.terminate()
        receiver_process.join()
        for p in workers:
            p.join()

    app.aboutToQuit.connect(cleanup)
    sys.exit(app.exec_())

# Run the application if the module is executed as a script.
if __name__ == '__main__':
    run_app()
