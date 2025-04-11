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
matplotlib.use('Agg')  # Use a backend safe for multiprocessing
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# -------------------------
# CONFIGURATION / GLOBALS
# -------------------------
z_level = -0.155
x_limits = (-0.35, 0.81)
y_limits = (-0.25, 0.75)
z_limits = (z_level, 0.49)

SITE_COLOR = 'darkblue'
SITE_ALPHA = 0.5

ELEMENT_COLORS = {
    "Scanning site":        "royalblue",
    "Foundation":           "forestgreen",
    "Searching element":    "crimson",
    "Wall 1":               "gold",
    "Wall 2":               "darkgoldenrod",
    "Floor 1":              "indigo",
    "Floor 2":              "orchid",
    "Bathroom module 1":    "chocolate",
    "Bathroom module 2":    "sienna",
    "Bathroom module 3":    "tan",
}

line_styles = {"grab": "-", "place": "-"}
line_widths = {"grab": 5.5, "place": 1.5}
line_alpha = {"grab": 1.0, "place": 0.7}

ZMQ_ADDRESS = "tcp://127.0.0.1:5555"
SUBSCRIBE_TOPIC = ""
PACKET_SKIP = 10

# -------------------------
# FORWARD KINEMATICS HELPERS (copied from your old script)
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
    """Return the individual transformation matrix for each joint."""
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),               np.cos(alpha),               d],
        [0,              0,                           0,                           1]
    ])

def forward_kinematics(joint_angles):
    """Compute all joint positions in 3D."""
    translation = np.array([0.02099, 0.34301, -0.30002])
    positions = [np.array([0, 0, DH_params[0]['d']]) + translation]
    T = np.eye(4)
    for i, params in enumerate(DH_params):
        theta = params['theta'] + joint_angles[i]
        T_joint = transformation_matrix(params['a'], params['alpha'], params['d'], theta)
        T = T @ T_joint
        x, y, z = T[0, 3], T[1, 3], T[2, 3]
        positions.append(np.array([x, y, z]) + translation)
    return positions

# -------------------------
# HELPER DRAWING FUNCTIONS
# -------------------------
def draw_ground(ax, is3d=False):
    if is3d:
        xx, yy = np.meshgrid(np.linspace(x_limits[0], x_limits[1], 10),
                             np.linspace(y_limits[0], y_limits[1], 10))
        zz = np.full_like(xx, z_level)
        ax.plot_surface(xx, yy, zz, color='peru', alpha=0.3, rstride=100, cstride=100)
    else:
        rect = matplotlib.patches.Rectangle((x_limits[0], y_limits[0]),
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
            rect_z = [z_level]*5
            verts = [list(zip(rect_x, rect_y, rect_z))]
            poly = Poly3DCollection(verts, facecolors=SITE_COLOR, alpha=SITE_ALPHA, edgecolors='k')
            ax.add_collection3d(poly)
        else:
            w, h = dims
            x_left = x_center - w/2
            y_bottom = y_center - h/2
            rect = matplotlib.patches.Rectangle((x_left, y_bottom), w, h,
                                                facecolor=SITE_COLOR,
                                                alpha=SITE_ALPHA,
                                                edgecolor='k')
            ax.add_patch(rect)

# Modified draw_robot that checks for 1D joint data and computes 3D positions if needed.
def draw_robot(ax, joints, view='3d'):
    joints = np.array(joints)
    # If joints is a 1D array, assume it's a list of joint angles and compute 3D positions.
    if joints.ndim == 1:
        joints = forward_kinematics(joints)
        # Optionally add a ground anchor, as in your old code:
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

def process_incoming_data(data, translation=np.array([0.02099, 0.34301, -0.30002])):
    coords = np.array(data.get("coordinates", [0,0,0]), dtype=float) + translation
    joints = data.get("joints", None)
    return coords, joints

# -------------------------
# PLOTTING WORKER (multiprocessing)
# -------------------------
def plot_view_worker(input_queue, output_queue, view):
    fig = Figure(figsize=(4,3))
    if view == '3d':
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_zlim(z_limits)
        draw_ground(ax, is3d=True)
        draw_sites(ax, view='3d')
    else:
        ax = fig.add_subplot(111)
        if view == 'top':
            ax.set_xlim(x_limits)
            ax.set_ylim(y_limits)
        elif view == 'front':
            ax.set_xlim(x_limits)
            ax.set_ylim(z_limits)
        elif view == 'side':
            ax.set_xlim(y_limits)
            ax.set_ylim(z_limits)
        draw_ground(ax, is3d=False)
        draw_sites(ax, view='2d')
    
    traj_x, traj_y, traj_z = [], [], []
    robot_joints = None

    # Render an initial image so that UI isn't blank.
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    output_queue.put(buf.read())
    buf.close()

    while True:
        try:
            data = input_queue.get(timeout=0.1)
            coords, joints = process_incoming_data(data)
            traj_x.append(coords[0])
            traj_y.append(coords[1])
            traj_z.append(coords[2])
            if joints is not None:
                # Convert joint angles to 3D positions if needed.
                temp = np.array(joints)
                if temp.ndim == 1:
                    robot_joints = forward_kinematics(temp)
                    ground_anchor = np.array([0, 0.34301, -0.2])
                    robot_joints = [ground_anchor] + robot_joints
                else:
                    robot_joints = joints

            ax.cla()
            if view == '3d':
                ax.set_xlim(x_limits)
                ax.set_ylim(y_limits)
                ax.set_zlim(z_limits)
                draw_ground(ax, is3d=True)
                draw_sites(ax, view='3d')
                ax.plot(traj_x, traj_y, traj_z, color='royalblue', lw=1.5)
                if robot_joints is not None:
                    draw_robot(ax, robot_joints, view='3d')
            else:
                if view == 'top':
                    ax.set_xlim(x_limits)
                    ax.set_ylim(y_limits)
                    draw_ground(ax, is3d=False)
                    draw_sites(ax, view='2d')
                    ax.plot(traj_x, traj_y, color='royalblue', lw=1.5)
                    if robot_joints is not None:
                        draw_robot(ax, robot_joints, view='top')
                elif view == 'front':
                    ax.set_xlim(x_limits)
                    ax.set_ylim(z_limits)
                    draw_ground(ax, is3d=False)
                    ax.plot(traj_x, traj_z, color='royalblue', lw=1.5)
                    if robot_joints is not None:
                        draw_robot(ax, robot_joints, view='front')
                elif view == 'side':
                    ax.set_xlim(y_limits)
                    ax.set_ylim(z_limits)
                    draw_ground(ax, is3d=False)
                    ax.plot(traj_y, traj_z, color='royalblue', lw=1.5)
                    if robot_joints is not None:
                        draw_robot(ax, robot_joints, view='side')
            canvas = FigureCanvasAgg(fig)
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
        self.resize(1400, 800)
        self.plot_output_queues = plot_output_queues

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

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

        self.legend_area = QScrollArea()
        self.legend_area.setWidgetResizable(True)
        self.legend_widget = QWidget()
        self.legend_area.setWidget(self.legend_widget)
        v_leg_layout = QVBoxLayout(self.legend_widget)
        title = QLabel("<b>Legend</b>")
        title.setStyleSheet("font-size: 16px; margin-bottom: 10px;")
        v_leg_layout.addWidget(title)
        for element, color in ELEMENT_COLORS.items():
            for state, suffix in [('grab', ' (forward)'), ('place', ' (backward)')]:
                h_layout = QHBoxLayout()
                fig = Figure(figsize=(1, 0.2))
                ax = fig.add_axes([0, 0, 1, 1])
                line = matplotlib.lines.Line2D([0, 1], [0.5, 0.5],
                                                 color=color,
                                                 linestyle=line_styles.get(state, "-"),
                                                 linewidth=line_widths.get(state, 1.5),
                                                 alpha=line_alpha.get(state, 1.0))
                ax.add_line(line)
                ax.axis("off")
                canvas = FigureCanvasAgg(fig)
                canvas.draw()
                buf = BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)
                qt_img = QImage.fromData(buf.getvalue())
                buf.close()
                pixmap = QPixmap.fromImage(qt_img).scaled(50, 20, Qt.KeepAspectRatio)
                img_label = QLabel()
                img_label.setPixmap(pixmap)
                lbl = QLabel(f"{element}{suffix}")
                lbl.setStyleSheet("font-size: 14px; margin-left: 5px;")
                h_layout.addWidget(img_label)
                h_layout.addWidget(lbl)
                h_layout.addStretch()
                v_leg_layout.addLayout(h_layout)
        v_leg_layout.addStretch()

        main_layout.addWidget(self.plot_widget, 4)
        main_layout.addWidget(self.legend_area, 1)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(100)
        self.data_dispatcher = data_dispatcher

    def update_plots(self):
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
# DATA DISPATCHER
# -------------------------
def data_dispatcher_func(zmq_data_queue, plot_input_queues):
    while not zmq_data_queue.empty():
        try:
            data = zmq_data_queue.get_nowait()
            for q in plot_input_queues.values():
                q.put(data)
        except Exception as e:
            print("Error in dispatcher:", e)
            continue

def zmq_receiver(data_queue):
    packet_counter = 0
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://127.0.0.1:5555")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    socket.setsockopt(zmq.RCVTIMEO, 1000)
    while True:
        try:
            packed_data = socket.recv(flags=0)
            packet_counter += 1
            if packet_counter % 10 == 0:
                data = msgpack.unpackb(packed_data, raw=False)
                print("Receiver got data:", data)
                data_queue.put(data)
        except zmq.Again:
            continue
        except zmq.ZMQError as e:
            print("ZMQ error:", e)
            break

# -------------------------
# ENTRY POINT
# -------------------------
if __name__ == '__main__':
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
