#plotting/plotting.py
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
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D

# Import configuration constants from your config scripts.
from configs.construction_config import *
from configs.system_config       import *

# -------------------------
# CONFIGURATION / GLOBALS
# -------------------------
SCALING_FACTOR = 0.75  

# (For plotting in this script we use these axes limits.)
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

# -------------------------
# PLOTTING WORKER (for multiprocessing)
# -------------------------
def plot_view_worker(input_q, output_q, view,barrier):
    # --- static scene ---------------------------------------------------
    fig = Figure(figsize=(4, 3), dpi=100 * SCALING_FACTOR)
    if view == "3d":
        from mpl_toolkits.mplot3d import Axes3D  # noqa
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim(x_limits); ax.set_ylim(y_limits); ax.set_zlim(z_limits)
        ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
        draw_ground(ax, is3d=True); draw_sites(ax, view="3d")
    else:
        ax = fig.add_subplot(111)
        if view == "top":
            ax.set_xlim(x_limits); ax.set_ylim(y_limits)
            ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
            draw_ground(ax, is3d=False); draw_sites(ax, view="2d")
        elif view == "front":
            ax.set_xlim(x_limits); ax.set_ylim(z_limits)
            ax.set_xlabel("X [m]"); ax.set_ylabel("Z [m]")
        else:                                    # side
            ax.set_xlim(y_limits); ax.set_ylim(z_limits)
            ax.set_xlabel("Y [m]"); ax.set_ylabel("Z [m]")

    canvas   = FigureCanvasAgg(fig)
    canvas.draw()
    clean_bg = canvas.copy_from_bbox(ax.bbox)      # robot-free background

    # dynamic artists
    if view == "3d":
        robot_line, = ax.plot([], [], [], "ko-", lw=2)
        worker_scat = ax.scatter([], [], [], s=50, depthshade=False)
    else:
        robot_line, = ax.plot([], [], "ko-", lw=2)
        worker_scat = ax.scatter([], [], s=50)

    # keep last point & style info
    prev_coords = None
    prev_col    = None
    prev_ls     = "-"
    prev_lw     = 2.5

    # ------------------ main loop ---------------------------------------
    while True:
        pkt = input_q.get()                        # wait for data
        coords, joints = process_incoming_data(pkt)
        element = normalize_element_name(pkt.get("element"))
        state   = (pkt.get("state") or "").strip()

        # -------- 1) restore background ---------------------------------
        canvas.restore_region(clean_bg)

        # -------- 2) draw ONE new segment -------------------------------
        if prev_coords is not None:
            x0, y0, z0 = prev_coords
            x1, y1, z1 = coords
            if view == "3d":
                ax.plot([x0, x1], [y0, y1], [z0, z1],
                        color=prev_col, lw=prev_lw, ls=prev_ls)
            elif view == "top":
                ax.plot([x0, x1], [y0, y1],
                        color=prev_col, lw=prev_lw, ls=prev_ls)
            elif view == "front":
                ax.plot([x0, x1], [z0, z1],
                        color=prev_col, lw=prev_lw, ls=prev_ls)
            else:  # side
                ax.plot([y0, y1], [z0, z1],
                        color=prev_col, lw=prev_lw, ls=prev_ls)

            canvas.draw()                          # rasterise that line
            clean_bg = canvas.copy_from_bbox(ax.bbox)   # new background

        # decide style for NEXT segment
        if element.lower() == "scanning site":
            sty = SCANNING_STYLE
            prev_col = ELEMENT_COLORS.get(element, "black")
            prev_ls, prev_lw = sty["linestyle"], sty["linewidth"]
        else:
            prev_col = ELEMENT_COLORS.get(element, "black")
            d = STATE_STYLES_ASSEMBLY.get(state, {"linestyle": "-", "linewidth": 2.5})
            prev_ls, prev_lw = d["linestyle"], d["linewidth"]

        prev_coords = coords

        # -------- 3) robot ----------------------------------------------
        if joints is not None:
            r = np.asarray(forward_kinematics(joints))
            r = np.vstack([np.array([0, 0.34301, -0.2]), r])   # add ground anchor
            if view == "3d":
                robot_line.set_data(r[:, 0], r[:, 1]); robot_line.set_3d_properties(r[:, 2])
            elif view == "top":
                robot_line.set_data(r[:, 0], r[:, 1])
            elif view == "front":
                robot_line.set_data(r[:, 0], r[:, 2])
            else:
                robot_line.set_data(r[:, 1], r[:, 2])

        # -------- 4) workers (optional) ---------------------------------
        if pkt.get("worker spotted", False):
            wid = pkt.get("worker id")
            col = WORKER_COLORS.get(wid, "black")
            wx, wy, wz = pkt.get("worker coordinates", [0, 0, 0])
            wz = z_level + 0.05
            if view == "3d":
                worker_scat._offsets3d = ([wx], [wy], [wz]); worker_scat.set_color([col])
            elif view == "top":
                worker_scat.set_offsets([[wx, wy]]); worker_scat.set_color([col])
            elif view == "front":
                worker_scat.set_offsets([[wx, wz]]); worker_scat.set_color([col])
            else:
                worker_scat.set_offsets([[wy, wz]]); worker_scat.set_color([col])

        # -------- 5) blit & ship PNG ------------------------------------
        ax.draw_artist(robot_line); ax.draw_artist(worker_scat)
        canvas.blit(ax.bbox); canvas.flush_events()

        buf = BytesIO(); fig.savefig(buf, format="png"); buf.seek(0)
        output_q.put(buf.read()); buf.close()

        barrier.wait()


# -------------------------
# DATA DISPATCHER FUNCTION
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

# -------------------------
# ZMQ RECEIVER FUNCTION
# -------------------------
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
                data_queue.put(data)
        except zmq.Again:
            continue
        except zmq.ZMQError as e:
            print("ZMQ error:", e)
            break

# # -------------------------
# # MAIN WINDOW (GUI)
# # -------------------------
class MainWindow(QMainWindow):
    def __init__(self, zmq_data_queue, plot_input_queues, plot_output_queues):
        super().__init__()
        self.setWindowTitle("Real-time Trajectory Views")
        self.resize(int(1400 * SCALING_FACTOR), int(800 * SCALING_FACTOR))

        # keep references
        self.zmq_data_queue     = zmq_data_queue
        self.plot_input_queues  = plot_input_queues
        self.plot_output_queues = plot_output_queues

        # CENTRAL WIDGET + LAYOUT
        central = QWidget(self)
        self.setCentralWidget(central)
        main_v = QVBoxLayout(central)

        # 1) Status bar at top
        self.status_label = QLabel("Waiting for data…")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(
            "font-size: 18px; padding: 6px; background: #333; color: white;"
        )
        main_v.addWidget(self.status_label)

        # 2) Plot grid + legend side-by-side
        body_h = QHBoxLayout()
        main_v.addLayout(body_h)

        # 2a) 2×2 grid of views
        self.plot_widget = QWidget()
        grid = QGridLayout(self.plot_widget)
        self.labels = {}
        for pos, view in zip([(0,0),(0,1),(1,0),(1,1)], ['3d','top','front','side']):
            lbl = QLabel()
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            lbl.setStyleSheet("background-color: lightgray;")
            grid.addWidget(lbl, *pos)
            self.labels[view] = lbl
        body_h.addWidget(self.plot_widget, 4)

        # 2b) Legend on the right
        self.legend_area   = QScrollArea()
        self.legend_area.setWidgetResizable(True)
        self.legend_widget = QWidget()
        self.legend_area.setWidget(self.legend_widget)
        v_leg_layout = QVBoxLayout(self.legend_widget)

        # Legend Title
        title = QLabel("<b>Legend</b>")
        title.setStyleSheet("font-size: 16px; margin-bottom: 10px;")
        v_leg_layout.addWidget(title)

        default_states = ALLOWED_STATES_ASSEMBLY
        for element, color in sorted(ELEMENT_COLORS.items()):
            # Scanning‐site style
            if element.lower() == "scanning site":
                h = QHBoxLayout()
                fig = Figure(figsize=(1*SCALING_FACTOR, 0.2*SCALING_FACTOR))
                ax  = fig.add_axes([0,0,1,1])
                style = SCANNING_STYLE
                line  = Line2D([0,1],[0.5,0.5],
                               color=color,
                               linestyle=style["linestyle"],
                               linewidth=style["linewidth"],
                               alpha=0.75)
                ax.add_line(line)
                ax.axis("off")

                canvas = FigureCanvasAgg(fig); canvas.draw()
                buf    = BytesIO(); fig.savefig(buf, format="png"); buf.seek(0)
                qt_img = QImage.fromData(buf.getvalue()); buf.close()
                pix    = QPixmap.fromImage(qt_img).scaled(
                            int(50*SCALING_FACTOR), int(20*SCALING_FACTOR),
                            Qt.KeepAspectRatio)

                img_lbl = QLabel(); img_lbl.setPixmap(pix)
                txt_lbl = QLabel(f"{element} (scanning)")
                txt_lbl.setStyleSheet("font-size: 14px; margin-left: 5px;")

                h.addWidget(img_lbl); h.addWidget(txt_lbl); h.addStretch()
                v_leg_layout.addLayout(h)

            # Assembly states
            else:
                for state in default_states:
                    h = QHBoxLayout()
                    fig = Figure(figsize=(1*SCALING_FACTOR, 0.2*SCALING_FACTOR))
                    ax  = fig.add_axes([0,0,1,1])
                    style = STATE_STYLES_ASSEMBLY[state]
                    line  = Line2D([0,1],[0.5,0.5],
                                   color=color,
                                   linestyle=style["linestyle"],
                                   linewidth=style["linewidth"],
                                   alpha=0.75)
                    ax.add_line(line)
                    ax.axis("off")

                    canvas = FigureCanvasAgg(fig); canvas.draw()
                    buf    = BytesIO(); fig.savefig(buf, format="png"); buf.seek(0)
                    qt_img = QImage.fromData(buf.getvalue()); buf.close()
                    pix    = QPixmap.fromImage(qt_img).scaled(
                                int(50*SCALING_FACTOR), int(20*SCALING_FACTOR),
                                Qt.KeepAspectRatio)

                    img_lbl = QLabel(); img_lbl.setPixmap(pix)
                    txt_lbl = QLabel(f"{element} ({state})")
                    txt_lbl.setStyleSheet("font-size: 14px; margin-left: 5px;")

                    h.addWidget(img_lbl); h.addWidget(txt_lbl); h.addStretch()
                    v_leg_layout.addLayout(h)

            # Separator line
            sep_h = QHBoxLayout()
            fig_sep = Figure(figsize=(1*SCALING_FACTOR, 0.2*SCALING_FACTOR))
            ax_sep  = fig_sep.add_axes([0,0,1,1])
            sep_line = Line2D([0,1],[0.5,0.5],
                              color="gray", linewidth=2, linestyle="-", alpha=0.5)
            ax_sep.add_line(sep_line); ax_sep.axis("off")

            canvas_sep = FigureCanvasAgg(fig_sep); canvas_sep.draw()
            buf_sep    = BytesIO(); fig_sep.savefig(buf_sep, format="png"); buf_sep.seek(0)
            qt_img_sep = QImage.fromData(buf_sep.getvalue()); buf_sep.close()
            pix_sep    = QPixmap.fromImage(qt_img_sep).scaled(
                            int(50*SCALING_FACTOR), int(20*SCALING_FACTOR),
                            Qt.KeepAspectRatio)

            sep_lbl = QLabel(); sep_lbl.setPixmap(pix_sep)
            txt_sep = QLabel("________________________________")
            h_sep = QHBoxLayout()
            h_sep.addWidget(sep_lbl); h_sep.addWidget(txt_sep); h_sep.addStretch()
            v_leg_layout.addLayout(h_sep)

        v_leg_layout.addStretch()
        body_h.addWidget(self.legend_area, 1)

        # TIMER
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(100)

    def update_plots(self):
        # 1) drain ZMQ → plot queues, remember last pkt
        last_pkt = None
        while not self.zmq_data_queue.empty():
            pkt = self.zmq_data_queue.get_nowait()
            last_pkt = pkt
            for q in self.plot_input_queues.values():
                q.put(pkt)

        # 2) status bar
        if last_pkt is not None:
            el = normalize_element_name(last_pkt.get('element'))
            st = last_pkt.get('state','')
            self.status_label.setText(f"Element: {el}   |   State: {st}")

        # 3) update views
        for view, queue in self.plot_output_queues.items():
            while not queue.empty():
                try:
                    img_bytes = queue.get_nowait()
                    qt_img    = QImage.fromData(img_bytes)
                    pix       = QPixmap.fromImage(qt_img)
                    self.labels[view].setPixmap(pix)
                except Exception as e:
                    print(f"Error updating view {view}:", e)

# # -------------------------
# # ENTRY POINT
# # -------------------------
if __name__ == '__main__':
    mp.set_start_method('spawn')

    zmq_data_queue     = mp.Queue()
    views              = ['3d','top','front','side']
    plot_input_queues  = {v: mp.Queue() for v in views}
    plot_output_queues = {v: mp.Queue() for v in views}

    #One Barrier for the 4 worker processes
    sync_barrier = mp.Barrier(len(views), timeout=5)   # 5-s safety timeout

    # start ZMQ receiver
    receiver = mp.Process(target=zmq_receiver, args=(zmq_data_queue,))
    receiver.start()

    # start plot workers
    workers = []
    for v in views:
        p = mp.Process(
            target=plot_view_worker,
            args=(plot_input_queues[v], 
                  plot_output_queues[v], 
                  v,
                  sync_barrier)
        )
        p.start()
        workers.append(p)

    # launch Qt app
    app    = QApplication(sys.argv)
    window = MainWindow(zmq_data_queue, plot_input_queues, plot_output_queues)
    window.show()

    def cleanup():
        receiver.terminate()
        for p in workers:
            p.terminate()
        receiver.join()
        for p in workers:
            p.join()

    app.aboutToQuit.connect(cleanup)
    sys.exit(app.exec_())

