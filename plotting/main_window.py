# plotting/main_window.py
import sys
import numpy as np
import multiprocessing as mp
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QGridLayout, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QSizePolicy
)
from PyQt5.QtCore import QTimer
from matplotlib.lines import Line2D
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from .canvases import Plot3DCanvas, Plot2DCanvas
from .config import ELEMENT_COLORS, line_styles, line_widths, line_alpha, worker_colors, x_limits, y_limits, z_limits, z_level, RDP_TRIGGER_COUNT, RDP_TOLERANCE
from .helpers import rdp_3d, show_robot_3d, show_robot_2d

class MainWindow(QMainWindow):
    def __init__(self, data_queue):
        super().__init__()
        self.setWindowTitle("Real-time Robot Trajectory")
        self.resize(3000, 1000)
        self.data_queue = data_queue

        # Tracking state variables
        self.previous_element = None
        self.previous_state = None
        self.current_element = "default"
        self.current_state = "default"
        self.wall_count = 0
        self.floor_count = 0
        self.toilet_count = 0

        # Segment buffers for trajectories
        self.segment_xs, self.segment_ys, self.segment_zs = [], [], []
        self.ortho_segment_xs, self.ortho_segment_ys, self.ortho_segment_zs = [], [], []
        self.top_segment_xs, self.top_segment_ys = [], []
        self.front_segment_xs, self.front_segment_zs = [], []
        self.side_segment_ys, self.side_segment_zs = [], []

        # Current lines references
        self.current_trajectory_line_3d = None
        self.current_trajectory_line_ortho = None
        self.current_trajectory_line_top = None
        self.current_trajectory_line_front = None
        self.current_trajectory_line_side = None

        self.element_colors = dict(ELEMENT_COLORS)
        self.spotted_workers = []

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QGridLayout()
        central_widget.setLayout(layout)

        # Create canvases
        self.plot_main = Plot3DCanvas(self)
        self.current_trajectory_line_3d = self.plot_main.add_trajectory_line(
            color='royalblue', style='-', lw=1.5, alpha=1.0
        )

        self.plot_ortho = Plot3DCanvas(self, view_angle=(45, 45))
        self.current_trajectory_line_ortho = self.plot_ortho.add_trajectory_line(
            color='royalblue', style='-', lw=1.5, alpha=1.0
        )

        self.plot_top = Plot2DCanvas(self, view='top')
        self.current_trajectory_line_top = self.plot_top.add_trajectory_line_2d(
            color='royalblue', style='-', lw=1.5, alpha=1.0
        )

        self.plot_front = Plot2DCanvas(self, view='front')
        self.current_trajectory_line_front = self.plot_front.add_trajectory_line_2d(
            color='royalblue', style='-', lw=1.5, alpha=1.0
        )

        self.plot_side = Plot2DCanvas(self, view='side')
        self.current_trajectory_line_side = self.plot_side.add_trajectory_line_2d(
            color='royalblue', style='-', lw=1.5, alpha=1.0
        )

        # Create a scrollable legend
        self.legend_area = QScrollArea(self)
        self.legend_area.setWidgetResizable(True)
        self.legend_area.setStyleSheet("background-color: white;")
        self.legend_widget = QWidget()
        self.legend_widget.setStyleSheet("background-color: white;")
        self.legend_layout = QVBoxLayout(self.legend_widget)
        self.legend_area.setWidget(self.legend_widget)

        layout.addWidget(self.plot_main, 0, 0, 2, 8)
        layout.addWidget(self.plot_ortho, 0, 8, 1, 7)
        layout.addWidget(self.plot_top, 0, 15, 1, 7)
        layout.addWidget(self.plot_front, 1, 8, 1, 7)
        layout.addWidget(self.plot_side, 1, 15, 1, 7)
        layout.addWidget(self.legend_area, 0, 22, 2, 2)

        # Set layout stretch factors
        for c in range(24):
            layout.setColumnStretch(c, 1)
        layout.setRowStretch(0, 1)
        layout.setRowStretch(1, 1)

        self.create_static_legend()

        self.timer_global = QTimer(self)
        self.timer_global.timeout.connect(self.update_plots)
        self.timer_global.start(100)

    def resolve_element_name(self, raw_element):
        """Resolves a raw element string to a specific element name."""
        if raw_element is None or raw_element.strip() == "":
            return "default"
        raw_element = raw_element.lower().strip()
        mapping = {
            "foundation": "Foundation",
            "scanning site": "Scanning site",
            "searching element": "Searching element",
            "wall": "Wall",
            "floor": "Floor",
            "toilet": "Bathroom module"
        }
        for key, label in mapping.items():
            if key in raw_element:
                if key in ["scanning site", "searching element", "foundation"]:
                    return label
                if key == "wall":
                    self.wall_count = min(self.wall_count + 1, 2)
                    return f"{label} {self.wall_count}"
                if key == "floor":
                    self.floor_count = min(self.floor_count + 1, 2)
                    return f"{label} {self.floor_count}"
                if key == "toilet":
                    self.toilet_count = min(self.toilet_count + 1, 3)
                    return f"{label} {self.toilet_count}"
        return "default"

    def create_static_legend(self):
        """Creates a static legend with element names and states."""
        while self.legend_layout.count():
            child = self.legend_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        title_label = QLabel("<b>Legend</b>")
        title_label.setStyleSheet("font-size: 16px; margin-bottom: 10px;")
        self.legend_layout.addWidget(title_label)

        for element, color in ELEMENT_COLORS.items():
            for state, suffix in [('grab', ' (forward)'), ('place', ' (backward)')]:
                style = line_styles.get(state, "-")
                width = line_widths.get(state, 1.5)
                alpha = line_alpha.get(state, 1.0)
                legend_item_layout = QHBoxLayout()
                legend_item_widget = QWidget()
                legend_item_widget.setLayout(legend_item_layout)
                figure = Figure(figsize=(1, 0.2))
                canvas = FigureCanvas(figure)
                ax = figure.add_axes([0, 0, 1, 1])
                line = Line2D([0, 1], [0.5, 0.5],
                              color=color, linestyle=style, linewidth=width, alpha=alpha)
                ax.add_line(line)
                ax.axis("off")
                canvas.setFixedSize(50, 20)
                label_widget = QLabel(f"{element}{suffix}")
                label_widget.setStyleSheet("font-size: 14px; margin-left: 5px;")
                legend_item_layout.addWidget(canvas)
                legend_item_layout.addWidget(label_widget)
                legend_item_layout.addStretch()
                self.legend_layout.addWidget(legend_item_widget)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.legend_layout.addWidget(spacer)
        self.legend_widget.update()
        self.legend_widget.repaint()

    def finalize_segment(self):
        """Finalize the current trajectory segment and reset buffers."""
        if self.segment_xs:
            self.current_trajectory_line_3d = self.plot_main.add_trajectory_line(
                color=self.segment_color,
                style=self.segment_style,
                lw=self.segment_width,
                alpha=self.segment_alpha
            )
            self.segment_xs.clear()
            self.segment_ys.clear()
            self.segment_zs.clear()
        if self.ortho_segment_xs:
            self.current_trajectory_line_ortho = self.plot_ortho.add_trajectory_line(
                color=self.segment_color,
                style=self.segment_style,
                lw=self.segment_width,
                alpha=self.segment_alpha
            )
            self.ortho_segment_xs.clear()
            self.ortho_segment_ys.clear()
            self.ortho_segment_zs.clear()
        if self.top_segment_xs:
            self.current_trajectory_line_top = self.plot_top.add_trajectory_line_2d(
                color=self.segment_color,
                style=self.segment_style,
                lw=self.segment_width,
                alpha=self.segment_alpha
            )
            self.top_segment_xs.clear()
            self.top_segment_ys.clear()
        if self.front_segment_xs:
            self.current_trajectory_line_front = self.plot_front.add_trajectory_line_2d(
                color=self.segment_color,
                style=self.segment_style,
                lw=self.segment_width,
                alpha=self.segment_alpha
            )
            self.front_segment_xs.clear()
            self.front_segment_zs.clear()
        if self.side_segment_ys:
            self.current_trajectory_line_side = self.plot_side.add_trajectory_line_2d(
                color=self.segment_color,
                style=self.segment_style,
                lw=self.segment_width,
                alpha=self.segment_alpha
            )
            self.side_segment_ys.clear()
            self.side_segment_zs.clear()

    def process_data(self, data):
        TRANSLATION = np.array([0.02099, 0.34301, -0.30002])
        coords = np.array(data.get("coordinates", [0, 0, 0]), dtype=float) + TRANSLATION
        x, y, z = coords
        raw_element = data.get("element", "default").strip()
        new_state = data.get("state", "default").strip()

        resolved_element = self.resolve_element_name(raw_element)
        if self.previous_element is not None and (resolved_element != self.previous_element or new_state != self.previous_state):
            self.finalize_segment()

        self.current_element = resolved_element
        self.current_state = new_state
        self.segment_color = self.element_colors.get(self.current_element, "black")
        self.segment_style = line_styles.get(self.current_state, "-")
        self.segment_width = line_widths.get(self.current_state, 1.5)
        self.segment_alpha = line_alpha.get(self.current_state, 1.0)

        # Append to segment buffers
        self.segment_xs.append(x)
        self.segment_ys.append(y)
        self.segment_zs.append(z)
        self.ortho_segment_xs.append(x)
        self.ortho_segment_ys.append(y)
        self.ortho_segment_zs.append(z)
        self.top_segment_xs.append(x)
        self.top_segment_ys.append(y)
        self.front_segment_xs.append(x)
        self.front_segment_zs.append(z)
        self.side_segment_ys.append(y)
        self.side_segment_zs.append(z)

        # Update lines
        self.current_trajectory_line_3d.set_data(self.segment_xs, self.segment_ys)
        self.current_trajectory_line_3d.set_3d_properties(self.segment_zs)
        self.current_trajectory_line_ortho.set_data(self.ortho_segment_xs, self.ortho_segment_ys)
        self.current_trajectory_line_ortho.set_3d_properties(self.ortho_segment_zs)
        self.current_trajectory_line_top.set_xdata(self.top_segment_xs)
        self.current_trajectory_line_top.set_ydata(self.top_segment_ys)
        self.current_trajectory_line_front.set_xdata(self.front_segment_xs)
        self.current_trajectory_line_front.set_ydata(self.front_segment_zs)
        self.current_trajectory_line_side.set_xdata(self.side_segment_ys)
        self.current_trajectory_line_side.set_ydata(self.side_segment_zs)

        self.previous_element = self.current_element
        self.previous_state = self.current_state

        # Update robot if joint data exists
        if self.current_state and data.get("joints"):
            joints = np.array(data["joints"])
            show_robot_3d(self.plot_main.robot_line, joints)
            show_robot_3d(self.plot_ortho.robot_line, joints)
            self.plot_top.update_robot(joints)
            self.plot_front.update_robot(joints)
            self.plot_side.update_robot(joints)

        # Worker detection example
        if data.get('worker spotted', False):
            worker_id = data['worker id']
            if worker_id in worker_colors:
                current_worker = np.array(data.get('worker coordinates', (0, 0, 0)), dtype=float)
                threshold = 0.0
                too_close = any(worker_id == prev_id and np.linalg.norm(current_worker - np.array([px, py, pz])) < threshold
                                for prev_id, px, py, pz in self.spotted_workers)
                if not too_close:
                    self.spotted_workers.append((worker_id, current_worker[0], current_worker[1], z_level+0.05))
        return coords, data.get("joints", None)

    def simplify_data(self):
        if len(self.segment_xs) > RDP_TRIGGER_COUNT:
            points = np.column_stack((self.segment_xs, self.segment_ys, self.segment_zs))
            simplified = rdp_3d(points, RDP_TOLERANCE)
            self.segment_xs, self.segment_ys, self.segment_zs = simplified.T.tolist()

    def update_plots(self):
        if not self.data_queue.empty():
            data = self.data_queue.get()
            self.process_data(data)
            self.simplify_data()
            # Draw worker positions
            if self.spotted_workers:
                plotted = set()
                for worker_id, wx, wy, wz in self.spotted_workers:
                    color = worker_colors[worker_id]
                    label = f'Worker {worker_id}' if worker_id not in plotted else "_nolegend_"
                    self.plot_main.ax.scatter(wx, wy, wz, color=color, marker='x', s=50, label=label)
                    self.plot_ortho.ax.scatter(wx, wy, wz, color=color, marker='x', s=50, label=label)
                    self.plot_top.ax.scatter(wx, wy, color=color, marker='x', s=50, label=label)
                    plotted.add(worker_id)
            self.plot_main.draw()
            self.plot_ortho.draw()
            self.plot_top.draw()
            self.plot_front.draw()
            self.plot_side.draw()
