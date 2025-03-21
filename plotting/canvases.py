# plotting/canvases.py
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Multiprocessing safe backend
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PyQt5.QtWidgets import QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from .config import x_limits, y_limits, z_limits, z_level

from .helpers import add_environment_2d_top

class Plot3DCanvas(FigureCanvas):
    def __init__(self, parent=None, view_angle=None):
        self.fig = Figure()
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Y [m]")
        self.ax.set_zlabel("Z [m]")
        self.ax.set_xlim(x_limits)
        self.ax.set_ylim(y_limits)
        self.ax.set_zlim(z_limits)

        if view_angle:
            self.ax.view_init(elev=view_angle[0], azim=view_angle[1])
        try:
            self.ax.set_box_aspect(None)
        except Exception:
            pass

        # Draw ground plane
        xx, yy = np.meshgrid(np.linspace(x_limits[0], x_limits[1], 10),
                             np.linspace(y_limits[0], y_limits[1], 10))
        zz = np.full_like(xx, z_level)
        self.ax.plot_surface(xx, yy, zz, color='peru', alpha=0.3, rstride=100, cstride=100)

        # Add sample environment boxes
        box_positions = [(-0.1522, -0.0485), (0.5802, 0.2528), (0.5802, 0.5669), (0.242, -0.0485)]
        box_dims = [(29.5/100, 21/100), (21/100, 29.5/100), (21/100, 29.5/100), (45/100, 29.5/100)]
        for (cx, cy), dims in zip(box_positions, box_dims):
            w, h = dims
            xs = [cx - w/2, cx + w/2, cx + w/2, cx - w/2, cx - w/2]
            ys = [cy - h/2, cy - h/2, cy + h/2, cy + h/2, cy - h/2]
            zs = [z_level] * 5
            verts = [list(zip(xs, ys, zs))]
            poly = Poly3DCollection(verts, facecolors='darkblue', alpha=0.5, edgecolors='k')
            self.ax.add_collection3d(poly)

        self.fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.15)
        self.robot_line = self.ax.plot([], [], [], 'ko-', lw=2)[0]
        self.trajectory_lines = []

    def add_trajectory_line(self, color='black', style='-', lw=1.5, alpha=1.0):
        line = self.ax.plot([], [], [], linestyle=style, color=color, linewidth=lw, alpha=alpha)[0]
        self.trajectory_lines.append(line)
        return line

class Plot2DCanvas(FigureCanvas):
    def __init__(self, parent=None, view='top'):
        self.fig = Figure()
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.ax = self.fig.add_subplot(111)
        self.view = view

        if view == 'top':
            self.ax.set_xlabel("X [m]")
            self.ax.set_ylabel("Y [m]")
            self.ax.set_xlim(x_limits)
            self.ax.set_ylim(y_limits)
            add_environment_2d_top(self.ax)
        elif view == 'front':
            self.ax.set_xlabel("X [m]")
            self.ax.set_ylabel("Z [m]")
            self.ax.set_xlim(x_limits)
            self.ax.set_ylim(z_limits)
        elif view == 'side':
            self.ax.set_xlabel("Y [m]")
            self.ax.set_ylabel("Z [m]")
            self.ax.set_xlim(y_limits)
            self.ax.set_ylim(z_limits)

        self.fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
        self.robot_line_2d, = self.ax.plot([], [], 'ko-', lw=2)
        self.trajectory_lines_2d = []

    def add_trajectory_line_2d(self, color='black', style='-', lw=1.5, alpha=1.0):
        line2d, = self.ax.plot([], [], linestyle=style, color=color, linewidth=lw, alpha=alpha)
        self.trajectory_lines_2d.append(line2d)
        return line2d

    def update_robot(self, joint_angles):
        from helpers import show_robot_2d  # local import to avoid circular dependency
        show_robot_2d(self.robot_line_2d, joint_angles, view=self.view)
