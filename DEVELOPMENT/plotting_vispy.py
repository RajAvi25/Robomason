#!/usr/bin/env python
from system_config import *
from construction_config import *

import sys
import base64
import json
import numpy as np
import threading
import time
import zmq
import msgpack
from vispy import scene
from vispy.color import Color  # For converting color names to RGBA tuples
from PyQt5 import QtWidgets, QtCore, QtGui, QtSvg
import websocket

# --- Forward Kinematics Helpers ---
def transformation_matrix(a, alpha, d, theta):
    """Return the individual transformation matrix for each joint."""
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),               np.cos(alpha),               d],
        [0,              0,                           0,                           1]
    ])

def forward_kinematics(joint_angles):
    """
    Compute all joint positions in 3D using the DH parameters.
    Each computed joint position is shifted by TRANSLATION.
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

# --- ZeroMQ Realtime Subscriber ---
class RealtimeSubscriber(QtCore.QObject):
    data_received = QtCore.pyqtSignal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect("tcp://127.0.0.1:5555")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self._running = True

    def start_listening(self):
        while self._running:
            try:
                msg = self.socket.recv()  # Blocking call
                data = msgpack.unpackb(msg, raw=False)
                self.data_received.emit(data)
            except Exception as e:
                print("Error receiving data:", e)
    
    def stop(self):
        self._running = False
        self.socket.close()
        self.context.term()

# --- Camera Panel ---
class CameraPanel(QtWidgets.QGroupBox):
    def __init__(self, title="Camera Feed"):
        super().__init__(title)
        self.setLayout(QtWidgets.QVBoxLayout())
        # Use a QLabel that scales its pixmap to fit the label size
        self.label = QtWidgets.QLabel()
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setScaledContents(True)
        self.layout().addWidget(self.label)
        
        self.latest_frame = None
        self.printed_message_once = False

        self.ws = websocket.WebSocketApp(
            "ws://localhost:9090",
            on_open=self.on_open,
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        self.ws_thread = threading.Thread(target=self.ws.run_forever, daemon=True)
        self.ws_thread.start()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_image)
        self.timer.start(100)

    def on_open(self, ws):
        print("[Subscriber] WebSocket connection opened.")
        subscribe_msg = {"op": "subscribe", "topic": "/image"}
        ws.send(json.dumps(subscribe_msg))
        print("[Subscriber] Sent subscribe message to /image")

    def on_message(self, ws, message):
        if not self.printed_message_once:
            print("[Subscriber] Received message.")
            self.printed_message_once = True
        try:
            data = json.loads(message)
            if 'msg' in data and 'data' in data['msg']:
                img_base64 = data['msg']['data']
                img_bytes = base64.b64decode(img_base64)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                import cv2
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    self.latest_frame = frame
        except Exception as e:
            print("[Subscriber] Error in on_message:", e)

    def on_error(self, ws, error):
        print("[Subscriber] WebSocket error:", error)

    def on_close(self, ws, code, msg):
        print("[Subscriber] WebSocket closed.")

    def update_image(self):
        if self.latest_frame is not None:
            rgb = self.latest_frame[..., ::-1].copy()
            h, w, ch = rgb.shape
            qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qimg)
            self.label.setPixmap(pixmap)
        else:
            self.label.setText("No camera feed")

from PyQt5 import QtCore, QtGui, QtWidgets, QtSvg

class ScalableSvgWidget(QtSvg.QSvgWidget):
    def __init__(self, svg_file, parent=None):
        super().__init__(parent)
        self.load(svg_file)

    def paintEvent(self, event):
        """Reimplemented to scale the SVG to the widget size while preserving aspect ratio."""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)

        renderer = self.renderer()
        if not renderer.isValid():
            return  # No valid SVG loaded

        # The default SVG size
        svg_size = renderer.defaultSize()
        # The widget's current size
        widget_size = self.size()

        # Figure out how much we need to scale to fit both width and height
        scale_w = widget_size.width() / svg_size.width()
        scale_h = widget_size.height() / svg_size.height()

        # To maintain aspect ratio, scale by the smaller factor
        scale_factor = min(scale_w, scale_h)
        painter.scale(scale_factor, scale_factor)

        # Now render the SVG
        renderer.render(painter)


# --- Legend Panel ---
# class LegendPanel(QtWidgets.QGroupBox):
#     def __init__(self, title="Legend"):
#         super().__init__(title)
#         self.setLayout(QtWidgets.QVBoxLayout())
#         svg_path = "_workingdata/_siteinfo/Legend/legend.svg"
#         self.svg_widget = QtSvg.QSvgWidget(svg_path)
#         # Fix the size so that the SVG is rescaled to a smaller footprint
#         self.svg_widget.setFixedSize(120, 120)
#         self.layout().addWidget(self.svg_widget)

class LegendPanel(QtWidgets.QGroupBox):
    def __init__(self, title="Legend"):
        super().__init__(title)
        layout = QtWidgets.QVBoxLayout(self)
        
        svg_path = "_workingdata/_siteinfo/Legend/legend.svg"
        
        # Use our ScalableSvgWidget
        self.svg_widget = ScalableSvgWidget(svg_path)
        # Let it expand freely
        self.svg_widget.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                      QtWidgets.QSizePolicy.Expanding)

        layout.addWidget(self.svg_widget)
        self.setLayout(layout)


# --- View Panel with Dynamic Elements ---
class ViewPanel(QtWidgets.QGroupBox):
    def __init__(self, title, azimuth, elevation):
        super().__init__(title)
        self.title_str = title
        self.setLayout(QtWidgets.QVBoxLayout())
        self.canvas = scene.SceneCanvas(keys='interactive', show=False, bgcolor='white', parent=self)
        self.canvas.create_native()
        self.canvas.native.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.layout().setSpacing(5)
        self.layout().setContentsMargins(5, 5, 5, 5)
        
        button_row = QtWidgets.QHBoxLayout()
        self.toggle_btn = QtWidgets.QPushButton("Toggle Camera")
        self.toggle_btn.setCheckable(True)
        self.reset_btn = QtWidgets.QPushButton("Reset View")
        button_row.addWidget(self.toggle_btn)
        button_row.addWidget(self.reset_btn)
        self.layout().addLayout(button_row)
        self.layout().addWidget(self.canvas.native)
        
        self.grid = self.canvas.central_widget.add_grid()
        self.view = self.grid.add_view()
        self.create_scene()
        self.camera = scene.cameras.TurntableCamera(fov=0, azimuth=azimuth, elevation=elevation,
                                                    parent=self.view.scene)
        self.camera.set_range(x=x_limits, y=y_limits, z=z_limits)
        self.view.camera = self.camera
        self.camera.interactive = False
        self.default_azimuth = azimuth
        self.default_elevation = elevation
        self.toggle_btn.toggled.connect(self.toggle_camera)
        self.reset_btn.clicked.connect(self.reset_camera)
        
        self.robot_line = scene.visuals.Line(
            pos=np.empty((0, 3)), color='red', width=2, parent=self.view.scene
        )
        self.robot_line.set_gl_state(
            depth_test=True, cull_face=False,
            polygon_offset_fill=True, polygon_offset=(1, 1)
        )
        
        self.trajectory_segments = []
        self.current_segment_points = []
        self.current_segment_line = None
        
        self.prev_element = None
        self.prev_state = None
        
        self.worker_markers = {}

    def toggle_camera(self, checked):
        self.camera.interactive = checked

    def reset_camera(self):
        self.camera.azimuth = self.default_azimuth
        self.camera.elevation = self.default_elevation
        self.camera.set_range(x=x_limits, y=y_limits, z=z_limits)

    def create_scene(self):
        ground_z = z_limits[0]
        ground_vertices = np.array([
            [x_limits[0], y_limits[0], ground_z],
            [x_limits[1], y_limits[0], ground_z],
            [x_limits[1], y_limits[1], ground_z],
            [x_limits[0], y_limits[1], ground_z]
        ], dtype=np.float32)

        fill_color = (0.85, 0.7, 0.5, 0.3)
        plane_mesh = scene.visuals.Mesh(vertices=ground_vertices,
                                        faces=np.array([[0,1,2],[0,2,3]]),
                                        color=fill_color,
                                        shading='flat',
                                        parent=self.view.scene)
        plane_mesh.set_gl_state(depth_test=True, cull_face=False, blend=True,
                                polygon_offset_fill=True, polygon_offset=(3,3))
        
        border_vertices = np.vstack([ground_vertices, ground_vertices[0:1]])
        border_line = scene.visuals.Line(pos=border_vertices,
                                         color=(0.85,0.7,0.5,1),
                                         width=2,
                                         parent=self.view.scene)
        border_line.set_gl_state(depth_test=True, cull_face=False,
                                 polygon_offset_fill=True, polygon_offset=(3,3))
        
        minor_step = 0.1
        major_step = 0.5
        xs_minor = np.arange(x_limits[0], x_limits[1]+minor_step, minor_step)
        ys_minor = np.arange(y_limits[0], y_limits[1]+minor_step, minor_step)
        xs_major = np.arange(x_limits[0], x_limits[1]+major_step, major_step)
        ys_major = np.arange(y_limits[0], y_limits[1]+major_step, major_step)
        
        minor_lines_pos = []
        for x in xs_minor:
            if not any(abs(x - xm) < 1e-8 for xm in xs_major):
                minor_lines_pos.append([x, y_limits[0], ground_z])
                minor_lines_pos.append([x, y_limits[1], ground_z])
        for y in ys_minor:
            if not any(abs(y - ym) < 1e-8 for ym in ys_major):
                minor_lines_pos.append([x_limits[0], y, ground_z])
                minor_lines_pos.append([x_limits[1], y, ground_z])
        if minor_lines_pos:
            line_minor = scene.visuals.Line(
                pos=np.array(minor_lines_pos, dtype=np.float32),
                connect='segments',
                color=(0.5,0.5,0.5,0.4),
                width=0.5,
                parent=self.view.scene
            )
            line_minor.set_gl_state(depth_test=True, cull_face=False, blend=True,
                                    polygon_offset_fill=True, polygon_offset=(3,3))
        
        major_lines_pos = []
        for x in xs_major:
            major_lines_pos.append([x, y_limits[0], ground_z])
            major_lines_pos.append([x, y_limits[1], ground_z])
        for y in ys_major:
            major_lines_pos.append([x_limits[0], y, ground_z])
            major_lines_pos.append([x_limits[1], y, ground_z])
        line_major = scene.visuals.Line(
            pos=np.array(major_lines_pos, dtype=np.float32),
            connect='segments',
            color=(0.3,0.3,0.3,0.8),
            width=1.0,
            parent=self.view.scene
        )
        line_major.set_gl_state(depth_test=True, cull_face=False, blend=True,
                                polygon_offset_fill=True, polygon_offset=(3,3))
        
        # Tick labels (already placed with offset)
        label_offset = 0.03
        for x in xs_major:
            if self.title_str == "Side View":
                continue
            text = scene.visuals.Text(
                text=f"{x:.2f}",
                color='black',
                font_size=9,
                parent=self.view.scene,
                anchor_x='center',
                anchor_y='top'
            )
            text.pos = (x, y_limits[0]-label_offset, ground_z)
            text.set_gl_state(depth_test=False)
        
        for y in ys_major:
            if self.title_str == "Front View":
                continue
            text = scene.visuals.Text(
                text=f"{y:.2f}",
                color='black',
                font_size=9,
                parent=self.view.scene,
                anchor_x='right',
                anchor_y='center'
            )
            text.pos = (x_limits[0]-label_offset, y, ground_z)
            text.set_gl_state(depth_test=False)
        
        # Add axis labels for visible axes
        # Conditions:
        #   - X is visible unless view == "Side View"
        #   - Y is visible unless view == "Front View"
        #   - Z is visible unless view == "Top View"
        axis_offset = 0.05  # adjust offset as needed
        if self.title_str != "Side View":
            x_label = scene.visuals.Text(
                text="X[m]", color='black', font_size=10,
                parent=self.view.scene,
                anchor_x='right', anchor_y='bottom'
            )
            x_label.pos = (x_limits[1], y_limits[0]-axis_offset, ground_z)
            x_label.set_gl_state(depth_test=False)
        if self.title_str != "Front View":
            y_label = scene.visuals.Text(
                text="Y[m]", color='black', font_size=10,
                parent=self.view.scene,
                anchor_x='left', anchor_y='top'
            )
            y_label.pos = (x_limits[0]-axis_offset, y_limits[1], ground_z)
            y_label.set_gl_state(depth_test=False)
        if self.title_str != "Top View":
            z_label = scene.visuals.Text(
                text="Z[m]", color='black', font_size=10,
                parent=self.view.scene,
                anchor_x='left', anchor_y='bottom'
            )
            z_label.pos = (x_limits[0], y_limits[0], z_limits[1]+axis_offset)
            z_label.set_gl_state(depth_test=False)
        
        axis = scene.visuals.XYZAxis(parent=self.view.scene)
        axis_scale_factor = 0.8
        axis.transform = scene.transforms.STTransform(
            translate=(x_limits[0], y_limits[0], z_limits[0]),
            scale=((x_limits[1]-x_limits[0])*axis_scale_factor,
                   (y_limits[1]-y_limits[0])*axis_scale_factor,
                   (z_limits[1]-z_limits[0])*axis_scale_factor)
        )
        
        for pos, dims in zip(box_positions, box_dims):
            self.add_site(pos, dims)

    def add_site(self, pos, dims):
        z_site = z_limits[0]
        x, y = pos
        w, h = dims
        vertices = np.array([
            [x,   y,   z_site],
            [x+w, y,   z_site],
            [x+w, y+h, z_site],
            [x,   y+h, z_site]
        ], dtype=np.float32)
        faces = np.array([[0,1,2],[0,2,3]])
        base_rgba = Color(box_color).rgba
        color_rgba = (base_rgba[0], base_rgba[1], base_rgba[2], box_alpha)
        site_mesh = scene.visuals.Mesh(vertices=vertices,
                                       faces=faces,
                                       color=color_rgba,
                                       shading='flat',
                                       parent=self.view.scene)
        site_mesh.set_gl_state(depth_test=True, cull_face=False, blend=True,
                               polygon_offset_fill=True, polygon_offset=(2,2))

    def determine_segment_style(self, element, state):
        base_color = ELEMENT_COLORS.get(element, "green")
        rgba = Color(base_color).rgba
        alpha = line_alpha.get(state, 1.0)
        new_color = (rgba[0], rgba[1], rgba[2], alpha)
        width = line_widths.get(state, 2.0)
        return {"color": new_color, "width": width, "alpha": alpha}

    def finalize_trajectory_segment(self):
        if self.current_segment_points and self.current_segment_line is not None:
            self.trajectory_segments.append(self.current_segment_line)
            self.current_segment_line = None

    def update_trajectory(self, data):
        coordinate = data.get("coordinates", None)
        if coordinate is None:
            return
        point = np.array(coordinate, dtype=float) + TRANSLATION
        new_element = data.get("element", "default")
        new_state = data.get("state", "default")
        if self.prev_element is None:
            self.current_segment_style = self.determine_segment_style(new_element, new_state)
            self.prev_element = new_element
            self.prev_state = new_state
        elif new_element != self.prev_element or new_state != self.prev_state:
            self.finalize_trajectory_segment()
            self.current_segment_points = []
            self.current_segment_style = self.determine_segment_style(new_element, new_state)
            self.prev_element = new_element
            self.prev_state = new_state
        self.current_segment_points.append(point)
        pts = np.array(self.current_segment_points)
        if self.current_segment_line is None:
            self.current_segment_line = scene.visuals.Line(
                pos=pts,
                color=self.current_segment_style["color"],
                width=self.current_segment_style["width"],
                parent=self.view.scene
            )
            self.current_segment_line.set_gl_state(
                depth_test=True, cull_face=False,
                polygon_offset_fill=True, polygon_offset=(1,1)
            )
        else:
            self.current_segment_line.set_data(pos=pts)

    def update_worker_marker(self, worker_id, worker_coords):
        pos = np.array(worker_coords, dtype=float) + TRANSLATION
        base_color = WORKER_COLORS.get(worker_id, "yellow")
        color_rgba = Color(base_color).rgba
        marker_style = worker_marker_styles.get(worker_id, 'x')
        if marker_style == 'zone':
            size = WORKER_SQUARE_SIZE
            wx, wy, wz = pos
            vertices = np.array([
                [wx,       wy,       wz],
                [wx+size,  wy,       wz],
                [wx+size,  wy-size,  wz],
                [wx,       wy-size,  wz]
            ], dtype=np.float32)
            faces = np.array([[0,1,2],[0,2,3]])
            zone_marker = scene.visuals.Mesh(vertices=vertices,
                                             faces=faces,
                                             color=color_rgba,
                                             shading='flat',
                                             parent=self.view.scene)
            zone_marker.set_gl_state(depth_test=True, cull_face=False,
                                     polygon_offset_fill=True, polygon_offset=(0,0))
            if worker_id in self.worker_markers:
                old_marker = self.worker_markers[worker_id]
                old_marker.parent = None
            self.worker_markers[worker_id] = zone_marker
        else:
            if worker_id in self.worker_markers:
                marker = self.worker_markers[worker_id]
                marker.set_data(
                    pos[np.newaxis, :],
                    face_color=color_rgba,
                    size=10, symbol=marker_style
                )
            else:
                marker = scene.visuals.Markers()
                marker.set_data(
                    pos[np.newaxis, :],
                    face_color=color_rgba,
                    size=10, symbol=marker_style
                )
                marker.parent = self.view.scene
                self.worker_markers[worker_id] = marker
            marker.set_gl_state(depth_test=True, cull_face=False,
                                polygon_offset_fill=True, polygon_offset=(0,0))

    def update_robot_visual(self, joint_angles):
        pts = np.array(forward_kinematics(joint_angles))
        self.robot_line.set_data(pos=pts)

# --- Main Window ---
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Advanced 3D Viewer")
        
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        
        # Use a splitter for a resizable layout
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        
        # 1) Camera Panel (left)
        self.camera_panel = CameraPanel("Camera Feed")
        self.camera_panel.setMinimumWidth(400)
        splitter.addWidget(self.camera_panel)
        
        # 2) The 3D Views in the center
        center_widget = QtWidgets.QWidget()
        center_layout = QtWidgets.QGridLayout(center_widget)
        center_layout.setSpacing(10)
        center_layout.setContentsMargins(10, 10, 10, 10)

        panels = [
            ("Front View",      0,  0,  0,   0),
            ("Top View",        0,  1,  0,  90),
            ("Side View",       1,  0, 90,   0),
            ("Isometric View",  1,  1, 45,  45)
        ]
        self.view_panels = []
        for title, row, col, azimuth, elevation in panels:
            panel = ViewPanel(title, azimuth, elevation)
            self.view_panels.append(panel)
            center_layout.addWidget(panel, row, col)
        center_widget.setLayout(center_layout)
        splitter.addWidget(center_widget)
        
        # 3) Legend (right)
        self.legend_panel = LegendPanel("Legend")
        self.legend_panel.setMinimumWidth(100)
        self.legend_panel.setMaximumWidth(150)
        splitter.addWidget(self.legend_panel)
        
        splitter.setSizes([500, 1200, 100])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 8)
        splitter.setStretchFactor(2, 0)
        
        main_layout = QtWidgets.QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.addWidget(splitter)
        
        self.subscriber = RealtimeSubscriber()
        self.subscriber.data_received.connect(self.handle_new_data)
        threading.Thread(target=self.subscriber.start_listening, daemon=True).start()

        self.resize(1900, 1000)

    def handle_new_data(self, data):
        joints = data.get('joints', None)
        if joints is not None:
            for panel in self.view_panels:
                panel.update_robot_visual(joints)
        if data.get('coordinates', None) is not None:
            for panel in self.view_panels:
                panel.update_trajectory(data)
        if data.get('worker spotted', False):
            worker_id = data.get('worker id', None)
            worker_coords = data.get('worker coordinates', [0, 0, 0])
            for panel in self.view_panels:
                panel.update_worker_marker(worker_id, worker_coords)

def main():
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()

if __name__ == "__main__":
    main()
