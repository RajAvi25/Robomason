#plotting/workers.py
import numpy as np
import zmq
import msgpack
import multiprocessing as mp
from io import BytesIO
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from .utils import *

import os
import psutil
import queue

# from plotting.utils import (
#     draw_ground, draw_sites, process_incoming_data,
#     forward_kinematics, normalize_element_name,
#     SCALING_FACTOR, x_limits, y_limits, z_limits, z_level,
#     SCANNING_STYLE, ELEMENT_COLORS,
#     STATE_STYLES_ASSEMBLY, WORKER_COLORS
# )

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
    socket.setsockopt(zmq.SUBSCRIBE, b'')    # subscribe to all messages
    socket.setsockopt(zmq.CONFLATE, 1)       # keep only the latest message in socket buffer
    socket.setsockopt(zmq.RCVHWM, 1)         # drop old messages if not read
    # socket.connect("tcp://127.0.0.1:5555")
    socket.connect("ipc:///tmp/plotter.ipc")
    socket.setsockopt(zmq.RCVTIMEO, 1000)    # 1 s receive timeout

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

# def plot_view_worker(input_q, output_q, view, barrier):
#     # --- static scene ---------------------------------------------------
#     fig = Figure(figsize=(4, 3), dpi=100 * SCALING_FACTOR)
#     if view == "3d":
#         from mpl_toolkits.mplot3d import Axes3D  # noqa
#         ax = fig.add_subplot(111, projection="3d")
#         ax.set_xlim(x_limits); ax.set_ylim(y_limits); ax.set_zlim(z_limits)
#         ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
#         draw_ground(ax, is3d=True); draw_sites(ax, view="3d")
#     else:
#         ax = fig.add_subplot(111)
#         if view == "top":
#             ax.set_xlim(x_limits); ax.set_ylim(y_limits)
#             ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
#             draw_ground(ax, is3d=False); draw_sites(ax, view="2d")
#         elif view == "front":
#             ax.set_xlim(x_limits); ax.set_ylim(z_limits)
#             ax.set_xlabel("X [m]"); ax.set_ylabel("Z [m]")
#         else:  # side
#             ax.set_xlim(y_limits); ax.set_ylim(z_limits)
#             ax.set_xlabel("Y [m]"); ax.set_ylabel("Z [m]")

#     canvas = FigureCanvasAgg(fig)
#     canvas.draw()
#     clean_bg = canvas.copy_from_bbox(ax.bbox)

#     # dynamic artists
#     if view == "3d":
#         robot_line, = ax.plot([], [], [], "ko-", lw=2)
#         worker_scat = ax.scatter([], [], [], s=50, depthshade=False)
#     else:
#         robot_line, = ax.plot([], [], "ko-", lw=2)
#         worker_scat = ax.scatter([], [], s=50)

#     # keep last point & style info
#     prev_coords = None
#     prev_col    = None
#     prev_ls     = "-"
#     prev_lw     = 2.5

#     # Pin this worker to cores & boost priority
#     p = psutil.Process()
#     p.cpu_affinity([4,5,6,7])       # adjust core list as needed
#     try:
#         os.nice(-5)
#     except PermissionError:
#         pass

#     # ------------------ main loop ---------------------------------------
#     while True:
#         # 0) drain to the newest packet, dropping the rest
#         pkt = None
#         while True:
#             try:
#                 pkt = input_q.get_nowait()
#             except queue.Empty:
#                 break
#         # if nothing in queue, block until next arrives
#         if pkt is None:
#             pkt = input_q.get()

#         coords, joints = process_incoming_data(pkt)

#         # 1) compute current metadata
#         element = normalize_element_name(pkt.get("element"))
#         state   = (pkt.get("state") or "").strip()

#         # 2) restore background
#         canvas.restore_region(clean_bg)

#         # 3) draw the single new trajectory segment
#         if prev_coords is not None:
#             x0, y0, z0 = prev_coords
#             x1, y1, z1 = coords
#             if view == "3d":
#                 ax.plot([x0,x1], [y0,y1], [z0,z1],
#                         color=prev_col, lw=prev_lw, ls=prev_ls)
#             elif view == "top":
#                 ax.plot([x0,x1], [y0,y1], color=prev_col, lw=prev_lw, ls=prev_ls)
#             elif view == "front":
#                 ax.plot([x0,x1], [z0,z1], color=prev_col, lw=prev_lw, ls=prev_ls)
#             else:
#                 ax.plot([y0,y1], [z0,z1], color=prev_col, lw=prev_lw, ls=prev_ls)

#             canvas.draw()
#             clean_bg = canvas.copy_from_bbox(ax.bbox)

#         # 4) prepare style for next segment
#         if element.lower() == "scanning site":
#             sty = SCANNING_STYLE
#             prev_col = ELEMENT_COLORS.get(element, "black")
#             prev_ls, prev_lw = sty["linestyle"], sty["linewidth"]
#         else:
#             prev_col = ELEMENT_COLORS.get(element, "black")
#             d = STATE_STYLES_ASSEMBLY.get(state, {"linestyle":"-","linewidth":2.5})
#             prev_ls, prev_lw = d["linestyle"], d["linewidth"]

#         prev_coords = coords

#         # 5) draw robot
#         if joints is not None:
#             r = np.vstack([np.array([0,0.34301,-0.2]), np.asarray(forward_kinematics(joints))])
#             if view == "3d":
#                 robot_line.set_data(r[:,0], r[:,1])
#                 robot_line.set_3d_properties(r[:,2])
#             elif view == "top":
#                 robot_line.set_data(r[:,0], r[:,1])
#             elif view == "front":
#                 robot_line.set_data(r[:,0], r[:,2])
#             else:
#                 robot_line.set_data(r[:,1], r[:,2])

#         # 6) draw workers
#         if pkt.get("worker spotted", False):
#             wid = pkt.get("worker id")
#             col = WORKER_COLORS.get(wid, "black")
#             wx, wy, wz = pkt.get("worker coordinates", [0,0,0])
#             wz = z_level + 0.05
#             if view == "3d":
#                 worker_scat._offsets3d = ([wx],[wy],[wz]); worker_scat.set_color([col])
#             else:
#                 offs = [wx, wy] if view=="top" else ([wx, wz] if view=="front" else [wy, wz])
#                 worker_scat.set_offsets([offs]); worker_scat.set_color([col])

#         # 7) blit & render
#         ax.draw_artist(robot_line); ax.draw_artist(worker_scat)
#         canvas.blit(ax.bbox); canvas.flush_events()

#         # 8) capture PNG + send metadata tuple
#         buf = BytesIO()
#         fig.savefig(buf, format="png")
#         png_bytes = buf.getvalue()
#         buf.close()

#         output_q.put((element, state, png_bytes))

#         # 9) sync with other views
#         barrier.wait()

import os
import psutil
import queue
from io import BytesIO

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from .utils import (
    draw_ground, draw_sites, process_incoming_data, forward_kinematics,
    normalize_element_name, SCALING_FACTOR,
    x_limits, y_limits, z_limits, z_level,
    SCANNING_STYLE, ELEMENT_COLORS, STATE_STYLES_ASSEMBLY, WORKER_COLORS
)

def plot_view_worker(input_q, output_q, view, barrier):
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
        else:  # side
            ax.set_xlim(y_limits); ax.set_ylim(z_limits)
            ax.set_xlabel("Y [m]"); ax.set_ylabel("Z [m]")

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    clean_bg = canvas.copy_from_bbox(ax.bbox)

    # dynamic artists
    if view == "3d":
        robot_line,    = ax.plot([], [], [],   "ko-", lw=2)
        history_line,  = ax.plot([], [], [],   color="blue", lw=1)
        worker_scat    = ax.scatter([], [], [], s=50, depthshade=False)
    else:
        robot_line,    = ax.plot([], [],       "ko-", lw=2)
        history_line,  = ax.plot([], [],       color="blue", lw=1)
        worker_scat    = ax.scatter([], [],    s=50)

    # pre-allocate history buffer
    MAX_POINTS = 100000
    history    = np.zeros((MAX_POINTS, 3), dtype=float)
    tail       = 0

    # keep last point & style info
    prev_coords = None
    prev_col    = None
    prev_ls     = "-"
    prev_lw     = 2.5

    # Pin this worker to cores & boost priority
    p = psutil.Process()
    p.cpu_affinity([4,5,6,7])
    try:
        os.nice(-5)
    except PermissionError:
        pass

    # ------------------ main loop ---------------------------------------
    while True:
        # 0) drain to the newest packet, dropping the rest
        pkt = None
        while True:
            try:
                pkt = input_q.get_nowait()
            except queue.Empty:
                break
        if pkt is None:
            pkt = input_q.get()  # block if no packets

        coords, joints = process_incoming_data(pkt)
        element = normalize_element_name(pkt.get("element"))
        state   = (pkt.get("state") or "").strip()

        # 1) restore background
        canvas.restore_region(clean_bg)

        # 2) record & draw full-history from pre-alloc buffer
        if tail < MAX_POINTS:
            history[tail] = coords
            tail += 1

        if view == "3d":
            history_line.set_data(history[:tail,0], history[:tail,1])
            history_line.set_3d_properties(history[:tail,2])
        else:
            if view == "top":
                history_line.set_data(history[:tail,0], history[:tail,1])
            elif view == "front":
                history_line.set_data(history[:tail,0], history[:tail,2])
            else:
                history_line.set_data(history[:tail,1], history[:tail,2])

        # 3) incremental segment draw (optional styling)
        if prev_coords is not None:
            x0, y0, z0 = prev_coords
            x1, y1, z1 = coords
            if view == "3d":
                ax.plot([x0,x1], [y0,y1], [z0,z1],
                        color=prev_col, lw=prev_lw, ls=prev_ls)
            elif view == "top":
                ax.plot([x0,x1], [y0,y1],
                        color=prev_col, lw=prev_lw, ls=prev_ls)
            elif view == "front":
                ax.plot([x0,x1], [z0,z1],
                        color=prev_col, lw=prev_lw, ls=prev_ls)
            else:
                ax.plot([y0,y1], [z0,z1],
                        color=prev_col, lw=prev_lw, ls=prev_ls)

            canvas.draw()
            clean_bg = canvas.copy_from_bbox(ax.bbox)

        # 4) update styling for next segment
        if element.lower() == "scanning site":
            sty = SCANNING_STYLE
            prev_col = ELEMENT_COLORS.get(element, "black")
            prev_ls, prev_lw = sty["linestyle"], sty["linewidth"]
        else:
            prev_col = ELEMENT_COLORS.get(element, "black")
            d = STATE_STYLES_ASSEMBLY.get(state, {"linestyle":"-","linewidth":2.5})
            prev_ls, prev_lw = d["linestyle"], d["linewidth"]

        prev_coords = coords

        # 5) draw robot
        if joints is not None:
            r = np.vstack([np.array([0,0.34301,-0.2]), np.asarray(forward_kinematics(joints))])
            if view == "3d":
                robot_line.set_data(r[:,0], r[:,1]); robot_line.set_3d_properties(r[:,2])
            elif view == "top":
                robot_line.set_data(r[:,0], r[:,1])
            elif view == "front":
                robot_line.set_data(r[:,0], r[:,2])
            else:
                robot_line.set_data(r[:,1], r[:,2])

        # 6) draw workers
        if pkt.get("worker spotted", False):
            wid = pkt.get("worker id")
            col = WORKER_COLORS.get(wid, "black")
            wx, wy, wz = pkt.get("worker coordinates", [0,0,0])
            wz = z_level + 0.05
            if view == "3d":
                worker_scat._offsets3d = ([wx],[wy],[wz]); worker_scat.set_color([col])
            else:
                offs = [wx, wy] if view=="top" else ([wx, wz] if view=="front" else [wy, wz])
                worker_scat.set_offsets([offs]); worker_scat.set_color([col])

        # 7) blit & render
        ax.draw_artist(robot_line); ax.draw_artist(worker_scat)
        canvas.blit(ax.bbox); canvas.flush_events()

        # 8) bypass PNG: push raw RGBA → Qt
        canvas.draw()
        w, h = canvas.get_width_height()
        raw = canvas.buffer_rgba()  # (h, w, 4) uint8
        output_q.put((element, state, raw.tobytes(), int(w), int(h)))

        # 9) sync with other views
        barrier.wait()

