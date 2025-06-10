import numpy as np
import zmq
import msgpack
import multiprocessing as mp
from io import BytesIO
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from .utils import *

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

# def plot_view_worker(input_q, output_q, view,barrier):
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
#         else:                                    # side
#             ax.set_xlim(y_limits); ax.set_ylim(z_limits)
#             ax.set_xlabel("Y [m]"); ax.set_ylabel("Z [m]")

#     canvas   = FigureCanvasAgg(fig)
#     canvas.draw()
#     clean_bg = canvas.copy_from_bbox(ax.bbox)      # robot-free background

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

#     # ------------------ main loop ---------------------------------------
#     while True:
#         pkt = input_q.get()                        # wait for data
#         coords, joints = process_incoming_data(pkt)
#         element = normalize_element_name(pkt.get("element"))
#         state   = (pkt.get("state") or "").strip()

#         # -------- 1) restore background ---------------------------------
#         canvas.restore_region(clean_bg)

#         # -------- 2) draw ONE new segment -------------------------------
#         if prev_coords is not None:
#             x0, y0, z0 = prev_coords
#             x1, y1, z1 = coords
#             if view == "3d":
#                 ax.plot([x0, x1], [y0, y1], [z0, z1],
#                         color=prev_col, lw=prev_lw, ls=prev_ls)
#             elif view == "top":
#                 ax.plot([x0, x1], [y0, y1],
#                         color=prev_col, lw=prev_lw, ls=prev_ls)
#             elif view == "front":
#                 ax.plot([x0, x1], [z0, z1],
#                         color=prev_col, lw=prev_lw, ls=prev_ls)
#             else:  # side
#                 ax.plot([y0, y1], [z0, z1],
#                         color=prev_col, lw=prev_lw, ls=prev_ls)

#             canvas.draw()                          # rasterise that line
#             clean_bg = canvas.copy_from_bbox(ax.bbox)   # new background

#         # decide style for NEXT segment
#         if element.lower() == "scanning site":
#             sty = SCANNING_STYLE
#             prev_col = ELEMENT_COLORS.get(element, "black")
#             prev_ls, prev_lw = sty["linestyle"], sty["linewidth"]
#         else:
#             prev_col = ELEMENT_COLORS.get(element, "black")
#             d = STATE_STYLES_ASSEMBLY.get(state, {"linestyle": "-", "linewidth": 2.5})
#             prev_ls, prev_lw = d["linestyle"], d["linewidth"]

#         prev_coords = coords

#         # -------- 3) robot ----------------------------------------------
#         if joints is not None:
#             r = np.asarray(forward_kinematics(joints))
#             r = np.vstack([np.array([0, 0.34301, -0.2]), r])   # add ground anchor
#             if view == "3d":
#                 robot_line.set_data(r[:, 0], r[:, 1]); robot_line.set_3d_properties(r[:, 2])
#             elif view == "top":
#                 robot_line.set_data(r[:, 0], r[:, 1])
#             elif view == "front":
#                 robot_line.set_data(r[:, 0], r[:, 2])
#             else:
#                 robot_line.set_data(r[:, 1], r[:, 2])

#         # -------- 4) workers (optional) ---------------------------------
#         if pkt.get("worker spotted", False):
#             wid = pkt.get("worker id")
#             col = WORKER_COLORS.get(wid, "black")
#             wx, wy, wz = pkt.get("worker coordinates", [0, 0, 0])
#             wz = z_level + 0.05
#             if view == "3d":
#                 worker_scat._offsets3d = ([wx], [wy], [wz]); worker_scat.set_color([col])
#             elif view == "top":
#                 worker_scat.set_offsets([[wx, wy]]); worker_scat.set_color([col])
#             elif view == "front":
#                 worker_scat.set_offsets([[wx, wz]]); worker_scat.set_color([col])
#             else:
#                 worker_scat.set_offsets([[wy, wz]]); worker_scat.set_color([col])

#         # -------- 5) blit & ship PNG ------------------------------------
#         ax.draw_artist(robot_line); ax.draw_artist(worker_scat)
#         canvas.blit(ax.bbox); canvas.flush_events()

#         buf = BytesIO(); fig.savefig(buf, format="png"); buf.seek(0)
#         output_q.put(buf.read()); buf.close()

#         barrier.wait()

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
    clean_bg = canvas.copy_from_bbox(ax.bbox)  # robot‐free background

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
        # 0) get next packet
        pkt = input_q.get()  
        coords, joints = process_incoming_data(pkt)

        # 1) compute current metadata
        element = normalize_element_name(pkt.get("element"))
        state   = (pkt.get("state") or "").strip()

        # 2) restore background
        canvas.restore_region(clean_bg)

        # 3) draw the single new trajectory segment
        if prev_coords is not None:
            x0, y0, z0 = prev_coords
            x1, y1, z1 = coords
            if view == "3d":
                ax.plot([x0,x1],[y0,y1],[z0,z1],
                        color=prev_col, lw=prev_lw, ls=prev_ls)
            elif view == "top":
                ax.plot([x0,x1],[y0,y1],
                        color=prev_col, lw=prev_lw, ls=prev_ls)
            elif view == "front":
                ax.plot([x0,x1],[z0,z1],
                        color=prev_col, lw=prev_lw, ls=prev_ls)
            else:
                ax.plot([y0,y1],[z0,z1],
                        color=prev_col, lw=prev_lw, ls=prev_ls)

            canvas.draw()
            clean_bg = canvas.copy_from_bbox(ax.bbox)

        # 4) prepare style for next segment
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

        # 8) capture PNG + send metadata tuple
        buf = BytesIO()
        fig.savefig(buf, format="png")
        png_bytes = buf.getvalue()
        buf.close()

        output_q.put((element, state, png_bytes))

        # 9) sync with other views
        barrier.wait()
