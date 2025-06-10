import zmq
import cv2
import base64
import time
import os
from datetime import datetime
import msgpack
import json
import threading
from threading import Lock
from camera.frame_handler import FrameHandler
from ui.mobility import set_robot_control_socket
from . import MarkerDetectionLocalization as mdl
from construct import construction_status

import multiprocessing as mp, uuid, time, tkinter as tk

current_element = None
current_state = None
tracking_packets = []
tracking_packets_lock = Lock()

def publisher_trackingworker(framehandler):
    context = zmq.Context()
    publisher_socket = context.socket(zmq.PUB)
    publisher_socket.bind("tcp://127.0.0.1:5550")
    while True:
        frame = framehandler.get_latest_frame()
        if frame is not None:
            _, img_encoded = cv2.imencode('.jpg', frame)
            img_base64 = base64.b64encode(img_encoded).decode('utf-8')
            # For UI, you might not have actual robot data – we use stubs:
            # robot_coords = [0, 0, 0]
            # orientation = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            # joints = []
            robot_coords =mdl.get_EE_coords()
            orientation = mdl.get_orientation()
            joints = mdl.get_joints()
            obstacle_maneuver = mdl.get_obsacle_maneuver_status()

            # print(f'coordinates: {robot_coords}')
            # print(f'orientation: {orientation}')
            # print(f'joints: {joints}')

            with construction_status.state_lock:
                current_element = construction_status.state["current_element"] 
                current_state = construction_status.state["current_state"] 

            coordinates = {'x': robot_coords[0], 'y': robot_coords[1], 'z': robot_coords[2]}
            message = json.dumps({
                'image': img_base64,
                'coordinates': coordinates,
                'orientation': orientation,
                'element': current_element,
                'state': current_state,
                'joints': joints,
                "obstacle_maneuver":obstacle_maneuver
            })
            publisher_socket.send_string(message)

def send_data_plotting(data):
    data['timestamp_send'] = time.time()
    packed_data = msgpack.packb(data)
    plotting_socket.send(packed_data)

def receive_data_UI():
    global tracking_packets
    context = zmq.Context()
    subscriber_socket = context.socket(zmq.SUB)
    subscriber_socket.connect("tcp://127.0.0.1:5552")
    subscriber_socket.setsockopt_string(zmq.SUBSCRIBE, "")
    print("Listening for tracking messages...")
    while True:
        try:
            message = subscriber_socket.recv_string()
            response = json.loads(message)
            # print(response["obstacle_maneuver"])
            send_data_plotting(response)
            tracking_packets.append(response)
        except Exception as e:
            print(f"Error in tracking data receiving thread: {e}")

def get_tracking_packets(clear_after_retrieval=False):
    global tracking_packets
    with tracking_packets_lock:
        data_copy = tracking_packets.copy()
        if clear_after_retrieval:
            tracking_packets.clear()
    return data_copy

def refresh_tracking_packets():
    """Clears tracking packets collected so far without interrupting new updates."""
    global tracking_packets
    with tracking_packets_lock:
        tracking_packets.clear()

def get_current_packet():
    with tracking_packets_lock:
        if tracking_packets:
            return tracking_packets[-1]
        else:
            return None
        
def save_failed_run(run_phase,block_list, pl_pos):
    """
    Grabs all current tracking packets via get_tracking_packets(),
    then saves them (plus metadata) into a newly created subfolder under
    '_workingdata/_siteinfo/_failedruns'. Subfolders are named '00', '01', '02', etc.
    
    Args:
        run_phase (str): One of 'assembly', 'disassembly', or 'reassembly'.
    """
    # 1) Retrieve a copy of the packets
    packets = get_tracking_packets(clear_after_retrieval=False)
    
    # 2) Ensure base directory exists
    base_dir = os.path.join("_workingdata", "_siteinfo", "_failedruns")
    os.makedirs(base_dir, exist_ok=True)
    
    # 3) Find existing numeric subfolders and compute next index
    existing_indices = []
    for name in os.listdir(base_dir):
        full_path = os.path.join(base_dir, name)
        if os.path.isdir(full_path) and name.isdigit():
            try:
                existing_indices.append(int(name))
            except ValueError:
                pass
    
    if existing_indices:
        next_index = max(existing_indices) + 1
    else:
        next_index = 0
    
    folder_name = f"{next_index:02d}"  # zero-pad to at least two digits
    run_dir = os.path.join(base_dir, folder_name)
    os.makedirs(run_dir, exist_ok=False)
    
    # 4) Build the JSON payload
    payload = {
        "timestamp": datetime.now().isoformat(),
        "phase": run_phase,
        "tracking_packets": packets,
        "block_list": block_list,
        "pl_pos": pl_pos,
    }

    ts_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    
    # 5) Write out a single JSON file in the new run_dir
    filename = f"{folder_name}--{run_phase}--{ts_str}.json"
    out_path = os.path.join(run_dir, filename)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    # image_filename = f"{folder_name}--{run_phase}--{ts_str}.png"
    # image_path = os.path.join(run_dir, image_filename)
    # cv2.imwrite(image_path, img)
    
    print(f"Saved failed run to: {run_dir}")
    return run_dir

def get_saved_paths(base_dir, selection):
    """
    Args:
        base_dir (str): e.g. "_workingdata/_siteinfo/_failedruns" or your test_base
        selection (str): "NN" or "NN-MM"
    Returns:
        List[str]: one or more JSON paths, in ascending index order
    Raises:
        ValueError on bad input, missing folder, or no JSON files
    """
    # parse selection
    if "-" in selection:
        start_str, end_str = selection.split("-", 1)
        if not (start_str.isdigit() and end_str.isdigit()):
            raise ValueError(f"Invalid range '{selection}'. Use '00-03'.")
        start, end = int(start_str), int(end_str)
        if start > end:
            start, end = end, start
    else:
        if not selection.isdigit():
            raise ValueError(f"Invalid index '{selection}'. Use '02'.")
        start = end = int(selection)

    paths = []
    for idx in range(start, end + 1):
        folder = os.path.join(base_dir, f"{idx:02d}")
        if not os.path.isdir(folder):
            raise ValueError(f"Folder not found: {folder}")
        # pick the last JSON in that folder
        candidates = sorted(glob.glob(os.path.join(folder, "*.json")))
        if not candidates:
            raise ValueError(f"No .json files in {folder}")
        paths.append(candidates[-1])
    return paths


def connectRobotserver():
    context = zmq.Context()
    robot_control_socket = context.socket(zmq.REQ)
    robot_control_socket.connect("tcp://localhost:5556")
    time.sleep(0.25)
    set_robot_control_socket(robot_control_socket)

def connectPlottingserver():
    global plotting_socket
    context = zmq.Context()
    plotting_socket = context.socket(zmq.PUB)
    plotting_socket.bind("tcp://127.0.0.1:5555")
    time.sleep(0.25)

def connectTrackerserver():
    global start_tracking_context, start_tracking_socket
    start_tracking_context = zmq.Context()
    start_tracking_socket = start_tracking_context.socket(zmq.PUB)
    start_tracking_socket.bind("tcp://127.0.0.1:5551")
    time.sleep(0.25)

def initCameraHandler(ws_url="ws://localhost:9090", camera_index=4, frame_rate=15):
    global cameraHandler
    cameraHandler = FrameHandler(ws_url=ws_url, camera_index=camera_index, frame_rate=frame_rate, is_sender=False)
    time.sleep(0.25)
    threading.Thread(target=cameraHandler.start_streaming, daemon=True).start()
    time.sleep(0.25)

def initTracker(delay=0.75):
    time.sleep(delay)
    start_tracking_socket.send_string("start_tracking")
    time.sleep(delay)
    threading.Thread(target=publisher_trackingworker, args=(cameraHandler,), daemon=True).start()
    time.sleep(delay-0.25)

def initPlotting():
    threading.Thread(target=receive_data_UI, daemon=True).start()

# # ----------------- child process (runs Tk) -----------------
# def _board_process(q_in: mp.Queue, q_out: mp.Queue, bg: str = "white"):
#     root = tk.Tk()
#     root.title("✦ Question Board ✦")
#     root.configure(bg=bg)
#     frm = tk.Frame(root, bg=bg)
#     frm.pack(expand=True, fill="both", padx=40, pady=40)

#     # holder objects so nested funcs can mutate them
#     state = {"frame": None, "timer_id": None}

#     def _display_question(payload):
#         """Create widgets for a new question."""
#         qid, text, options, timeout, default_idx = payload
#         # clear previous contents
#         for child in frm.winfo_children():
#             child.destroy()

#         lbl_q = tk.Label(frm, text=text, font=("Helvetica", 14), bg=bg)
#         lbl_q.pack(pady=(0, 10))

#         btn_frame = tk.Frame(frm, bg=bg)
#         btn_frame.pack(pady=(0, 10))

#         # callback factory
#         def make_cmd(opt):
#             return lambda: _finish(qid, opt)

#         for opt in options:
#             tk.Button(btn_frame, text=opt, width=10,
#                       command=make_cmd(opt)).pack(side="left", padx=6)

#         lbl_timer = tk.Label(frm, bg=bg, fg="grey")
#         lbl_timer.pack()

#         # countdown using root.after
#         def _countdown(t_left):
#             if t_left < 0:
#                 _finish(qid, options[default_idx])
#                 return
#             lbl_timer.config(text=f"Auto in {t_left}s")
#             state["timer_id"] = root.after(1000, _countdown, t_left - 1)

#         _countdown(timeout)

#     def _finish(qid, choice):
#         """Send answer back and wipe widgets."""
#         if state["timer_id"]:
#             root.after_cancel(state["timer_id"])
#             state["timer_id"] = None
#         q_out.put((qid, choice))
#         for child in frm.winfo_children():
#             child.destroy()  # blank board

#     # poll the input queue every 100 ms
#     def _poll():
#         try:
#             while True:      # drain
#                 payload = q_in.get_nowait()
#                 if payload == "__quit__":
#                     root.destroy(); return
#                 _display_question(payload)
#         except Exception:
#             pass
#         root.after(100, _poll)

#     root.after(100, _poll)
#     root.mainloop()

# # ----------------- public helper class -----------------
# class QuestionBoard:
#     def __init__(self, bg: str = "white"):
#         self.q_in  : mp.Queue = mp.Queue()
#         self.q_out : mp.Queue = mp.Queue()
#         self.proc           = mp.Process(target=_board_process,
#                                          args=(self.q_in, self.q_out, bg),
#                                          daemon=True)
#         self.proc.start()

#     # ----------------------------------------------------
#     def ask(self, question: str, options=("Yes", "No"),
#             timeout: int = 5, default_idx: int = 0) -> str:
#         """
#         Show a single question, block until answer or timeout.
#         Returns the chosen option.
#         """
#         qid = str(uuid.uuid4())
#         self.q_in.put((qid, question, options, timeout, default_idx))
#         # wait for answer
#         while True:
#             ans_qid, choice = self.q_out.get()
#             if ans_qid == qid:
#                 return choice

#     # ----------------------------------------------------
#     def close(self):
#         """Terminate the board process cleanly."""
#         self.q_in.put("__quit__")
#         self.proc.join(timeout=1)


# ----------------- child process (runs Tk) -----------------
def _board_process(q_in: mp.Queue, q_out: mp.Queue, bg: str = "white"):
    root = tk.Tk()
    root.title("✦ Question Board ✦")
    root.configure(bg=bg)
    root.geometry("400x250")
    root.resizable(False, False)

    # Permanent header
    header = tk.Label(
        root,
        text="📝 Question Board",
        font=("Helvetica", 16, "bold"),
        bg=bg,
        fg="#333"
    )
    header.pack(pady=(10, 5))

    # Dynamic content frame
    frm = tk.Frame(root, bg=bg)
    frm.pack(expand=True, fill="both", padx=20, pady=10)

    # holder for timer callback id
    state = {"timer_id": None}

    def _clear_content():
        for w in frm.winfo_children():
            w.destroy()

    def _finish(qid, choice):
        # cancel any pending countdown
        if state["timer_id"] is not None:
            try:
                root.after_cancel(state["timer_id"])
            except Exception:
                pass
            state["timer_id"] = None

        q_out.put((qid, choice))
        _clear_content()

    def _display_question(payload):
        qid, text, options, timeout, default_idx = payload
        _clear_content()

        # Question text
        lbl_q = tk.Label(
            frm,
            text=text,
            font=("Helvetica", 13),
            wraplength=360,
            justify="center",
            bg=bg
        )
        lbl_q.pack(pady=(0, 15))

        # Option buttons
        btn_frame = tk.Frame(frm, bg=bg)
        btn_frame.pack(pady=(0, 10))
        def make_cmd(opt):
            return lambda: _finish(qid, opt)

        for opt in options:
            tk.Button(
                btn_frame,
                text=opt,
                width=8,
                font=("Helvetica", 11),
                relief="raised",
                bg="#eee",
                command=make_cmd(opt)
            ).pack(side="left", padx=8)

        # Countdown label
        lbl_timer = tk.Label(frm, bg=bg, fg="#666")
        lbl_timer.pack()

        def _countdown(t_left):
            # stop if label gone
            if not lbl_timer.winfo_exists():
                return
            if t_left < 0:
                _finish(qid, options[default_idx])
                return
            try:
                lbl_timer.config(text=f"Auto-select in {t_left}s")
            except tk.TclError:
                return
            state["timer_id"] = root.after(1000, _countdown, t_left - 1)

        # start countdown
        _countdown(timeout)

    def _poll():
        try:
            while True:
                payload = q_in.get_nowait()
                if payload == "__quit__":
                    root.destroy()
                    return
                _display_question(payload)
        except mp.queues.Empty:
            pass
        root.after(100, _poll)

    root.after(100, _poll)
    root.mainloop()

# ----------------- public helper class -----------------
class QuestionBoard:
    def __init__(self, bg: str = "white"):
        self.q_in  : mp.Queue = mp.Queue()
        self.q_out : mp.Queue = mp.Queue()
        self.proc = mp.Process(
            target=_board_process,
            args=(self.q_in, self.q_out, bg),
            daemon=True
        )
        self.proc.start()

    def ask(self, question: str, options=("Yes", "No"),
            timeout: int = 5, default_idx: int = 0) -> str:
        """
        Show a single question, block until answer or timeout.
        Returns the chosen option.
        """
        qid = str(uuid.uuid4())
        self.q_in.put((qid, question, options, timeout, default_idx))
        while True:
            ans_qid, choice = self.q_out.get()
            if ans_qid == qid:
                return choice

    def close(self):
        """Terminate the board process cleanly."""
        self.q_in.put("__quit__")
        self.proc.join(timeout=1)








