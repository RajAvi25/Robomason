import zmq
import cv2
import base64
import time
import os
from datetime import datetime
import msgpack
import json
import glob
from pathlib import Path
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
    """
    Publish frames and robot state to the tracking worker (over ZeroMQ).

    This thread function opens a PUB socket on tcp://127.0.0.1:5550 and continuously:
      - Grabs the latest camera frame,
      - Encodes it as a JPEG and base64 string,
      - Retrieves current robot end-effector coordinates, orientation matrix, joint angles, 
        and obstacle maneuver flag via `mdl.get_EE_coords()`, etc.
      - Packages all these into a JSON message and sends it out.

    The JSON message has structure:
    ```
    {
      'image': <base64_jpg_string>,
      'coordinates': {'x': ..., 'y': ..., 'z': ...},
      'orientation': [[..., ..., ...], [...], [...]],   # 3x3 matrix
      'element': <current_element_label>,
      'state': <current_state_label>,
      'joints': [...six joint values...],
      'obstacle_maneuver': <bool>
    }
    ```
    where `element` and `state` come from `construction_status` (updated by construction routines to reflect ongoing action).

    **Purpose:**
    This is the publisher side of the vision pipeline: it streams data to the external *tracking node*, 
    which is responsible for marker detection and intrusion monitoring. By including `element` and `state`, 
    the tracking node knows what the robot is doing (e.g., "Wall_1 pick") when analyzing frames, and by including robot pose, 
    it can do coordinate transformations if needed.

    **Role in System:**
    This enables the decoupling of heavy image processing from real-time robot control, using ZeroMQ as the message bus:contentReference[oaicite:20]{index=20}. 
    It supports *real-time worker-aware replanning* by ensuring that the tracking node has up-to-date images and robot pose; 
    if a worker enters (detected in the image), the tracking node will publish a message that can trigger the robot to detour mid-action:contentReference[oaicite:21]{index=21}.
    """
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
    """
    Subscribe to processed tracking messages and forward them to the UI plotting.

    Opens a SUB socket to tcp://127.0.0.1:5552 (which is where the tracking worker publishes results).
    For each incoming message (JSON string), it:
      - Parses the JSON into a dict (`response`).
      - Pushes the data into `tracking_packets` list (with thread lock protection).
      - Also calls `send_data_plotting(response)` which forwards the data (packed via msgpack) to another socket for real-time plotting.

    The data in `response` includes:
    - "coordinates": [x, y, z] of whatever object was tracked (e.g., the ArUco marker or worker),
    - "element": current element label during that frame,
    - "state": current state label,
    - "joints": robot joint angles,
    - "orientation": robot EE orientation,
    - "worker spotted": boolean if a worker was seen,
    - "worker coordinates": (if worker spotted, the image or world coords of the worker),
    - "worker id": ID of the detected worker marker,
    - "obstacle_maneuver": bool flag mirrored from robot (whether robot was in avoidance mode),
    - "timestamp_send": time when tracking message was sent.

    **Role:**
    This is the receiving end of the tracking pipeline for the UI. All messages appended to `tracking_packets` serve as a timeline of events 
    that can be analyzed (e.g., by `construct/analysis.py` to find hazard events). Real-time plotting uses this to visualize robot motion and hazards.
    """
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
    """
    Retrieve all accumulated tracking packets.

    **Parameters:**
    - *clear_after_retrieval (bool)* – If True, the internal list is cleared after copying its contents.

    **Returns:**
    - *list of dict* – A shallow copy of all tracking packets received so far (each dict is one message from tracking).

    Use this to get the log of events (e.g., for analysis at the end of a run). Optionally clear the log if it’s no longer needed.
    """
    global tracking_packets
    with tracking_packets_lock:
        data_copy = tracking_packets.copy()
        if clear_after_retrieval:
            tracking_packets.clear()
    return data_copy

def refresh_tracking_packets():
    """Thread-safe clear of the tracking_packets list (e.g., to start fresh for a new run)."""
    global tracking_packets
    with tracking_packets_lock:
        tracking_packets.clear()

def get_current_packet():
    """
    Get the most recent tracking packet (if any).

    Returns the last dictionary in tracking_packets list without removing it, or None if no data yet.
    Useful for quickly checking the latest status (e.g., whether a worker was just spotted).
    """
    with tracking_packets_lock:
        if tracking_packets:
            return tracking_packets[-1]
        else:
            return None
        
def save_failed_run(run_phase,block_list, pl_pos,img):
    """
    Save data about a failed run (assembly/disassembly) for offline analysis.

    When a run is aborted or verification fails, this can be called to record:
    - All tracking packets up to the failure,
    - The sequence of elements attempted (`block_list`),
    - The positions at which they were placed (`pl_pos`),
    - A snapshot image from the moment of failure (`img`).

    It creates a new folder under `_workingdata/_siteinfo/_failedruns` with an incremented index, and saves:
    - A JSON file containing the packets, block list, and placed positions.
    - An image file (PNG) of the failure scene.

    **Parameters:**
    - *run_phase (str)* – Phase of operation ("assembly", "disassembly", or "reassembly").
    - *block_list (list)* – The sequence of elements that were being processed.
    - *pl_pos (list)* – The list of placement positions (end-effector poses) for those elements.
    - *img* – An image (NumPy array) to save representing the scene at failure.

    **Returns:**
    - *Path* – Filesystem path to the directory where data was saved.

    **Note:**
    This function ensures the directory exists, finds the next numeric index, and writes the data. It's used to accumulate evidence for what went wrong, enabling *post-run analysis* of safety or accuracy failures.
    """
    # 1) Retrieve a copy of the packets
    packets = get_tracking_packets(clear_after_retrieval=False)
    
    root = Path("_workingdata/_siteinfo/_failedruns").resolve()
    root.mkdir(parents=True, exist_ok=True)

    # find next index
    existing = [int(p.name) for p in root.iterdir()
                if p.is_dir() and p.name.isdigit()]
    next_idx = max(existing)+1 if existing else 0

    run_dir = root / f"{next_idx:02d}"
    run_dir.mkdir(exist_ok=False)
    
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
    filename = f"{next_idx:02d}--{run_phase}--{ts_str}.json"
    (run_dir / filename).write_text(json.dumps(payload, indent=2))
    # out_path = os.path.join(run_dir, filename)
    # with open(out_path, "w") as f:
    #     json.dump(payload, f, indent=2)

    image_filename = f"{next_idx:02d}--{run_phase}--{ts_str}.png"
    image_path = os.path.join(run_dir, image_filename)
    cv2.imwrite(image_path, img)
    
    print(f"Saved failed run to: {run_dir}")
    return run_dir

def get_saved_paths(base_dir, selection):
    base = Path(base_dir).resolve()
    # strip off a trailing run-folder if passed accidentally
    if base.name.isdigit() and (base.parent.name == "_failedruns"):
        base = base.parent
    
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
            raise ValueError(f"Invalid index '{selection}'.")
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
    # plotting_socket.bind("tcp://127.0.0.1:5555")
    plotting_socket.bind("ipc:///tmp/plotter.ipc")
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

# ----------------- child process (runs Tk) -----------------
def _board_process(q_in: mp.Queue, q_out: mp.Queue, bg: str = "white"):
    root = tk.Tk()
    root.title("Robomason")
    root.configure(bg=bg)
    root.geometry("400x250")
    root.resizable(False, False)

    # Permanent header
    header = tk.Label(
        root,
        text="Question board",
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








