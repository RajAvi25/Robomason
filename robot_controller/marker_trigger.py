#robot_controller/marker_trigger.py
import time
from robot_controller.globals import DETECTION_STATUS, marker_stop_flag, prompt_user_lock
from detections.aruco_detection import find_aruco_markers
from configs.system_config import *
from configs.construction_config import *

def markerTrigger(frame_handler):
    """
    Monitor for any worker marker in the camera view and set stop flag on first detection.

    **Role in System:**
    Implements the vision-based intrusion detector that triggers an immediate
    trajectory interruption when a human enters the workspace, regardless of position.
    This corresponds to the safety layer described in Section 3.4 of the paper.

    **Parameters:**
    - *frame_handler*: An object providing `get_latest_frame()` and a `.stopped` flag.

    **Behavior:**
    1. Loop until `frame_handler.stopped` is True.
    2. Grab the latest frame.
    3. Detect ArUco markers (`find_aruco_markers` returns bounding boxes and IDs).
    4. If any detected ID is in `WORKER_COLORS`:
       - Update `DETECTION_STATUS`.
       - On first detection since last clear, acquire `prompt_user_lock` and set `marker_stop_flag`.
    5. If no worker marker, reset detection state and update `DETECTION_STATUS`.
    6. Sleep 50 ms between checks for ~20 Hz polling.

    **Global Effects:**
    - `DETECTION_STATUS` is updated with either "Marker seen in frame" or "No detection".
    - `marker_stop_flag` is set once per intrusion event, driving the RobotController’s stop.
    """
    global DETECTION_STATUS
    marker_detected_recently = False
    while not frame_handler.stopped:
        frame = frame_handler.get_latest_frame()
        if frame is not None:
            _, ids = find_aruco_markers(frame)
            if ids is not None and any(worker_id in WORKER_COLORS for worker_id in ids.flatten()):
                DETECTION_STATUS = "Marker seen in frame"
                if not marker_detected_recently:
                    marker_detected_recently = True
                    with prompt_user_lock:
                        if not marker_stop_flag.is_set():
                            marker_stop_flag.set()
            else:
                DETECTION_STATUS = "No detection"
                marker_detected_recently = False
        time.sleep(0.05)

def markerTrigger_30(frame_handler):
    """
    Monitor for worker markers in the bottom 30% of the image (critical zone) and set stop flag.

    **Role in System:**
    Implements the image‐band intrusion detector described in Section 3.4:
    only markers entering the designated lower 30% of the frame (where robot trajectories
    cross human working zones) trigger a stop and detour.

    **Parameters:**
    - *frame_handler*: An object providing `get_latest_frame()` and a `.stopped` flag.

    **Behavior:**
    1. Loop until `frame_handler.stopped` is True.
    2. Grab the latest frame and determine its height.
    3. Detect ArUco markers (`find_aruco_markers`).
    4. For each worker marker ID in `WORKER_COLORS`:
       - Compute the maximum Y-coordinate of its bounding-box corners.
       - If that Y ≥ 70% of image height (i.e., in bottom 30%), treat as intrusion.
       - On first valid detection, acquire `prompt_user_lock` and set `marker_stop_flag`.
    5. If no valid marker in the critical zone, reset detection state and update `DETECTION_STATUS`.
    6. Sleep 50 ms between checks.

    **Global Effects:**
    - `DETECTION_STATUS` is updated with either "Marker seen in frame" or "No detection".
    - `marker_stop_flag` is set once per valid intrusion, driving the RobotController’s safety stop.
    """
    global DETECTION_STATUS
    marker_detected_recently = False
    while not frame_handler.stopped:
        frame = frame_handler.get_latest_frame()
        if frame is not None:
            bboxs, ids = find_aruco_markers(frame)
            height, _, _ = frame.shape
            valid_marker_found = False
            if ids is not None:
                for bbox, marker_id in zip(bboxs, ids.flatten()):
                    if marker_id in WORKER_COLORS:
                        marker_y_positions = [corner[1] for corner in bbox[0]]
                        max_marker_y = max(marker_y_positions)
                        if max_marker_y >= 0.7 * height:
                            valid_marker_found = True
                            DETECTION_STATUS = "Marker seen in frame"
                            if not marker_detected_recently:
                                marker_detected_recently = True
                                with prompt_user_lock:
                                    if not marker_stop_flag.is_set():
                                        marker_stop_flag.set()
                            break
            if not valid_marker_found:
                DETECTION_STATUS = "No detection"
                marker_detected_recently = False
        time.sleep(0.05)
