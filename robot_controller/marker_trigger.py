#robot_controller/marker_trigger.py
import time
from robot_controller.globals import DETECTION_STATUS, marker_stop_flag, prompt_user_lock
from detections.aruco_detection import find_aruco_markers
from system_config import *
from construction_config import *

def markerTrigger(frame_handler):
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
