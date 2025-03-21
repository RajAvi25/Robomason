#robot_controller/globals.py
import threading

DETECTION_STATUS = "No detection"
marker_stop_flag = threading.Event()
prompt_user_lock = threading.Lock()

OBSTACLE_MANEUVER = False