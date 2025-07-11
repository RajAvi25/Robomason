# MarkerDetectionLocalization.py
"""
Marker Detection and Localization Library (Integration Module).

This module interfaces between the robot controller, the vision system, and the ZeroMQ messaging 
infrastructure for marker detection and robot state updates. It runs background threads to publish frames 
and listen for tracking data, maintaining global state for use by other parts of the application.

Key functionalities:
- **Data Publishing:** Captures camera frames and robot pose data, and publishes them over ZeroMQ to a tracking 
  process (the "tracking worker"). This enables a decoupled vision module to detect markers and potential intrusions.
- **Tracking Data Receiving:** Subscribes to the tracking worker's outputs (marker positions, worker detection, etc.) 
  to update internal state.
- **Robot State Listener:** Listens for real-time robot joint and orientation data broadcast by the robot controller 
  (over ZeroMQ), updating global variables for positions and orientation (accessible via `get_joints()`, etc.).
- **Worker Detection Flag:** Maintains a `worker_spotted_flag` that indicates if a human worker (marker) is detected 
  in the scene, which can be used to trigger safety stops in the robot motion (see `marker_trigger.py` usage).

By combining these, the module supports the *real-time worker-aware replanning* capability: if the tracking system 
flags a worker intruding (e.g., marker detected in a specific image band), the RobotController can immediately 
halt or adjust its path:contentReference[oaicite:18]{index=18}. Similarly, robot joint/orientation streaming ensures that all components have 
a consistent view of the robot's current state.
"""

import threading
import zmq
import json
import base64
import queue
import cv2
import numpy as np

from camera.frame_handler import FrameHandler


# Global variables for managing detection state and data.
worker_spotted_flag = False
received_coordinates = None
received_id = None
coordinates_lock = threading.Lock()
message_queue = queue.Queue()  # Queue to hold new detection data dictionaries.
all_received_data = []         # List to hold a history of all detections.

##############################################
# NEW: Global state for robot data received via ZMQ
##############################################
latest_robot_data = {}
robot_data_lock = threading.Lock()

orientation = np.eye(3)
translation = np.zeros(3)

def robot_data_listener():
    """
    Thread function: subscribe to robot state updates via ZeroMQ.

    Connects to a local IPC endpoint (or TCP, as configured) where the robot controller publishes 
    joint angles and orientation (as JSON). On receiving each data packet, it updates the `latest_robot_data` dict.

    Expected keys in data_packet:
    - "joints": list of 6 joint angles (rad)
    - "orientation": 3x3 rotation matrix of end-effector (as nested lists)
    - "EE cords": [x, y, z] end-effector coordinates in robot base frame
    - "obstacle_maneuver": bool flag if robot is in an obstacle avoidance maneuver.

    This function runs in a loop and should be started in a separate daemon thread via `start_robot_data_listener()`.

    **Role in System:**
    It is part of the *ZeroMQ control stack*, receiving real-time robot telemetry:contentReference[oaicite:19]{index=19}. By updating `latest_robot_data`, 
    other functions like `get_joints()`, `get_orientation()`, etc., can retrieve current values without directly querying the robot every time.
    """
    context = zmq.Context.instance()
    sub_socket = context.socket(zmq.SUB)
    sub_socket.connect("ipc:///tmp/robot.ipc")
    sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
    print("Robot data listener started on ipc:///tmp/robot.ipc...")
    while True:
        try:
            data_packet = sub_socket.recv_json()
            with robot_data_lock:
                latest_robot_data.update(data_packet)
        except Exception as e:
            print(f"Error in robot data listener: {e}")

def get_joints():
    """
    Returns the latest robot joint angles.
    """
    with robot_data_lock:
        return latest_robot_data.get("joints")

def get_orientation():
    """
    Returns the latest robot end effector orientation vector.
    """
    with robot_data_lock:
        return latest_robot_data.get("orientation")
    
def get_EE_coords():
    """
    Returns the latest robot end-effector coordinates.
    """
    with robot_data_lock:
        return latest_robot_data.get("EE cords")
    
def get_obsacle_maneuver_status():
    """
    Returns the status variable differentiating normal movements vs obstacle avoidance movements.
    """
    with robot_data_lock:
        return latest_robot_data.get("obstacle_maneuver")
    
def set_transform_params(new_orientation, new_translation):
    """
    Set the global transform parameters for coordinate transformations.

    This is used to update the `orientation` matrix and `translation` vector that `transform_coordinates` applies.
    Typically called when calibrating or switching coordinate frames (e.g., aligning camera frame to robot base frame).

    **Parameters:**
    - *new_orientation (np.ndarray)* – 3x3 rotation matrix for the transform.
    - *new_translation (np.ndarray)* – 3x1 translation vector for the transform.
    """
    global orientation, translation
    orientation = new_orientation
    translation = new_translation

def transform_coordinates(local_coords):
    """
    Transform coordinates from a local frame to the global frame using the set transform parameters.

    Given a point in the camera's coordinate system (or other local frame), this applies the stored rotation and translation 
    to convert it to robot base coordinates.

    **Parameters:**
    - *local_coords (array-like)* – Coordinates [x, y, z] in the local frame.

    **Returns:**
    - *np.ndarray* – Transformed coordinates [x, y, z] in the global frame.

    **Note:**
    - If the current orientation is the identity matrix (no transform set), it returns the input as is.
    - Otherwise computes: `global_coords = orientation * local_coords + translation`.
    """
    global orientation, translation  # Access the global variables
    local_coords = np.array(local_coords)
    if np.array_equal(orientation, np.eye(3)):
        return local_coords
    else:
        # Perform the transformation: R * P + T
        global_coords = np.dot(orientation, local_coords) + translation
        return global_coords

def start_robot_data_listener():
    """
    Starts the robot data listener in a separate daemon thread.
    """
    thread = threading.Thread(target=robot_data_listener, daemon=True)
    thread.start()
    return thread

##############################################
# Existing functions for marker detection/localization
##############################################
def worker_spotted(isSpotted):
    """
    Set the flag indicating whether a worker (marker) has been spotted in the scene.

    This flag might be used by other modules (like `marker_trigger`) to initiate a stop. 
    In practice, the tracking pipeline will send a "worker spotted" boolean in its messages, 
    and the UI/communication layer could call this to update the flag.

    **Parameters:**
    - *isSpotted (bool)* – True if a worker/human marker is detected, False otherwise.
    """
    global worker_spotted_flag
    worker_spotted_flag = isSpotted


