# MarkerDetectionLocalization.py
"""
Marker Detection and Localization Library

This module provides functionality for:
  - Publishing image frames along with robot data over ZeroMQ (data_publisher).
  - Receiving marker detection (coordinate) messages over ZeroMQ (receive_coordinates_from_tracking_worker).
  - Continuously updating robot joint and orientation data from a dedicated ZMQ publisher.

Global state is maintained via global variables and message queues.
The module exposes helper functions to start each thread and to retrieve detection and robot data.
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
    Listens on tcp://127.0.0.1:5560 for robot data packets published by the main program.
    Each packet is expected to be a JSON dict with keys "joints" and "orientation".
    """
    context = zmq.Context()
    sub_socket = context.socket(zmq.SUB)
    sub_socket.connect("tcp://127.0.0.1:5560")
    sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
    print("Robot data listener started on tcp://127.0.0.1:5560...")
    while True:
        try:
            data_packet = sub_socket.recv_json()
            # print("Received data_packet:", data_packet)
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
    Sets the global orientation and translation used in transform_coordinates().
    """
    global orientation, translation
    orientation = new_orientation
    translation = new_translation

def transform_coordinates(local_coords):
    """
    Transforms local coordinates to global coordinates using the global orientation and translation.
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
    Update the global flag indicating whether a marker/worker has been spotted.
    
    Args:
        isSpotted (bool): True if detected; False otherwise.
    """
    global worker_spotted_flag
    worker_spotted_flag = isSpotted


