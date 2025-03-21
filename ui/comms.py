import zmq
import cv2
import base64
import time
import msgpack
import json
import threading
from threading import Lock
from camera.frame_handler import FrameHandler
from ui.mobility import set_robot_control_socket
from . import MarkerDetectionLocalization as mdl
from construct import construction_status


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



