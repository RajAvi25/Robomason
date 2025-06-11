#robot_controller/data_publisher.py
import zmq
import time

import collections.abc 
collections.Iterable=collections.abc.Iterable

import robot_controller.globals as rc_globals # Import the globals module to update the flag

# def publish_robot_data(robot):
#     context = zmq.Context()
#     pub_socket = context.socket(zmq.PUB)
#     pub_socket.bind("tcp://127.0.0.1:5560")
#     while True:
#         try:
#             coords = robot.get_pos()
#             joints = robot.getj()
#             orientation = robot.get_orientation()
#             orientation = [orientation[0].tolist(), orientation[1].tolist(), orientation[2].tolist()]
#             coords = [coords[0], coords[1], coords[2]]
#             data_packet = {
#                 "joints": joints,
#                 "orientation": orientation,
#                 "EE cords": coords,
#                 "obstacle_maneuver":rc_globals.OBSTACLE_MANEUVER
#             }
#             pub_socket.send_json(data_packet)
#         except Exception as e:
#             print(f"Error publishing robot data: {e}")
#         time.sleep(0.05)

def publish_robot_data(robot):
    """
    Publishes robot data at ~20 Hz on ipc:///tmp/robot.ipc.
    """
    context = zmq.Context.instance()
    pub_socket = context.socket(zmq.PUB)
    # ensure old socket file is removed if present
    try:
        import os
        os.remove("/tmp/robot.ipc")
    except OSError:
        pass
    pub_socket.bind("ipc:///tmp/robot.ipc")
    print("Publishing robot data on ipc:///tmp/robot.ipc...")
    while True:
        try:
            coords = robot.get_pos()
            joints = robot.getj()
            orientation = robot.get_orientation()
            # convert any numpy arrays to lists
            orientation = [orientation[i].tolist() for i in range(3)]
            coords = [coords[0], coords[1], coords[2]]
            data_packet = {
                "joints": joints,
                "orientation": orientation,
                "EE cords": coords,
                "obstacle_maneuver": rc_globals.OBSTACLE_MANEUVER
            }
            pub_socket.send_json(data_packet)
        except Exception as e:
            print(f"Error publishing robot data: {e}")
        time.sleep(0.05)

