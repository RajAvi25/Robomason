#robot_controller/data_publisher.py
import zmq
import time

import collections.abc 
collections.Iterable=collections.abc.Iterable

import robot_controller.globals as rc_globals # Import the globals module to update the flag

def publish_robot_data(robot):
    """
    Continuously broadcast UR5e state over IPC for external monitoring.

    **Purpose & Role:**
    Streams the robot’s telemetry to `ipc:///tmp/robot.ipc` so that
    the tracking worker and safety manager can subscribe and detect 
    intrusions or log performance. This is the publisher counterpart 
    to `robot_data_listener` in `MarkerDetectionLocalization.py`.

    **Parameters:**
    - *robot*: A `urx.Robot` (or compatible) instance with methods:
        - `get_pos()`: returns [x, y, z, rx, ry, rz]
        - `getj()`: returns list of 6 joint angles
        - `get_orientation()`: returns 3×3 rotation matrix

    **Behavior:**
    1. Creates a ZeroMQ PUB socket bound to `ipc:///tmp/robot.ipc`.
       Removes any stale socket file at `/tmp/robot.ipc`.
    2. In an infinite loop (~20 Hz):
       - Reads:
         - `coords = robot.get_pos()` → the current pose.
         - `joints = robot.getj()` → the 6 joint angles.
         - `orientation = robot.get_orientation()` → EE rotation matrix.
       - Converts all NumPy arrays to native Python lists.
       - Reads the global flag `rc_globals.OBSTACLE_MANEUVER` to indicate 
         whether the robot is currently executing its 10 cm detour.
       - Packs these into a JSON‐serializable dict:
         ```python
         {
           "joints": joints,
           "orientation": [[…], […], […]],
           "EE cords": [x, y, z],
           "obstacle_maneuver": rc_globals.OBSTACLE_MANEUVER
         }
         ```
       - Sends it over the socket.
    3. Sleeps 50 ms between iterations to maintain ~20 Hz rate.

    **Assumptions & Dependencies:**
    - The `robot` object provides the stated methods and returns numeric types or NumPy arrays.
    - `rc_globals.OBSTACLE_MANEUVER` is toggled by the safety layer in `RobotController` 
      when a human marker intrusion is detected during motion.

    **Outputs:**
    - A ZeroMQ PUB on `ipc:///tmp/robot.ipc` that subscribers can connect to.
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

