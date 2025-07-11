#robot_controller/rc.py
import time
import threading
import socket
import json
import urx
import numpy as np
from robot_controller.globals import marker_stop_flag, prompt_user_lock
from robot_controller import globals as rc_globals  # Import the globals module to update the flag

_ENABLE_OBSTACLE_AVOIDANCE = True

class RobotController:
    """
    Controller for the UR5e robot arm, integrating direct URScript commands with safety checks.

    This class handles:
    - Connecting to the physical robot (UR5e) via network.
    - Controlling a custom parallel gripper through digital outputs.
    - Providing high-level movement commands (movej, movel, translate_tool, set_orientation) 
      that include an automatic stop and avoidance if a worker intrusion is detected.

    **Key Attributes:**
    - *robot*: The `urx.Robot` instance representing the robot (established on connect).
    - *gripper_socket*: TCP socket to the gripper controller (assuming a Robotiq or custom gripper via UR controller IO).
    - *debug (bool)*: Whether to print debug information about connections and motions.

    **Safety Mechanism – Worker-Aware Replanning:**
    Each motion command uses `_move_robot_command` which wraps `_execute_robot_command`. In `_execute_robot_command`, a background thread sends the move command to the robot, while the main thread monitors the global `marker_stop_flag`. If a worker’s ArUco marker is detected (flag is set by `marker_trigger`), the robot:
    - Stops the motion immediately (`robot.stopj()`),
    - Sets a resume flag to allow later continuation,
    - If obstacle avoidance is enabled, performs an automatic retreat (`makerAvoidmovement()` which moves the arm slightly away from the obstacle) while `rc_globals.OBSTACLE_MANEUVER` is True:contentReference[oaicite:22]{index=22},
    - Once avoidance is done, it clears the flag and signals the move as incomplete.

    The outer loop `_move_robot_command` will then retry the intended motion. This results in a brief detour whenever a human intrudes, then a resumption of the task once clear – implementing the *real-time replanning around a human*:contentReference[oaicite:23]{index=23}.

    **Hand-Tuned Motion Adjustments:**
    The class defines helper logic like `move_inwards` in `_execute_robot_command` to slightly retract along a vector (used in makerAvoidmovement with `safetyDist=0.1m`). These distances and speeds for avoidance are manually chosen to balance safety and speed.

    **Methods:**
    - `connect_to_robot()`: Attempts to connect to the UR5e multiple times.
    - `gripper_width(width)`: Opens/closes gripper to preset widths (0=fully closed, 100=fully open, with intermediate discrete steps).
    - `_execute_robot_command(command_fn, pos, acc, vel, name)`: Low-level execution with safety monitoring.
    - `_move_robot_command(...)`: Retries motion if `_execute_robot_command` indicates it was interrupted by safety stop.
    - High-level motion commands: `movej`, `movel`, `translate_tool`, `set_orientation` – all parse input, ensure robot connection, and call `_move_robot_command` with appropriate URScript command.

    **Usage:**
    This controller is used under the hood by higher-level functions in `ui.mobility` (likely a convenience wrapper for these methods to be called easily from the UI or scripts). It abstracts the direct robot commands and injects the safety layer.
    """
    def __init__(self, host="192.168.1.10", port=30002, debug=True):
        self.host = host
        self.port = port
        self.debug = debug
        self.robot = self.connect_to_robot()
        # Start marker detection using the common camera frame handler.
        from camera.frame_handler import cam_init
        cam_thread = threading.Thread(target=cam_init, daemon=True)
        cam_thread.start()
        # Create a socket connection for the gripper.
        self.gripper_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.gripper_socket.connect((self.host, self.port))
        except Exception as e:
            print(f"Warning: Could not create socket connection: {e}")
        self.gripper_width(50)
        self.gripper_width(100)

    def connect_to_robot(self, max_retries=10, delay_between_retries=2.5):
        ip = self.host
        while True:
            try:
                robot = urx.Robot(ip, use_rt=True)
                if self.debug:
                    print("Connected to the robot successfully.")
                return robot
            except Exception as e:
                print(f"Connection failed: {e}")
                for _ in range(max_retries):
                    try:
                        print(f"Retrying connection to {ip}...")
                        robot = urx.Robot(ip, use_rt=True)
                        print("Connected to the robot successfully.")
                        return robot
                    except Exception as inner_e:
                        print(f"Failed to connect: {inner_e}")
                        print(f"Waiting {delay_between_retries} seconds before retrying...")
                        time.sleep(delay_between_retries)
                print("Exceeded maximum retries. Trying again after a short delay...")
                time.sleep(delay_between_retries)

    def gripper_width(self, width):
        t_sleep = 0.05
        time.sleep(t_sleep)
        self.gripper_socket.send(("set_digital_out(0,False)\n").encode())
        time.sleep(t_sleep)
        self.gripper_socket.send(("set_digital_out(1,False)\n").encode())
        time.sleep(t_sleep)
        self.gripper_socket.send(("set_digital_out(2,False)\n").encode())
        time.sleep(t_sleep)
        self.gripper_socket.send(("set_digital_out(3,False)\n").encode())
        time.sleep(t_sleep)
        self.gripper_socket.send(("set_digital_out(4,False)\n").encode())
        time.sleep(t_sleep)
        self.gripper_socket.send(("set_digital_out(5,False)\n").encode())
        time.sleep(t_sleep)
        self.gripper_socket.send(("set_digital_out(6,False)\n").encode())
        time.sleep(t_sleep)
        self.gripper_socket.send(("set_digital_out(7,False)\n").encode())
        time.sleep(t_sleep)
        if width == 0:
            self.gripper_socket.send(("set_digital_out(0,True)\n").encode())
        elif width == 20:
            self.gripper_socket.send(("set_digital_out(1,True)\n").encode())
        elif width == 50:
            self.gripper_socket.send(("set_digital_out(2,True)\n").encode())
        elif width == 70:
            self.gripper_socket.send(("set_digital_out(3,True)\n").encode())
        elif width == 100:
            self.gripper_socket.send(("set_digital_out(4,True)\n").encode())
        elif width == 60:
            self.gripper_socket.send(("set_digital_out(5,True)\n").encode())
        elif width == 65:
            self.gripper_socket.send(("set_digital_out(6,True)\n").encode())
        elif width == 95:
            self.gripper_socket.send(("set_digital_out(7,True)\n").encode())
        else:
            print("Width not defined")
        return f"Gripper width set to {width}"

    def _execute_robot_command(self, command_fn, pos, acc, vel, command_name):
        """
        Internal helper to execute a robot move command with safety monitoring.

        Launches the given URX `command_fn` (like robot.movej or movel) in a thread so that we can monitor 
        the `marker_stop_flag` concurrently. If a stop is signaled (meaning a worker intruded), it stops the robot and optionally performs avoidance.

        **Parameters:**
        - *command_fn*: Function handle to the URX move command (e.g., self.robot.movej).
        - *pos*: Position target (list of 6 floats for joint targets or pose).
        - *acc (float)*: Acceleration for the move.
        - *vel (float)*: Velocity for the move.
        - *command_name (str)*: Name of the command (for logging/debug).

        **Returns:**
        - *bool* – True if the move completed without interruption, False if it was stopped due to a marker detection.

        **Mechanism:**
        - Starts the move in a separate thread (so it doesn't block).
        - In the main thread, loops until that thread finishes, checking every 0.05s if `marker_stop_flag` is set.
        - If a marker is detected (flag set), it prints a message, stops the robot smoothly (`stopj(1.0)` for a 1s stop ramp), and sets `stop_signal`.
        - After the move thread ends (either normally or via stop), if a stop occurred:
            * Within a locked section (`prompt_user_lock` to avoid race with other prompts), 
              it checks the global `_ENABLE_OBSTACLE_AVOIDANCE`. If enabled:
                - Sets `rc_globals.OBSTACLE_MANEUVER = True` (so the rest of system knows we are in avoidance mode, e.g., data_publisher will include this:contentReference[oaicite:24]{index=24}).
                - Clears the marker_stop_flag (to allow subsequent detections).
                - Calls `makerAvoidmovement()` which computes a slight retreat pose (10cm back along the vector from current pose to origin) and executes a slow `movel` to that pose. This is the detour step to create distance from the obstacle (worker).
                - After that move, sets `rc_globals.OBSTACLE_MANEUVER = False` and returns False indicating interruption.
            * If avoidance is not enabled, it would simply return False (stop without detour).
        - If no stop occurred, returns True.

        **Note:** `makerAvoidmovement()` uses the current robot position and orientation to move outwards (or inwards to origin) by `safetyDist`. It's a simplistic avoidance: effectively backing off straight-line from whatever point it was stopped.

        **Threading Consideration:** The use of `prompt_user_lock` ensures that if the robot was stopped and is about to perform avoidance, no other prompt (e.g., a verify prompt) interferes. It treats the avoidance as a critical section.
        """
        stop_signal = threading.Event()
        robot = self.robot

        def move_inwards(point, n, is3d=True):
            point = np.array(point)
            if not is3d and len(point) == 3:
                point = point[:2]
            origin = np.zeros_like(point)
            direction = origin - point
            magnitude = np.linalg.norm(direction)
            if magnitude == 0:
                return tuple(point) if is3d else (point[0], point[1])
            unit_vector = direction / magnitude
            new_point = point + n * unit_vector
            if not is3d:
                new_point = (new_point[0], new_point[1])
            return tuple(new_point)

        def rotation_matrix_to_vector(rotation_matrix):
            theta = np.arccos((np.trace(rotation_matrix) - 1) / 2)
            if np.isclose(theta, 0):
                return np.zeros(3)
            r = np.array([
                rotation_matrix[2, 1] - rotation_matrix[1, 2],
                rotation_matrix[0, 2] - rotation_matrix[2, 0],
                rotation_matrix[1, 0] - rotation_matrix[0, 1]
            ]) / (2 * np.sin(theta))
            return r * theta

        def makerAvoidmovement():
            dimension = 3
            safetyDist = 0.1
            current_coords = tuple(robot.get_pos())
            orient = robot.get_orientation()
            original_z = current_coords[2]
            if dimension not in {2, 3}:
                print("Undefined. Choose either 2D or 3D.")
                return
            current_coordinates = tuple(current_coords[i] for i in range(dimension))
            corrected_loc = move_inwards(current_coordinates, safetyDist, len(current_coordinates)==3)
            if dimension == 2:
                corrected_loc = (corrected_loc[0], corrected_loc[1], original_z)
            rotation_matrix = np.array([orient[0], orient[1], orient[2]])
            rotation_vector = rotation_matrix_to_vector(rotation_matrix)
            target_pose = list(corrected_loc) + rotation_vector.tolist()
            robot.movel(target_pose, acc=0.1, vel=0.1)

        # Thread to execute robot motion
        def move_thread_func():
            try:
                if self.debug:
                    print(f"Starting robot move ({command_name})...")
                command_fn(pos, acc, vel)
                if self.debug:
                    print(f"Robot move ({command_name}) completed successfully.")
            except Exception as e:
                if 'Goal not reached but no program has been running' in str(e):
                    if self.debug:
                        print(f"Forced stop encountered in {command_name}. Ignoring expected error.")
                else:
                    if self.debug:
                        print(f"Unexpected error in move_thread_func for {command_name}: {e}")
            finally:
                stop_signal.set()

        move_thread = threading.Thread(target=move_thread_func, daemon=True)
        move_thread.start()

        # Monitor thread for marker stop
        while not stop_signal.is_set():
            if marker_stop_flag.is_set():
                print(f"Marker detected: stopping robot mid-motion ({command_name})...")
                robot.stopj(1.0)
                stop_signal.set()
            time.sleep(0.05)
        move_thread.join()

        # After movement attempt:
        if marker_stop_flag.is_set():
            with prompt_user_lock:
                if _ENABLE_OBSTACLE_AVOIDANCE:
                    rc_globals.OBSTACLE_MANEUVER = True  # Set the flag before avoidance
                    marker_stop_flag.clear()
                    makerAvoidmovement()
                    time.sleep(0.1)
                    rc_globals.OBSTACLE_MANEUVER = False  # Reset the flag after avoidance
                return False
            
        return True

    def _move_robot_command(self, command_fn, pos, acc, vel, command_name):
        """
        Wrapper to execute a robot command, retrying if interrupted by safety stop.

        This will call `_execute_robot_command`. If it returns False (meaning the robot was stopped due to a marker detection and performed avoidance), 
        it will log that event (if debug) and then **retry** the motion by looping back until a True is returned.

        This means the robot will attempt the motion repeatedly until it either succeeds or is manually interrupted, 
        effectively implementing *resilient motion that pauses and resumes around obstacles*.

        **Parameters:** same as `_execute_robot_command`.

        **Returns:** None – it prints/logs the outcome and ensures the motion is carried out eventually.
        """
        while True:
            result = self._execute_robot_command(command_fn, pos, acc, vel, command_name)
            if result:
                if self.debug:
                    print(f"Robot move ({command_name}) completed or no forced stop.")
                break
            else:
                if self.debug:
                    print(f"Marker was detected and avoidance maneuver executed during {command_name}. Retrying move...")

    def movej(self, destination_str):
        """
        Execute a joint-space movement (moveJ) to the target joint angles with safety monitoring.

        **Parameters:**
        - *destination_str (str)* – A JSON string or Python literal string that decodes to `[position, acceleration, velocity]`.
            - position: list of 6 joint target angles (radians).
            - acceleration: float (rad/s^2).
            - velocity: float (rad/s).

        **Returns:**
        - *str* – Confirmation message or error message.

        **Function:**
        - Parses the input string into a Python object.
        - Validates format.
        - If robot is not connected, attempts to reconnect.
        - Uses `_move_robot_command(self.robot.movej, ...)` to perform the motion with obstacle avoidance enabled.
        - If successful, returns a message that movej was executed.

        This is typically called via a remote interface (like Jupyter) sending JSON commands to the robot. 
        """
        try:
            destination = json.loads(destination_str)
            if not isinstance(destination, list) or len(destination) != 3:
                return "Error: Destination must be a list of 3 elements: [position, acceleration, velocity]."
        except Exception as e:
            return f"Error parsing destination: {e}"
        pos = destination[0]
        acc = destination[1]
        vel = destination[2]
        if not isinstance(pos, list) or len(pos) != 6:
            return "Error: For movej, position must be a list of 6 numbers."
        if not self.robot:
            self.robot = self.connect_to_robot()
        self._move_robot_command(self.robot.movej, pos, acc, vel, "movej")
        return f"movej executed with destination: {destination}"

    def movel(self, destination_str):
        try:
            destination = json.loads(destination_str)
            if not isinstance(destination, list) or len(destination) != 3:
                return "Error: Destination must be a list of 3 elements: [position, acceleration, velocity]."
        except Exception as e:
            return f"Error parsing destination: {e}"
        pos = destination[0]
        acc = destination[1]
        vel = destination[2]
        if not isinstance(pos, list) or len(pos) != 6:
            return "Error: For movel, position must be a list of 6 numbers."
        if not self.robot:
            self.robot = self.connect_to_robot()
        self._move_robot_command(self.robot.movel, pos, acc, vel, "movel")
        return f"movel executed with destination: {destination}"

    def translate_tool(self, destination_str):
        try:
            destination = json.loads(destination_str)
            if not isinstance(destination, list) or len(destination) != 3:
                return "Error: Destination must be a list of 3 elements: [position, acceleration, velocity]."
        except Exception as e:
            return f"Error parsing destination: {e}"
        pos = destination[0]
        acc = destination[1]
        vel = destination[2]
        if not isinstance(pos, list) or len(pos) != 3:
            return "Error: For translate_tool, position must be a list of 3 numbers."
        if not self.robot:
            self.robot = self.connect_to_robot()
        self._move_robot_command(self.robot.translate_tool, pos, acc, vel, "translate_tool")
        return f"translate_tool executed with destination: {destination}"
    
    def set_orientation(self, destination_str):
        try:
            destination = json.loads(destination_str)
            if not isinstance(destination, list) or len(destination) != 3:
                return "Error: Destination must be a list of 3 elements: [orientation, acceleration, velocity]."
        except Exception as e:
            return f"Error parsing destination: {e}"
        
        orientation = destination[0]
        acc = destination[1]
        vel = destination[2]

        if not self.robot:
            self.robot = self.connect_to_robot()
        # Using the _move_robot_command helper to handle execution and safety checks.
        self._move_robot_command(self.robot.set_orientation, orientation, acc, vel, "set_orientation")
        return f"set_orientation executed with destination: {destination}"

