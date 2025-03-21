#robot_controller/rc.py
import time
import threading
import socket
import json
import urx
import numpy as np
from robot_controller.globals import marker_stop_flag, prompt_user_lock
from robot_controller import globals as rc_globals  # Import the globals module to update the flag

class RobotController:
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

        while not stop_signal.is_set():
            if marker_stop_flag.is_set():
                print(f"Marker detected: stopping robot mid-motion ({command_name})...")
                robot.stopj(1.0)
                stop_signal.set()
            time.sleep(0.05)
        move_thread.join()

        # if marker_stop_flag.is_set():
        #     with prompt_user_lock:
        #         marker_stop_flag.clear()
        #         makerAvoidmovement()
        #         time.sleep(0.1)
        #         return False

        if marker_stop_flag.is_set():
            with prompt_user_lock:
                rc_globals.OBSTACLE_MANEUVER = True  # Set the flag before avoidance
                marker_stop_flag.clear()
                makerAvoidmovement()
                time.sleep(0.1)
                rc_globals.OBSTACLE_MANEUVER = False  # Reset the flag after avoidance
                return False
            
        return True

    def _move_robot_command(self, command_fn, pos, acc, vel, command_name):
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

