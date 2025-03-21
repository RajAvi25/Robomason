#robot_controller/robot_controller.py
import zmq
import time
import json
import threading
from robot_controller.rc import RobotController
from robot_controller.data_publisher import publish_robot_data

def main():
    context = zmq.Context()
    socket_zmq = context.socket(zmq.REP)
    socket_zmq.bind("tcp://*:5556")
    print("Robot controller server is running and waiting for commands...")

    # Create one instance of RobotController (which connects immediately)
    robot_controller = RobotController()

    # Start a thread to continuously publish robot data
    publisher_thread = threading.Thread(target=publish_robot_data, args=(robot_controller.robot,), daemon=True)
    publisher_thread.start()

    while True:
        # Wait for a command (expects JSON messages)
        message = socket_zmq.recv_json()
        command = message.get("command", "")
        print(f"Received command: {command}")

        if command == "movej":
            argument = message.get("argument", "")
            result = robot_controller.movej(argument)
        elif command == "movel":
            argument = message.get("argument", "")
            result = robot_controller.movel(argument)
        elif command == "translate_tool":
            argument = message.get("argument", "")
            result = robot_controller.translate_tool(argument)
        elif command == "gripper_width":
            try:
                width = int(message.get("argument"))
            except Exception as e:
                result = f"Error parsing gripper_width argument: {e}"
            else:
                result = robot_controller.gripper_width(width)

        elif command == "set_orientation":
            argument = message.get("argument", "")
            result = robot_controller.set_orientation(argument)
        else:
            result = f"Unknown command: {command}"

        response = {"status": "ack", "result": result}
        socket_zmq.send_json(response)

if __name__ == '__main__':
    main()
