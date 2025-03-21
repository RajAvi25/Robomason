import json

# Global variable to hold the REQ socket for robot control.
robot_control_socket = None

def set_robot_control_socket(socket):
    global robot_control_socket
    robot_control_socket = socket

def moveJ(pos, acc, vel):
    argument_str = json.dumps([pos, acc, vel])
    payload = {"command": "movej", "argument": argument_str}
    robot_control_socket.send_json(payload)
    reply = robot_control_socket.recv_json()
    return reply

def moveL(pos, acc, vel):
    argument_str = json.dumps([pos, acc, vel])
    payload = {"command": "movel", "argument": argument_str}
    robot_control_socket.send_json(payload)
    reply = robot_control_socket.recv_json()
    return reply

def translate(pos, acc, vel):
    argument_str = json.dumps([pos, acc, vel])
    payload = {"command": "translate_tool", "argument": argument_str}
    robot_control_socket.send_json(payload)
    reply = robot_control_socket.recv_json()
    return reply

def gripper_width(width):
    payload = {"command": "gripper_width", "argument": width}
    robot_control_socket.send_json(payload)
    reply = robot_control_socket.recv_json()
    return reply

def set_orientation(orient,acc,vel):
    argument_str = json.dumps([orient, acc, vel])
    payload = {"command": "set_orientation", "argument": argument_str}
    robot_control_socket.send_json(payload)
    reply = robot_control_socket.recv_json()
    return reply
