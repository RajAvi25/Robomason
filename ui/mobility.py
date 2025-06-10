# import json

# # Global variable to hold the REQ socket for robot control.
# robot_control_socket = None

# def set_robot_control_socket(socket):
#     global robot_control_socket
#     robot_control_socket = socket

# def moveJ(pos, acc, vel):
#     argument_str = json.dumps([pos, acc, vel])
#     payload = {"command": "movej", "argument": argument_str}
#     robot_control_socket.send_json(payload)
#     reply = robot_control_socket.recv_json()
#     return reply

# def moveL(pos, acc, vel):
#     argument_str = json.dumps([pos, acc, vel])
#     payload = {"command": "movel", "argument": argument_str}
#     robot_control_socket.send_json(payload)
#     reply = robot_control_socket.recv_json()
#     return reply

# def translate(pos, acc, vel):
#     argument_str = json.dumps([pos, acc, vel])
#     payload = {"command": "translate_tool", "argument": argument_str}
#     robot_control_socket.send_json(payload)
#     reply = robot_control_socket.recv_json()
#     return reply

# def gripper_width(width):
#     payload = {"command": "gripper_width", "argument": width}
#     robot_control_socket.send_json(payload)
#     reply = robot_control_socket.recv_json()
#     return reply

# def set_orientation(orient,acc,vel):
#     argument_str = json.dumps([orient, acc, vel])
#     payload = {"command": "set_orientation", "argument": argument_str}
#     robot_control_socket.send_json(payload)
#     reply = robot_control_socket.recv_json()
#     return reply

# ui/mobility.py
import zmq
import json

# ————— globals —————
_ROBOT_ADDR = None  # string like "tcp://1.2.3.4:5555"

def set_robot_control_socket(sock):
    """
    Backwards-compatible setter for ui.comms.
    Extracts the last endpoint from the passed ZMQ REQ socket,
    closes it, and remembers that address for future calls.
    """
    global _ROBOT_ADDR

    try:
        # ZMQ_LAST_ENDPOINT holds the peer you connected to.
        endpoint = sock.getsockopt(zmq.LAST_ENDPOINT).decode('utf-8')
    except Exception as e:
        raise RuntimeError(
            "Failed to extract endpoint from passed socket. "
            "Make sure you're passing a connected REQ socket."
        ) from e

    # close the old socket so we don't leak handles
    try:
        sock.close(linger=0)
    except Exception:
        pass

    _ROBOT_ADDR = endpoint


def _do_req(payload):
    """
    Internal helper: create a fresh REQ socket, send/recv, then close.
    """
    if not _ROBOT_ADDR:
        raise RuntimeError(
            "Robot address not set. "
            "ui.comms must call set_robot_control_socket() first."
        )

    ctx = zmq.Context.instance()
    s   = ctx.socket(zmq.REQ)
    s.connect(_ROBOT_ADDR)
    try:
        s.send_json(payload)
        reply = s.recv_json()
    finally:
        s.close(linger=0)
    return reply


def moveJ(pos, acc, vel):
    """
    Joint-move to 'pos' with acceleration acc and velocity vel.
    """
    return _do_req({
        "command":  "movej",
        "argument": json.dumps([pos, acc, vel])
    })


def moveL(pos, acc, vel):
    """
    Linear-move to 'pos' with acceleration acc and velocity vel.
    """
    return _do_req({
        "command":  "movel",
        "argument": json.dumps([pos, acc, vel])
    })


def translate(pos, acc, vel):
    """
    Translate tool by 'pos' with acceleration acc and velocity vel.
    """
    return _do_req({
        "command":  "translate_tool",
        "argument": json.dumps([pos, acc, vel])
    })


def gripper_width(width):
    """
    Set gripper opening to 'width'.
    """
    return _do_req({
        "command":  "gripper_width",
        "argument": width
    })


def set_orientation(orient, acc, vel):
    """
    Set end-effector orientation to 'orient' with acceleration acc and velocity vel.
    """
    return _do_req({
        "command":  "set_orientation",
        "argument": json.dumps([orient, acc, vel])
    })
