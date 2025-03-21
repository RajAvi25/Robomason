# plotting/config.py
import numpy as np

# Display & ZMQ configuration
display_robot = True
ZMQ_ADDRESS = "tcp://127.0.0.1:5555"
SUBSCRIBE_TOPIC = ""

# Limits for plots
z_level = -0.155
x_limits = (-0.35, 0.81)
y_limits = (-0.25, 0.75)
z_limits = (z_level, 0.49)

# RDP (Ramer-Douglas-Peucker) parameters
RDP_TRIGGER_COUNT = 2000
RDP_TOLERANCE = 0.01
PACKET_SKIP = 10

# Worker colors
worker_colors = {
    5: 'cyan', 6: 'green', 7: 'blue', 8: 'orange', 9: 'purple',
    13: 'brown', 14: 'pink', 15: 'lime', 16: 'red', 17: 'gold',
    18: 'magenta', 19: 'teal'
}

# UR5 DH parameters
DH_params = [
    {'theta': 0, 'a': 0, 'd': 0.1625, 'alpha': np.pi / 2},
    {'theta': 0, 'a': -0.425, 'd': 0, 'alpha': 0},
    {'theta': 0, 'a': -0.3922, 'd': 0, 'alpha': 0},
    {'theta': 0, 'a': 0, 'd': 0.1333, 'alpha': np.pi / 2},
    {'theta': 0, 'a': 0, 'd': 0.0997, 'alpha': -np.pi / 2},
    {'theta': 0, 'a': 0, 'd': 0.0996, 'alpha': 0}
]

# Element naming and color
ELEMENT_COLORS = {
    "Scanning site": "royalblue",
    "Foundation": "forestgreen",
    "Searching element": "crimson",
    "Wall 1": "gold",
    "Wall 2": "darkgoldenrod",
    "Floor 1": "indigo",
    "Floor 2": "orchid",
    "Bathroom module 1": "chocolate",
    "Bathroom module 2": "sienna",
    "Bathroom module 3": "tan",
}

# Line styles, widths, and alpha values
line_styles = {"grab": "-", "place": "-"}
line_widths = {"grab": 5.5, "place": 1.5}
line_alpha = {"grab": 1.0, "place": 0.7}
