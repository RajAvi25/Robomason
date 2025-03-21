# construct/utils.py

import math
import random
from ui.mobility import moveJ, translate
from system_config import *
from construction_config import *

def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def generate_random_value(base_value, threshold):
    """Generate a small random variation around a base value."""
    variation = random.uniform(-threshold, threshold)
    sign = random.choice([-1, 1])
    return sign * (base_value + variation)

def wiggle(turns, base=0.005, variance=0.0005):
    """Perform a wiggle motion by translating the gripper slightly."""
    for _ in range(turns):
        dx = generate_random_value(base, variance)
        dy = generate_random_value(base, variance)
        translate((dx, dy, 0), ACC, VEL)

def twist(turns, base=0.005, variance=0.0005):
    """Twist the robot's joint slightly to adjust orientation."""
    from ui.MarkerDetectionLocalization import get_joints
    for _ in range(turns):
        posj = get_joints()
        new_joints = posj[:-1] + [posj[-1] - generate_random_value(0.05, 0.025)]
        moveJ(new_joints, ACC, VEL)
