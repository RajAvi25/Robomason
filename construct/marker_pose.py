# construct/marker_pose.py

import numpy as np
import cv2
import time
from math import tan
from detections.marker_detector import MarkerDetector
from system_config import *
from construction_config import *
from ui.mobility import *
from .utils import calculate_distance
import ui.MarkerDetectionLocalization as mdl
from .utils import twist, wiggle
from system_config import *
from construction_config import *
from ui.mobility import translate, moveJ

def mark_pos_ang(img, id_find):
    """
    Process the image to find the fiducial marker with id_find.
    Returns: pos, angle, length, bboxs, ids.
    """
    bboxs, ids = MarkerDetector.find_objects(img)
    if bboxs:
        id_loc = np.where(ids == id_find)
        if id_loc[0].size:
            n_id = id_loc[0][0]
            pos = bboxs[n_id][0][0]
            dx = bboxs[n_id][0][0][0] - bboxs[n_id][0][3][0]
            dy = bboxs[n_id][0][0][1] - bboxs[n_id][0][3][1]
            ang = 0 if dy == 0 else tan(dx / dy)
            length = (dx**2 + dy**2) ** 0.5
        else:
            pos, ang, length, ids = [], 0, 0, []
            print("We can't see the correct marker!")
    else:
        pos, ang, length, ids = [], 0, 0, []
        print("I am unable to find a fiducial marker, so please help me")
    return pos, ang, length, bboxs, ids

def find_mark(id_find, frame_handler):
    """
    Moves the gripper so that the marker's corner is centered.
    Returns the robot's joint pose.
    """
    dist = 200
    while dist > 0.2:
        img = frame_handler.get_latest_frame()
        mark_pos, mark_ang, length, bboxs, ids = mark_pos_ang(img, id_find)
        length_mark = SITE_MARKER_SIZE if id_find in SITE_MARKER else ELEMENT_MARKER_SIZE
        if length:
            dist = calculate_distance(mark_pos[0], mark_pos[1], frame_length_by_2, frame_height)
            pix_per_meter = length / length_mark
            x_move = -(frame_length_by_2 - mark_pos[0]) / pix_per_meter
            y_move = -(frame_height - mark_pos[1]) / pix_per_meter
            translate((x_move, y_move, 0), ACC, VEL)
        else:
            time.sleep(1)
    return mdl.get_joints()

def find_part(id_find, frame_handler):
    """
    Find and orient the part relative to the marker.
    Returns the final joint positions after centering.
    """
    _ = find_mark(id_find, frame_handler)
    ang = 1
    while abs(ang) > 0.005:
        img = frame_handler.get_latest_frame()
        _, ang, _, _, _ = mark_pos_ang(img, id_find)
        if ang:
            posj = mdl.get_joints()
            new_joints = posj[:-1] + [posj[-1] - ang]
            moveJ(new_joints, ACC, VEL)
            _ = find_mark(id_find, frame_handler)
        elif not isinstance(ang, float):
            print('Debug: Calculated angle:', ang)
            twist(3)
            wiggle(5)
            _ = find_mark(id_find, frame_handler)
        else:
            print("Unable to find fiducial marker. Please check the setup.")
            wiggle(5)
            time.sleep(1)
    _ = find_mark(id_find, frame_handler)
    return mdl.get_joints()
