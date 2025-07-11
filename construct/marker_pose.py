# construct/marker_pose.py

import numpy as np
import time
from math import tan
from detections.marker_detector import MarkerDetector
from ui.mobility import *
from .utils import calculate_distance
import ui.MarkerDetectionLocalization as mdl
from .utils import twist, wiggle
from ui.mobility import translate, moveJ

from configs.system_config import *
from configs.construction_config import *

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
    Locate and align the robot with a part using its ArUco marker ID.

    This function uses an iterative vision feedback loop to center the robot’s end-effector over a part marked with a given ArUco ID, and then adjust the robot’s orientation so that the marker is directly below the gripper.

    **Parameters:**
    - *id_find (int)* – The ArUco marker ID of the target part (e.g., 0 for construction site, 1 for pickup, 10 for wall, etc.).
    - *frame_handler*: FrameHandler providing the latest camera image.

    **Returns:**
    - *list* – The robot’s joint configuration after final alignment (output of `mdl.get_joints()`).

    **Operation:**
    - Calls an internal `find_mark(id, frame_handler)` to translate the robot until the marker is centered under the camera (within a distance threshold of 0.2 in image coordinates).
    - Then enters a loop to correct orientation:
        * Captures image, gets `ang` (angle error of marker) from `mark_pos_ang`.
        * If the angle is significant (>0.005 radians difference), it adjusts the robot’s wrist joint (joint 6) by that angle to reduce orientation error, and recenters using `find_mark` again.
        * If `ang` returns a non-float (meaning marker lost or error), it performs a twist or wiggle (small motions defined in utils) to try to reacquire it.
        * Continues until the marker is essentially head-on (angle error negligible).
    - Final call to `find_mark` ensures final centering after orientation is corrected.
    - Returns the joint angles of the aligned pose.

    **Assumptions:**
    - The camera is mounted such that moving in the XY plane of the end-effector moves the marker in the image plane in a predictable way (calibration done via `frame_length_by_2`, etc. for pixel to meter conversion).
    - The marker is initially within view (the robot should be roughly in the correct area before calling this, e.g., `moveJ` to a scanning position).

    **Role in System:**
    This is a core part of the *ArUco-based localization* strategy:contentReference[oaicite:16]{index=16}:contentReference[oaicite:17]{index=17}. It is used heavily in `scan_site` and pick/place routines to ensure the robot precisely knows where a part or target location is before interacting with it. By iteratively aligning position and orientation to a marker, it achieves the centimeter-level accuracy needed for picking up and placing parts.
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

def align_with_random_part(id_find, length_mark,_frame_handler):
    """
    Align the end-effector with a randomly placed part using its marker, given a known marker size.

    This is similar to `find_part` but used for scenarios where the part’s marker is not one of the standard sizes 
    (using a provided `length_mark` for scale). It's applied in `pick_random_element` for demonstration or testing with arbitrary parts.

    **Parameters:**
    - *id_find (int)* – ArUco ID of the target to align with.
    - *length_mark (float)* – The real-world length of the marker side (in meters) for pixel-to-meter conversion.
    - *_frame_handler*: FrameHandler for obtaining images.

    **Returns:**
    - *list* – Robot joint configuration after alignment over the part.

    **Process:**
    - Internally defines `find_mark_random`, a version of `find_mark` that uses the provided marker length instead of inferring from ID categories.
    - Runs similar loops: first translate until centered (distance < 0.2), then adjust orientation using `ang` feedback, each time recentering.
    - Uses `twist` and `wiggle` motions if the marker is lost or angle returns a non-float (meaning an issue in detection).
    - Returns the joints after final alignment.

    **Use Case:**
    In a flexible or research setting, one might place random marked objects not defined in config. This function shows how to adapt the alignment routine with custom marker dimensions.

    **Note:** For consistency, the code structure mirrors `find_part`, but with an explicitly passed marker size instead of using `SITE_MARKER_SIZE` or `ELEMENT_MARKER_SIZE`.
    """
    def find_mark_random(id_find, length_mark, _frame_handler):
        """
        Moves the gripper so that the marker's corner is centered.
        Returns the robot's joint pose.
        """
        dist = 200
        while dist > 0.2:
            img = _frame_handler.get_latest_frame()
            mark_pos, mark_ang, length, bboxs, ids = mark_pos_ang(img, id_find)
            # length_mark = SITE_MARKER_SIZE if id_find in SITE_MARKER else ELEMENT_MARKER_SIZE
            if length:
                dist = calculate_distance(mark_pos[0], mark_pos[1], frame_length_by_2, frame_height)
                pix_per_meter = length / length_mark
                x_move = -(frame_length_by_2 - mark_pos[0]) / pix_per_meter
                y_move = -(frame_height - mark_pos[1]) / pix_per_meter
                translate((x_move, y_move, 0), ACC, VEL)
            else:
                time.sleep(1)
        return mdl.get_joints()
    _ = find_mark_random(id_find, length_mark,_frame_handler)
    ang = 1
    while abs(ang) > 0.005:
        img = _frame_handler.get_latest_frame()
        _, ang, _, _, _ = mark_pos_ang(img, id_find)
        if ang:
            posj = mdl.get_joints()
            new_joints = posj[:-1] + [posj[-1] - ang]
            moveJ(new_joints, ACC, VEL)
            _ = find_mark_random(id_find, length_mark, _frame_handler)
        elif not isinstance(ang, float):
            print('Debug: Calculated angle:', ang)
            twist(3)
            wiggle(5)
            _ = find_mark_random(id_find, length_mark, _frame_handler)
        else:
            print("Unable to find fiducial marker. Please check the setup.")
            wiggle(5)
            time.sleep(1)
    _ = find_mark_random(id_find, length_mark, _frame_handler)
    return mdl.get_joints()
