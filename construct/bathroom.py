# construct/bathroom.py

import cv2
from func_timeout import FunctionTimedOut, func_timeout
import random
import numpy as np
import matplotlib.pyplot as plt
from ui.mobility import *
from .utils import calculate_distance
import ui.MarkerDetectionLocalization as mdl
import time
from math import pi
from .construction import swing

from . import construction_status

from configs.system_config import *
from configs.construction_config import *

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

def find_center(frame, LB, UB, min_area, max_area, debug=False):
    """
    Process the frame to detect objects in the specified HSV range.
    Returns centers and geometric data.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    if hasattr(LB, 'ndim') and LB.ndim == 1:
        mask = cv2.inRange(hsv, LB, UB)
    else:
        masks = [cv2.inRange(hsv, LB[i], UB[i]) for i in range(len(LB))]
        mask = masks[0]
        for m in masks[1:]:
            mask = cv2.bitwise_or(mask, m)
    imgCanny = cv2.Canny(mask, 100, 100)
    kernel = np.ones((5, 5))
    imgDilate = cv2.dilate(imgCanny, kernel, iterations=3)
    imgErode = cv2.erode(imgDilate, kernel, iterations=2)
    contours, _ = cv2.findContours(imgErode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    geos = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            enclosing_area = np.pi * (radius ** 2)
            if area / enclosing_area > 0.8:
                centers.append(center)
                bbox = cv2.minAreaRect(cnt)
                geos.append(bbox[1])
    if debug:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(mask, cmap='gray')
        plt.title("Mask")
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title("Frame")
        plt.show()
    return centers, geos

def check_frame(color_thresholds, area_thresholds, frame_handler):
    """"
    Check the current camera frame for presence of red or blue objects (toilet modules).

    **Parameters:**
    - *color_thresholds (dict)* – Contains HSV lower/upper bounds for "red" and "blue" keys, e.g.:
         `{"red": {"LB": lower_bound_array, "UB": upper_bound_array}, "blue": {...}}`
    - *area_thresholds (dict)* – Dict with "min" and "max" pixel area for contours to be considered a valid block.
    - *frame_handler*: FrameHandler to get the latest frame from the camera.

    **Returns:**
    - *(red_found, blue_found)* – Tuple of bools indicating if a red object or blue object was found in the image.

    **Process:**
    - Grabs the latest frame from the camera.
    - Uses `find_center` to detect objects in the red and blue HSV ranges (with size filtering).
    - If any contour passes the area check for red, `red_found` is True; similarly for blue.
    - This is typically used to decide which colored module is available or to prompt the user to choose.

    **Role:**
    In the assembly context, this helps the system automatically identify which colored bathroom modules (if any) are present in view, enabling *semi-automated selection* of a red or blue module to place.
    """
    frame = frame_handler.get_latest_frame()
    red_centers, _ = find_center(frame,
                                 color_thresholds["red"]["LB"],
                                 color_thresholds["red"]["UB"],
                                 area_thresholds["min"],
                                 area_thresholds["max"])
    blue_centers, _ = find_center(frame,
                                  color_thresholds["blue"]["LB"],
                                  color_thresholds["blue"]["UB"],
                                  area_thresholds["min"],
                                  area_thresholds["max"])
    return bool(red_centers), bool(blue_centers)

def grab_toilet(LB, UB, frame_handler):
    """
    Grab a bathroom module (toilet block) of a specified color using visual feedback.

    This function performs a closed-loop alignment to pick up a colored block (red or blue), 
    using multiple image frames to gradually center and approach the object.

    **Parameters:**
    - *LB, UB*: NumPy arrays (or lists) representing the HSV lower-bound and upper-bound for the target color mask.
    - *frame_handler*: FrameHandler providing the latest camera frame.

    **Returns:**
    - *placed_pos (dict) or int* – If successful, returns a dictionary with `"coords"` and `"orientation"` of the end-effector after grabbing the block. If it fails to find the block within a few attempts, returns 0.

    **Strategy:**
    - The robot looks for the colored block in the current frame (`find_center`).
    - If found, it computes the offset in image pixels and translates the robot end-effector in the XY plane to center the gripper over the block.
    - It repeats the detection at a closer range for fine alignment.
    - Once the block is centered in view twice (initial and secondary detection), the robot closes the gripper slightly (width 50) to touch the block, then moves down a fixed small amount (z_offset relative to a known ground plane) and fully closes (`gripper_width(0)`) to grasp it.
    - After grasping, it records the end-effector position (which is effectively the pickup pose of the module).
    - The gripper then lifts the block slightly.

    **Assumptions:**
    - The block is within camera view and identifiable by color.
    - The camera is roughly above the block to start (the calling routine likely moved the robot above the storage area where modules lie).
    - `bathroom_module_radius` (from config) is known for size calibration (pixels per meter calculation).

    **Role in System:**
    This provides an automated way to handle *non-IFC items* (toilet modules) using simple vision. It complements the IFC-driven part of the framework by handling these extra pieces with a feedback loop, contributing to the *end-to-end assembly pipeline*.
    """
    dist_smallest = 200
    count = 0
    second_count = 0
    while dist_smallest > 0.5:
        frame = frame_handler.get_latest_frame()
        centers, geos = find_center(frame, LB, UB,
                                    bathroom_module_area_thresholds["min"],
                                    bathroom_module_area_thresholds["max"])
        if not centers:
            print("No centers found, retrying")
            count += 1
            if count >= 5:
                return 0
            continue
        distances = [calculate_distance(c[0], c[1],frame_length_by_2, frame_height)
                     for c in centers]
        min_ind = distances.index(min(distances))
        dist_smallest = distances[min_ind]
        block_x, block_y = centers[min_ind]
        r_meas = np.mean(geos[min_ind])
        pixpm = r_meas / bathroom_module_radius
        x_move = -(frame_length_by_2 - block_x) / pixpm
        y_move = -(frame_height - block_y) / pixpm
        translate((x_move, y_move, 0), ACC, VEL)
        
        frame = frame_handler.get_latest_frame()
        new_centers, new_geos = find_center(frame, LB, UB,
                                            bathroom_module_area_thresholds["min"],
                                            bathroom_module_area_thresholds["max"])
        if not new_centers:
            print("Second detection failed, retrying")
            second_count += 1
            if second_count >= 5:
                return 0
            continue
        new_distances = [calculate_distance(c[0], c[1], frame_length_by_2, frame_height)
                         for c in new_centers]
        new_min_ind = new_distances.index(min(new_distances))
        dist_smallest = new_distances[new_min_ind]
        new_block_x, new_block_y = new_centers[new_min_ind]
        new_r_meas = np.mean(new_geos[new_min_ind])
        new_pixpm = new_r_meas / bathroom_module_radius
        new_x_move = -(frame_length_by_2 - new_block_x) / new_pixpm
        new_y_move = -(frame_height - new_block_y) / new_pixpm
        translate((new_x_move, new_y_move, 0), ACC, VEL)
        gripper_width(50)
        z_offset = mdl.get_EE_coords()[2] - ground_z_zero - 0.009
        translate((camera_offsets['x'], camera_offsets['y'], z_offset), ACC, VEL)
        gripper_width(0)
        time.sleep(3)
        placed_pos = {
        "coords": mdl.get_EE_coords(),
        "orientation": mdl.get_orientation()
        }
        translate((0, 0, -z_offset), ACC, VEL)
        return placed_pos

def place_toilet(IFC_sorted, n_placed):
    """
    Place the grabbed bathroom module onto the previously placed part (foundation or floor).

    This uses the coordinates of the last placed structural element from `IFC_sorted` to position the toilet block.

    **Parameters:**
    - *IFC_sorted (np.ndarray)* – Sorted IFC elements array.
    - *n_placed (int)* – Index of the structural part on which to place the module:
         - 0 for foundation (module 1),
         - 2 for first floor (module 2),
         - 4 for second floor (module 3),
         etc. (This mapping corresponds to how bathroom modules were inserted after foundations/floors.)

    **Returns:**
    - *placed_pos (dict)* – End-effector pose after placing the module (with `"coords"` and `"orientation"`).

    **Process:**
    - Determines the reference part's average position from `IFC_sorted[n_placed]` (this is the part below the module).
    - Moves the robot above the construction site (`ct_pos` should be pre-set) and translates by the part’s X/Y (converted to meters).
    - If the part is a floor vs foundation:
         - For Floor: rotates wrist by 90° (since toilet module needs a certain orientation) and uses one of two sets of offsets (`x_offest_bathroom_module_place_1/2`, etc.) depending on whether it's the first or second floor module.
         - For Foundation: uses a different offset (`x_extra, y_extra, z_extra`) to align module on foundation.
    - Lowers the module by a computed Z (difference between current EE Z and part’s Z plus an offset), then opens gripper to release.
    - Lifts up slightly and returns the final pose.

    **Internal Calibration:**
    - `n_placed == 2` and `4` trigger the offsets for second module and third module respectively (the naming is a bit confusing with `bathroom_module_2` possibly placed on Floor1 and `bathroom_module_3` on Floor2).
    - A mysterious 0.98 factor is applied if this is the last item (perhaps to account for slight overestimation of height – “don’t question this” comment).

    **Role in System:**
    This performs the "hand-tuned" placement of bathroom modules on top of structural pieces. It exemplifies the *hybrid approach*: using IFC data for base position but applying manual offsets for alignment:contentReference[oaicite:15]{index=15}. The orientation set to `orientations['ct']` ensures the module is placed upright.
    """
    prev_part = IFC_sorted[n_placed, 1:4]
    prev_part_name = IFC_sorted[n_placed, 0]
    name = prev_part_name.split(":")[1].split("-")[0].strip()
    
    x_move = float(prev_part[0]) / 100
    y_move = float(prev_part[1]) / 100
    z_move = mdl.get_EE_coords()[2] - float(prev_part[2]) / 100
 
    translate((x_move, y_move, 0), ACC, VEL)
    
    if name == "Floor":
        print("Placing toilet on floor")
        set_orientation(orientations['ct'],ACC, VEL)

        # current_posj = mdl.get_joints()
        # current_posj[5] += pi/2
        # moveJ(current_posj, ACC, VEL)

        if n_placed == 2:
            print('Second bathroom')
            x_extra = x_offest_bathroom_module_place_1 
            y_extra = y_offest_bathroom_module_place_1 
            z_extra = z_offest_bathroom_module_place_1 
        elif n_placed == 4:
            x_extra = x_offest_bathroom_module_place_2 
            y_extra = y_offest_bathroom_module_place_2 
            z_extra = z_offest_bathroom_module_place_2 

        print(f"I am at: {mdl.get_EE_coords()}")
        translate((x_extra, y_extra,0), ACC, VEL)
        print(f"Now, I am at: {mdl.get_EE_coords()}")
        print(f"Ground: {z_move+z_extra} which is z_move({z_move}) + z_extra({z_extra})")
        translate((0, 0, z_move+z_extra), ACC, VEL)

    elif name == "Foundation":
        y_extra = 0.00125  # Bigger value in -ve moves it up (w.r.t foundation from top camera view)
        x_extra = 0.006  # Bigger value in -ve moves it right (w.r.t foundation from top camera view)
        z_extra = -0.205   #Bigger value in -ve moves it up.
        # print(f"I am at: {mdl.get_EE_coords()}")
    #If there will be no other part then do this:
    if len(IFC_sorted[:,0]) <= n_placed+1:
        z_move = z_move * 0.98  #dont question this.

    if name == 'Foundation':
        print("Placing toilet on Foundation")
        set_orientation(orientations['ct'],ACC, VEL)
        translate((x_extra, y_extra, z_move/2), ACC, VEL)
        # print(f"I am at: {mdl.get_EE_coords()}")
        translate((0, 0, (z_move/2)+z_extra), ACC, VEL)

    gripper_width(100)
    time.sleep(0.5)

    placed_pos = {
        "coords": mdl.get_EE_coords(),
        "orientation": mdl.get_orientation()
        }
    translate((0, 0, -0.04), ACC, VEL)
    return placed_pos

def handle_toilet_placement(IFC_sorted, n_placed, frame_handler, ct_pos, st_pos):
    """
    High-level routine to place a bathroom module with optional user input (used in older workflow).

    **Process:**
    1. Moves robot to storage zone (`st_pos`).
    2. Uses `check_frame` to see if red or blue modules are visible.
    3. Prompts the user (with a timeout) to choose which color to place (`r` or `b`). If user doesn't respond, it auto-selects a color that is present.
    4. Calls `grab_toilet` with the chosen color thresholds to pick up the module.
    5. Moves to construction site (`ct_pos`) and calls `place_toilet(IFC_sorted, n_placed)` to place it.
    6. Returns the poses from grab and place for logging.

    **Returns:** 
    - *(toilet_grab_pos, toilet_place_pos)* – The end-effector poses after grabbing and after placing the module, or `None` if no module could be placed.
    """
    moveJ(st_pos, ACC, VEL)
    red_found, blue_found = check_frame(bathroom_module_color_thresholds, bathroom_module_area_thresholds, frame_handler)
    print("Red found =", red_found, "Blue found =", blue_found)
    
    answer = None
    try:
        
        answer = func_timeout.func_timeout(5, lambda: input('Input color of toilet to place [r/b]:\n'))
    except Exception:
        print("Input timeout; choosing default based on detection")
    
    if answer not in ["r", "b"]:
        answer = "r" if red_found else "b" if blue_found else None
    
    if answer == "r" and red_found:
        toilet_grab_pos = grab_toilet(bathroom_module_color_thresholds["red"]["LB"],
                                      bathroom_module_color_thresholds["red"]["UB"],
                                      frame_handler)
    elif answer == "b" and blue_found:
        toilet_grab_pos = grab_toilet(bathroom_module_color_thresholds["blue"]["LB"],
                                      bathroom_module_color_thresholds["blue"]["UB"],
                                      frame_handler)
    else:
        print("No suitable toilet block detected.")
        return None
    
    moveJ(ct_pos, ACC, VEL)
    toilet_place_pos = place_toilet(IFC_sorted, n_placed)
    return (toilet_grab_pos, toilet_place_pos)

def ask_for_color(_board, _framehandler, idx, timeout=5):
    """
    Prompt the user (via the UI board) to select a color for the bathroom module.

    This function is part of the interactive workflow to decide whether to place a red or blue module.

    **Parameters:**
    - *_board*: UI board object that can present a question and get a response (e.g., a Jupyter frontend).
    - *_framehandler*: FrameHandler for checking current camera view for available colors.
    - *idx (int)* – Index of the bathroom module in sequence (1, 2, or 3).
    - *timeout (int)* – Seconds to wait for user response before auto-selecting.

    **Returns:**
    - *str or None* – 'r' or 'b' if a choice is made or detected, None if no module is available.

    **Behavior:**
    - Updates the `construction_status` state to indicate a bathroom module search (used by UI to display current action).
    - Calls `check_frame` to see which colors are present.
    - If none present, returns None (cannot place any).
    - If only one color present, auto-selects it (and prints a message).
    - If both present, uses `_board.ask` to prompt the user within the given timeout. If user fails to respond in time or input is invalid, randomly picks one of the available colors.
    - Normalizes possible inputs (allowing "red"/"blue" or 'r'/'b').
    - Returns the chosen color as 'r' or 'b'.

    **Role:**
    This function provides a safe interactive step ensuring the *worker (user) remains in the loop* for choosing module color when both are available. It reflects the system’s flexibility in either fully automating decisions or deferring to a human operator.
    """
    elements = f'bathroom_module_{idx}'
    activity= 'search'

    with construction_status.state_lock:
        construction_status.state["current_element"] = elements
        construction_status.state["current_state"] = activity
    # Unpack color detection results
    red_found, blue_found = check_frame(
        bathroom_module_color_thresholds, 
        bathroom_module_area_thresholds, 
        _framehandler
        )
    
    print("Red found =", red_found)
    print("Blue found =", blue_found)

    # Only allow colors that exist in the frame
    available_colors = []
    if red_found:
        available_colors.append('r')
    if blue_found:
        available_colors.append('b')

    # If neither color is found, no decision can be made
    if not available_colors:
        print("No red or blue color found in the frame. Cannot place any toilet.")
        return None
    
    if len(available_colors) == 1:
        chosen = available_colors[0]
        print(f"Only '{chosen}' detected; automatically selecting {chosen}.")
        return chosen

    else:
        # Ask user to pick a color
        answer = None

        try:
            prompt_text = f"Input color of bathroom module you want placed {available_colors}:"
            answer = _board.ask(prompt_text, tuple(available_colors), timeout)
        #     answer = func_timeout(timeout, lambda: input(f"Input color of toilet you want placed {available_colors}:\n")).strip().lower()
        # except FunctionTimedOut:
        except Exception:
            print("Too slow, I will just pick a color for you!")
            answer = None

        # Validate user input
        if answer in available_colors:
            pass  # valid input already
        elif answer in ("red", "Red"):
            answer = "r"
        elif answer in ("blue", "Blue"):
            answer = "b"
        else:
            answer = random.choice(available_colors)
            print(f"Auto-selected color: {answer}")
        return answer

def pick_bathroom_module(color, idx,_framehandler):
    elements = f'bathroom_module_{idx}'
    activity= 'pick'

    toilet_pos_grab = None

    with construction_status.state_lock:
        construction_status.state["current_element"] = elements
        construction_status.state["current_state"] = activity

    if color == 'r':
        toilet_pos_grab = grab_toilet(bathroom_module_color_thresholds["red"]["LB"], 
        bathroom_module_color_thresholds["red"]["UB"], _framehandler)

    elif color == 'b':
        toilet_pos_grab = grab_toilet(bathroom_module_color_thresholds["blue"]["LB"], 
            bathroom_module_color_thresholds["blue"]["UB"],_framehandler)
    return toilet_pos_grab

def place_bathroom_module(IFC_sorted, idx):
    elements = f'bathroom_module_{idx}'
    activity= 'place'

    with construction_status.state_lock:
        construction_status.state["current_element"] = elements
        construction_status.state["current_state"] = activity

    if idx == 1:
        n_placed = 0
    if idx == 2:
        n_placed = 2
    if idx == 3:
        n_placed = 4

    toilet_pos = place_toilet(IFC_sorted, n_placed)
    return toilet_pos

def routine_for_bathroom(
        _board,
        _framehandler, 
        idx, 
        place_pos_bathroom, 
        _IFC_sorted):
    print("We should now place a colored toilet!")
    answer = ask_for_color(_board, _framehandler,idx)
    if answer == None:
        return None,None
    time.sleep(0.5)
    toilet_pos_grab = pick_bathroom_module(answer, idx,_framehandler)
    time.sleep(0.5)
    swing(place_pos_bathroom,f"bathroom_module_{idx}","swing")
    toilet_pos = place_bathroom_module( _IFC_sorted, idx)
    # swing(pickup_pos_element,f"bathroom_module_{idx}","swing_back")
    return toilet_pos_grab, toilet_pos