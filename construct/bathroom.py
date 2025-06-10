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
    """
    Checks the current frame for red and blue objects.
    Returns booleans: (red_found, blue_found)
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
    Grabs the toilet block using a feedback loop for successive detections.
    Returns the position after a successful grab.
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
    Places the toilet block using configuration-specific offsets.
    Returns the final placement position.
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

        current_posj = mdl.get_joints()
        current_posj[5] += pi/2
        moveJ(current_posj, ACC, VEL)

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
    Orchestrates the full toilet placement routine:
      1. Move to storage zone.
      2. Use color detection to choose a toilet block.
      3. Grab the toilet block.
      4. Move to construction site and place the toilet.
    Returns a tuple: (toilet_grab_pos, toilet_place_pos) or None on failure.
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