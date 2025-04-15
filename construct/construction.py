# construct/construction.py
import time
from datetime import datetime
from func_timeout import FunctionTimedOut, func_timeout
import pickle

import pandas as pd
# import cv2

from math import pi

from collections import defaultdict, Counter

from system_config import *
from construction_config import *

from ui.mobility import moveJ, translate,set_orientation, gripper_width
import ui.MarkerDetectionLocalization as mdl

from .marker_pose import find_part, align_with_random_part
from detections.marker_detector import MarkerDetector
from .bathroom import *

from . import construction_status

def scan_site(frame_handler, use_stub=False):
    """
    Scans the construction site and returns key positions:
      - pu_pos: Pickup position (for picking parts)
      - ct_pos: Construction site position (for placement)
      - st_pos: Storage position
    If use_stub is True, positions are loaded from a file.
    """
    if not use_stub:
        print('Performing site scan')
        moveJ(camera_pos1, ACC, VEL)

        z_offset_value = 0.105
        # Determine Pickup Position using the storage marker:
        pu_pos = find_part(MARKER_DICT["storage"], frame_handler)
        
        translate((0.15, 0.1, z_offset_value), ACC, VEL)
        pu_pos = mdl.get_joints()
        
        # Determine Construction Site Position:
        moveJ(camera_pos2, ACC, VEL)
        ct_pos = find_part(MARKER_DICT["construction"], frame_handler)
        translate((0.075, 0.025, 0), ACC, VEL)
        ct_pos = mdl.get_joints()
        
        # Determine Storage Position:
        moveJ(drop_pos, ACC, VEL)
        # translate((0.1, 0, 0), ACC, VEL)
        _ = find_part(MARKER_DICT["pick_up"], frame_handler)
        st_pos = mdl.get_joints()
        
        moveJ(pu_pos, ACC, VEL)
    else:
        print('Using stub file for site scan')
        path = "/home/avi/Desktop/robomason/_workingdata/_siteinfo/saved_positions.pkl"
        with open(path, "rb") as f:
            pu_pos, ct_pos, st_pos = pickle.load(f)
    return pu_pos, ct_pos, st_pos

def update_site_positions(_framehandler,path = "/home/avi/Desktop/robomason/_workingdata/_siteinfo/saved_positions.pkl"):
    pu_pos, ct_pos, st_pos = scan_site(_framehandler,False)

    with open(path, "wb") as f:
        pickle.dump((pu_pos, ct_pos, st_pos), f)
    print("Positions saved successfully.")

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

def grab(pu_pos, part_id,_framehandler):
#Inputs are:
#pu_pos is the pickup posistion
#part_id is the fiducialmarker id that it's looking for!

    #move to pick pos
    moveJ(pu_pos, ACC, VEL)
    set_orientation(orientations['pu'],ACC, VEL)

    find_part(part_id, _framehandler)   #this centers over the foundation
    #open gripper
    gripper_width(100)

    # If part_if 11, then we are looking for the foundation!
    if part_id == 11:
        #approach and pick up part!
        z_offset = mdl.get_EE_coords()[2] + pickup_offsets["foundation"]["z"]  #mdl.get_EE_coords()[2] is the current z posistion of the tool
        translate((pickup_offsets["foundation"]["x"], pickup_offsets["foundation"]["y"], 0), ACC, VEL)
        translate((0, 0, z_offset), ACC, VEL)
        gripper_width(70)
        gripper_width(60)
        time.sleep(2.0)
        translate((0, 0, -z_offset), ACC, VEL)
        
    elif part_id == 10:   #For the wall
        z_offset = mdl.get_EE_coords()[2] + pickup_offsets["wall"]["z"] 
        translate((pickup_offsets["wall"]["x"], pickup_offsets["wall"]["y"], 0), ACC, VEL)
        translate((0, 0, z_offset), ACC, VEL)
        time.sleep(0.25)
        gripper_width(95)
        time.sleep(1.25)
        translate((0, 0, -z_offset), ACC, VEL)
    
    elif part_id == 12:   #For the floor
        #Center over marker
        #approach and pick up part!
        z_offset = mdl.get_EE_coords()[2] + pickup_offsets["floor"]["z"]
        translate((pickup_offsets["floor"]["x"], pickup_offsets["floor"]["y"], 0), ACC, VEL)
        translate((0, 0, z_offset), ACC, VEL)
        gripper_width(50)
        gripper_width(20) #adjusted floor
        time.sleep(2.0)
        translate((0, 0, -z_offset), ACC, VEL)

    else:
        pass

# --- PICK Routine for Non-Toilet Parts ---
def pick(element, pu_pos, frame_handler):
    """
    Executes the pick routine for non-toilet parts.
    Determines the correct marker ID based on element name and performs the pick sequence.
    """
    if "floor" in element.lower():
        element_key = "Floor"
    elif "wall" in element.lower():
        element_key = "Wall"
    elif "foundation" in element.lower():
        element_key = "Foundation"
    else:
        raise ValueError(f"Invalid element: {element}")

    part_id = MARKER_DICT.get(element_key)
    if part_id is None:
        raise ValueError(f"Element '{element}' not found in marker_dict")
    
    grab(pu_pos, part_id, frame_handler)


# --- PLACE Routine for Non-Toilet Parts ---
def place(ct_pos, place_coords, part_id,floorpartid):
    """
    Executes the placement routine for non-toilet parts.
    Uses offset values defined in the configuration.
    """
    
    moveJ(ct_pos, ACC, VEL)
    
    x_move = float(place_coords[0]) / 100
    y_move = float(place_coords[1]) / 100
    z_move = mdl.get_EE_coords()[2] + pickup_offsets["foundation"]["z"] - float(place_coords[2]) / 100
    
    if part_id == 11:
        translate((x_move + x_offset_fund_place, y_move + y_offset_fund_place, 0), ACC, VEL)
        set_orientation(orientations['ct'], ACC, VEL)

        #Rotates 90 deg. 
        current_posj = mdl.get_joints()
        current_posj[5] += pi/2 - 0.005
        moveJ(current_posj, ACC, VEL)

        translate((0, 0, z_move + z_offset_fund_place), ACC, VEL)
        gripper_width(100)
        time.sleep(0.5)

    elif part_id == 10:
        translate((x_move + x_offset_wall_place, y_move + y_offset_wall_place, 0), ACC, VEL)
        set_orientation(orientations['ct'], ACC, VEL)
        if floorpartid == 3:
            #Rotates 90 deg.
            temp_pos =mdl.get_joints()
            temp_pos[5] = temp_pos[5]+wall_place_rotation_finetune
            moveJ(temp_pos, ACC, VEL)

            # translate((0, 0, z_move + z_offset_wall_place  ), ACC, VEL) #Moves in z  
            translate((x_offset_wall_place_finetune_2, y_offset_wall_place_finetune_2, 
                       z_move + z_offset_wall_place - z_offset_wall_place_finetune_2), ACC, VEL)
        else:
            translate((x_offset_wall_place_finetune_1, y_offset_wall_place_finetune_1, 
                       z_move + z_offset_wall_place - z_offset_wall_place_finetune_1  ), ACC, VEL) #Moves in z        
            translate((0,0,0.004), ACC, VEL)
            translate((0,0,0.004), ACC, VEL)

        gripper_width(100)
        time.sleep(0.5)

    elif part_id == 12:   #For the floor                
        translate(( x_move + x_offset_floor_place,  y_move + y_offset_floor_place, 0), ACC, VEL)
        set_orientation(orientations['ct'],ACC, VEL)                  

        if floorpartid == 2:
            temp_pos =mdl.get_joints()
            temp_pos[5] = temp_pos[5] + floor_place_rotation_1
            moveJ(temp_pos, ACC, VEL)
    
            translate((x_offset_floor_place_finetune_1, y_offset_floor_place_finetune_1,
                       z_offset_wall_place_finetune_1), ACC, VEL)      

            translate((0, 0, z_move + z_offset_floor_place), ACC, VEL)     
            
        elif floorpartid == 4:
            temp_pos =mdl.get_joints()
            temp_pos[5] = temp_pos[5] + floor_place_rotation_2
            moveJ(temp_pos, ACC, VEL)  

            print('second floor')

            # translate((0, 0, z_offset_floor_place_finetune_2), ACC, VEL)

            translate((x_offset_floor_place_finetune_2, y_offset_floor_place_finetune_2,
                        z_offset_floor_place + z_move + z_offset_floor_place_finetune_2), ACC, VEL)
                    
        gripper_width(100)
        time.sleep(0.5)

    else:
        pass
    
    placed_pos = {
        "coords": mdl.get_EE_coords(),
        "orientation": mdl.get_orientation()
    }
    
    translate((0, 0, -0.05), ACC, VEL)  #Moves a bit up

    return placed_pos

def pick_random_element(_part_id, _length_mark, _x_offset, _y_offset,_z_level_, _gripper_width_, _framehandler):
    clarity_shift = 0.025
    #go to pickup location
    pu_pos, ct_pos, st_pos = scan_site(_framehandler,True)
    pickup_loaction = pu_pos
    moveJ(pickup_loaction, ACC, VEL)
    translate((0,0,clarity_shift),ACC, VEL) # Move down a bit to read marker better.
    #Orient
    set_orientation(orientations['pu'],ACC, VEL)
    #find object
    align_with_random_part(_part_id, _length_mark, _framehandler)   #this centers over the foundation
    gripper_width(100)
    #Align 
    translate((_x_offset,_y_offset,0),ACC,VEL)
    #Go down
    c = mdl.get_EE_coords()
    c[2] = _z_level_
    o = mdl.get_orientation()
    rotation_vector = rotation_matrix_to_vector(np.array(o))
    c.extend(rotation_vector.tolist())
    #Go-to new position
    moveL(c, ACC, VEL)
    #Close_gripper
    gripper_width(_gripper_width_)
    time.sleep(2)
    #Move up
    translate((0,0,-0.06),ACC, VEL)

def place_random_element(_z_level_, _rotation, _framehandler):
    construction_point_x = 0.15
    construction_point_y = 0.025

    pu_pos, ct_pos, st_pos = scan_site(_framehandler,True)
    moveJ(ct_pos, ACC, VEL)
    set_orientation(orientations['ct'], ACC, VEL)

    translate((construction_point_x,construction_point_y,0), ACC, VEL)

    #Rotates the gripper. 
    current_posj = mdl.get_joints()
    # current_posj[5] += pi/2 - 0.005
    current_posj[5] += _rotation
    moveJ(current_posj, ACC, VEL)

    #go down
    c = mdl.get_EE_coords()
    c[2] = _z_level_
    o = mdl.get_orientation()
    rotation_vector = rotation_matrix_to_vector(np.array(o))
    c.extend(rotation_vector.tolist())
    #Go-to new position
    moveL(c, ACC, VEL)
    gripper_width(100)
    translate((0,0,-0.06),ACC, VEL)

def check_continue():
    file_path = '/home/avi/Desktop/robomason/_workingdata/_siteinfo/_Construction_state/_stage.txt'
    with open(file_path, 'r') as file:
        index = int(file.read())
    return index

def update_continue(param):
    # Path to the file
    file_path = '/home/avi/Desktop/robomason/_workingdata/_siteinfo/_Construction_state/_stage.txt'
    index = str(param)
    with open(file_path, 'w') as file:
        file.write(index)


def swing(_pos, item,activity):
    elements = item
    # activity= 'swing' or 'swing_back'

    with construction_status.state_lock:
        construction_status.state["current_element"] = elements
        construction_status.state["current_state"] = activity
        
    moveJ(_pos, ACC, VEL)

def search_element(item,_framehandler, pos = 'pu', marker_dict = MARKER_DICT):
    elements = item
    activity= 'search'
    print(f"Searching for {item}") 

    if item != "Foundation":
        item = item.split("_")[0]

    with construction_status.state_lock:
        construction_status.state["current_element"] = elements
        construction_status.state["current_state"] = activity

    img = _framehandler.get_latest_frame()
    _, ids = MarkerDetector.find_objects(img)

    if (ids is not None and marker_dict.get(item) in ids):
        img = _framehandler.get_latest_frame()
        _, ids = MarkerDetector.find_objects(img)
        print("The item is here:")
    set_orientation(orientations[pos],ACC, VEL)

    find_part(marker_dict.get(item), _framehandler)   #this centers over the foundation
    #open gripper
    gripper_width(100)

def pick_element(item, marker_dict = MARKER_DICT):
    elements = item
    activity= 'pick'

    if item != "Foundation":
        item = item.split("_")[0]

    with construction_status.state_lock:
        construction_status.state["current_element"] = elements
        construction_status.state["current_state"] = activity

    if marker_dict.get(item) == 11:
        #approach and pick up part!
        z_offset = mdl.get_EE_coords()[2] + pickup_offsets["foundation"]["z"]  #mdl.get_EE_coords()[2] is the current z posistion of the tool
        translate((pickup_offsets["foundation"]["x"], pickup_offsets["foundation"]["y"], 0), ACC, VEL)
        translate((0, 0, z_offset), ACC, VEL)
        gripper_width(70)
        gripper_width(60)
        time.sleep(2.0)
        translate((0, 0, -z_offset), ACC, VEL)
        
    elif marker_dict.get(item) == 10:   #For the wall
        z_offset = mdl.get_EE_coords()[2] + pickup_offsets["wall"]["z"] 
        translate((pickup_offsets["wall"]["x"], pickup_offsets["wall"]["y"], 0), ACC, VEL)
        translate((0, 0, z_offset), ACC, VEL)
        time.sleep(0.25)
        gripper_width(95)
        time.sleep(1.25)
        translate((0, 0, -z_offset), ACC, VEL)
    
    elif marker_dict.get(item) == 12:   #For the floor
        #Center over marker
        #approach and pick up part!
        z_offset = mdl.get_EE_coords()[2] + pickup_offsets["floor"]["z"]
        translate((pickup_offsets["floor"]["x"], pickup_offsets["floor"]["y"], 0), ACC, VEL)
        translate((0, 0, z_offset), ACC, VEL)
        gripper_width(50)
        gripper_width(20) #adjusted floor
        time.sleep(2.0)
        translate((0, 0, -z_offset), ACC, VEL)

    else:
        pass

def place_element(item, floorpartid, place_coords, marker_dict = MARKER_DICT):
    elements = item
    activity= 'place'

    if item != "Foundation":
        item = item.split("_")[0]

    with construction_status.state_lock:
        construction_status.state["current_element"] = elements
        construction_status.state["current_state"] = activity

    x_move = float(place_coords[0]) / 100
    y_move = float(place_coords[1]) / 100
    z_move = mdl.get_EE_coords()[2] + pickup_offsets["foundation"]["z"] - float(place_coords[2]) / 100
    
    if marker_dict.get(item) == 11:
        translate((x_move + x_offset_fund_place, y_move + y_offset_fund_place, 0), ACC, VEL)
        set_orientation(orientations['ct'], ACC, VEL)

        #Rotates 90 deg. 
        current_posj = mdl.get_joints()
        current_posj[5] += pi/2 - 0.005
        moveJ(current_posj, ACC, VEL)

        translate((0, 0, z_move + z_offset_fund_place), ACC, VEL)
        gripper_width(100)
        time.sleep(0.5)

    elif marker_dict.get(item) == 10:
        translate((x_move + x_offset_wall_place, y_move + y_offset_wall_place, 0), ACC, VEL)
        set_orientation(orientations['ct'], ACC, VEL)
        if floorpartid == 3:
            #Rotates 90 deg.
            temp_pos =mdl.get_joints()
            temp_pos[5] = temp_pos[5]+wall_place_rotation_finetune
            moveJ(temp_pos, ACC, VEL)

            # translate((0, 0, z_move + z_offset_wall_place  ), ACC, VEL) #Moves in z  
            translate((x_offset_wall_place_finetune_2, y_offset_wall_place_finetune_2, 
                       z_move + z_offset_wall_place - z_offset_wall_place_finetune_2), ACC, VEL)
        else:
            translate((x_offset_wall_place_finetune_1, y_offset_wall_place_finetune_1, 
                       z_move + z_offset_wall_place - z_offset_wall_place_finetune_1  ), ACC, VEL) #Moves in z        
            translate((0,0,0.004), ACC, VEL)
            translate((0,0,0.004), ACC, VEL)

        gripper_width(100)
        time.sleep(0.5)

    elif marker_dict.get(item) == 12:   #For the floor                
        translate(( x_move + x_offset_floor_place,  y_move + y_offset_floor_place, 0), ACC, VEL)
        set_orientation(orientations['ct'],ACC, VEL)                  

        if floorpartid == 2:
            temp_pos =mdl.get_joints()
            temp_pos[5] = temp_pos[5] + floor_place_rotation_1
            moveJ(temp_pos, ACC, VEL)
    
            translate((x_offset_floor_place_finetune_1, y_offset_floor_place_finetune_1,
                       z_offset_wall_place_finetune_1), ACC, VEL)      

            translate((0, 0, z_move + z_offset_floor_place), ACC, VEL)     
            
        elif floorpartid == 4:
            temp_pos =mdl.get_joints()
            temp_pos[5] = temp_pos[5] + floor_place_rotation_2
            moveJ(temp_pos, ACC, VEL)  

            print('second floor')

            # translate((0, 0, z_offset_floor_place_finetune_2), ACC, VEL)

            translate((x_offset_floor_place_finetune_2, y_offset_floor_place_finetune_2,
                        z_offset_floor_place + z_move + z_offset_floor_place_finetune_2), ACC, VEL)
                    
        gripper_width(100)
        time.sleep(0.5)

    else:
        pass
    
    placed_pos = {
        "coords": mdl.get_EE_coords(),
        "orientation": mdl.get_orientation()
    }
    
    translate((0, 0, -0.05), ACC, VEL)  #Moves a bit up

    return placed_pos

def perform_construction(IFC_sorted, frame_handler, n_placed, use_stub=False, marker_dict=MARKER_DICT):
    """
    Performs construction based on the sorted IFC data.
    
    The IFC_sorted data is assumed to be an array (or list of lists) where each row
    is [item_name, x, y, z]. For example, the logical ordering is:
       Foundation, bathroom_module_1, Wall_1, Floor_1, bathroom_module_2, Wall_2, Floor_2, bathroom_module_3
       
    The routine:
      1. Scans the environment to get pickup (pu_pos), construction (ct_pos), and storage (st_pos) positions.
      2. Iterates through the ordered list:
           - For a standard item (not beginning with "bathroom_module"):
                 • Update state to "search" and call search_element.
                 • Swing to pickup (using pu_pos) then call pick_element.
                 • Swing to construction site (ct_pos).
                 • Set placement coordinates from the current IFC_sorted row (using ifc_index) and call place_element.
                 • Swing back (using pu_pos or st_pos depending on the next item).
                 • Record the placed position and increment ifc_index.
           - For an item that starts with "bathroom_module":
                 • Use the placement coordinate from the previous row (if available) and call routine_for_bathroom.
                 • Record the current end-effector position.
      3. Updates its state at each step.
      
    Returns:
       A tuple: (item_list, block_list, placed_positions)
         - item_list: list of processed item names.
         - block_list: any extra data returned from toilet routines (if applicable).
         - placed_positions: list of positions (dicts with 'coords' and 'orientation')
    """   
    # ----- Build ordered list from IFC_sorted -----
    raw_items = []
    bathroom_count = 1
    for row in IFC_sorted:
        # Extract the item name; assuming format "something: item - something"
        item = row[0].split(":")[1].split("-")[0].strip()
        raw_items.append(item)
        # For every "Foundation" or "Floor", insert an extra "bathroom_module" entry.
        if item in ["Foundation", "Floor"]:
            bathroom_count += 1
            raw_items.append(f"bathroom_module_{bathroom_count}")
    
    # Optionally, if you need numbering for duplicates.
    occ = Counter(raw_items)
    cnt = defaultdict(int)
    ordered_items = []
    for itm in raw_items:
        if occ[itm] > 1:
            cnt[itm] += 1
            ordered_items.append(f"{itm}_{cnt[itm]}")
        else:
            ordered_items.append(itm)
    
    # ----- Initialize result lists and indices -----
    item_list = []         # List of items processed.
    block_list = []        # For extra data (e.g. toilet block information).
    placed_positions = []  # To store the end-effector positions after placement.
    
    # 'ifc_index' will track how many IFC_sorted rows have been consumed for placement coordinates.
    ifc_index = 0
    # n_placed = 0 # 'n_placed' counts the overall items from the ordered_items list.
    bathroom_idx = 1 #'bathroom_idx' keeps the current count of the bathroom modules

    # ----- Scan the environment -----
    # Update state.
    with construction_status.state_lock:
        construction_status.state["current_element"] = "scanning_site"
        construction_status.state["current_state"] = "-"
    pu_pos, ct_pos, st_pos = scan_site(frame_handler, use_stub=use_stub)
    # The starting position for search is taken as pu_pos (pickup)
    start_pos = pu_pos
    moveJ(start_pos, ACC, VEL)

    # ----- Process each item in the ordered list -----
    while n_placed < len(ordered_items):
        current_item = ordered_items[n_placed]
                
        # Branch if this is a bathroom module entry.
        if current_item.startswith("bathroom_module"):
            # Use the placement coordinate from the previous IFC row.
            if ifc_index == 0:
                # If no previous row exists, default to a safe coordinate (or raise error)
                place_coords = [0, 0, 0]
            else:
                place_coords = IFC_sorted[ifc_index - 1][1:4]
            print("Initiating bathroom routine for associated element.")
            # Call the dedicated bathroom routine.
            from .bathroom import routine_for_bathroom #Do this to avoid circular imports.
            grab_pos,placed_pos = routine_for_bathroom(frame_handler,
                                    bathroom_idx,
                                    place_pos_bathroom=ct_pos,    # use construction position
                                    pickup_pos_element=pu_pos,     # use pickup position
                                    _IFC_sorted=IFC_sorted,
                                    n_placed=ifc_index - 1)
            block_list.append(grab_pos)
            bathroom_idx += 1

            # For swing-back, choose pu_pos.
            swing_target = pu_pos
            
        else:
            # Standard item – follow the sequence.
            # 1. Search: 
            search_element(current_item, frame_handler, marker_dict=marker_dict)

            # 2. Pick the item.
            pick_element(current_item, marker_dict = marker_dict)
            
            # 3. Swing to the construction (placement) site.
            swing(ct_pos, current_item, "swing")
            
            # 4. Set placement coordinates from the .IFC and place the item 
            if ifc_index < len(IFC_sorted):
                # Extract coordinates from the current row.
                place_coords = IFC_sorted[ifc_index][1:4]
            else:
                # If we run out of rows, use ct_pos or a safe default.
                place_coords = [0, 0, 0]
   
            placed_pos = place_element(current_item, ifc_index, place_coords, marker_dict = marker_dict)

            item_list.append(current_item)
            ifc_index += 1  # Increment the IFC coordinate index (only for standard items).
            
            # Here we choose st_pos if the next item (if any) starts with "bathroom_module"; otherwise, return to pu_pos.
            if n_placed + 1 < len(ordered_items) and ordered_items[n_placed + 1].startswith("bathroom_module"):
                swing_target = st_pos
            else:
                swing_target = pu_pos
            swing(swing_target, current_item, "swing_back")
        
     
        placed_positions.append(placed_pos)
  
        time.sleep(0.5)
        update_continue(n_placed)
        n_placed += 1
    
    print("Construction process completed!")
    return item_list, block_list, placed_positions


def perform_reconstruction(IFC_sorted, frame_handler, n_placed, _dis_pos = dis_pos, marker_dict=MARKER_DICT):
    """
    Performs the reconstruction (reassembly) routine using the sorted IFC data.
    
    IFC_sorted is assumed to be an array-like object where each row is:
         [ "something: item - something", x, y, z ]
    For example, the logical ordering might be:
         Foundation, bathroom_module_1, Wall_1, Floor_1, bathroom_module_2, Wall_2, Floor_2, bathroom_module_3
         
    In reconstruction:
      - The start (pickup) position is taken from the disassembly position (dis_pos).
      - The construction (placement) and storage positions are obtained by scanning the site (with use_stub=True).
      
    The routine processes the ordered items as follows:
      (A) If the item's base name is "Foundation":  
          • Skip normal processing (i.e. do NOT perform search–pick–place)  
          • Print a message that Foundation is skipped so that the following bathroom_module_1 will be used.
          
      (B) If the ordered item starts with "bathroom_module":  
          • Use the placement coordinates from the previous IFC row (if available)  
          • Call routine_for_bathroom(...) to handle the toilet placement  
          • Record the returned end-effector position.
          
      (C) Otherwise, for standard items (e.g. Wall or Floor):  
          1. Update state to "search" and call search_element(current_item, frame_handler, marker_dict).
          2. Swing from disassembly (start_pos) to the pick area.
          3. Update state to "pick" and call pick_element(current_item, marker_dict).
          4. Swing to the construction (placement) site.
          5. Set placement coordinates from the current IFC_sorted row (using ifc_index) and call place_element(current_item, ifc_index, place_coords, marker_dict).
          6. Determine a safe swing-back target (storage if the next item is a bathroom module, else disassembly) and swing back.
          7. Record the placed position and increment ifc_index.
          
    The function updates its state at each step.
    
    Returns:
       A tuple: (item_list, block_list, placed_positions)
         - item_list: list of processed item names (excluding skipped Foundation).
         - block_list: any extra data from toilet routines (if applicable).
         - placed_positions: list of positions (each a dict with 'coords' and 'orientation').
    """

    # ----- Step 1. Build the Ordered List from IFC_sorted -----
    raw_items = []
    bathroom_count = 1
    for row in IFC_sorted:
        # Extract the item name; assuming format "something: item - something"
        item = row[0].split(":")[1].split("-")[0].strip()
        raw_items.append(item)
        # For every "Foundation" or "Floor", insert an extra "bathroom_module" entry.
        if item in ["Foundation", "Floor"]:
            bathroom_count += 1
            raw_items.append(f"bathroom_module_{bathroom_count}")
    
    # Create a numbered ordered list (if duplicates exist)
    occ = Counter(raw_items)
    cnt = defaultdict(int)
    ordered_items = []
    for itm in raw_items:
        if occ[itm] > 1:
            cnt[itm] += 1
            ordered_items.append(f"{itm}_{cnt[itm]}")
        else:
            ordered_items.append(itm)
    
  
    item_list = []         # Will contain the names of standard items processed.
    block_list = []        # For any extra toilet/block data.
    placed_positions = []  # To record the end-effector position after each placement.
    
    # ifc_index tracks which row in IFC_sorted has been used for placement coordinates.
    ifc_index = 0
    # n_placed = 0 # n_placed is the counter for overall items in the ordered_items list.
    bathroom_idx = 1 #'bathroom_idx' keeps the current count of the bathroom modules

    with construction_status.state_lock:
            construction_status.state["current_element"] = "scanning_site"
            construction_status.state["current_state"] = "-"

    _, ct_pos, st_pos = scan_site(frame_handler, use_stub=True)
   
    # start_pos = dis_pos  
    moveJ(_dis_pos, ACC, VEL)
    _dis_pos = find_part(MARKER_DICT["deconstruction"], frame_handler)

    translate((-0.07, 0.045, 0), ACC, VEL) # fine-tuning for better view.
    start_pos = mdl.get_joints()

    moveJ(st_pos, ACC, VEL)
    
    while n_placed < len(ordered_items):
        current_item = ordered_items[n_placed]

        # If the base item is Foundation, skip normal processing.
        if current_item == "Foundation":
            print("Skipping placement for Foundation.")
            n_placed += 1
            ifc_index += 1
            continue
        
        # If the item is a bathroom module:
        if current_item.startswith("bathroom_module"):
            # Use the placement coordinate from the previous IFC row.
            if ifc_index == 0:
                place_coords = [0, 0, 0]
            else:
                place_coords = IFC_sorted[ifc_index - 1][1:4]
            print("Initiating bathroom routine for associated element.")
            from .bathroom import routine_for_bathroom  # local import to avoid circular imports
            grab_pos, placed_pos = routine_for_bathroom(frame_handler,
                                    bathroom_idx,
                                    place_pos_bathroom=ct_pos,    # use construction site position
                                    pickup_pos_element=start_pos, # use disassembly (pickup) position
                                    _IFC_sorted=IFC_sorted,
                                    n_placed=ifc_index - 1)
            block_list.append(grab_pos)
            bathroom_idx += 1

            # For swing-back, choose disassembly position.
            swing_target = start_pos
            
        else:
            # Standard item – follow the sequence.
            # 1. SEARCH:
            search_element(current_item, frame_handler, marker_dict=marker_dict)
                        
            # 2. PICK:
            pick_element(current_item, marker_dict=marker_dict)

            # 3. SWING: swing from disassembly to pick area.
            swing(ct_pos, current_item, "swing")
            
            # 4. Set placement coordinates from the .IFC and place the item 
            if ifc_index < len(IFC_sorted):
                place_coords = IFC_sorted[ifc_index][1:4]
            else:
                place_coords = [0, 0, 0]

            placed_pos = place_element(current_item, ifc_index, place_coords, marker_dict=marker_dict)
            item_list.append(current_item)
            ifc_index += 1  # Increment the IFC coordinate index (only for standard items).
            
            if n_placed + 1 < len(ordered_items) and ordered_items[n_placed + 1].startswith("bathroom_module"):
                swing_target = st_pos
            else:
                swing_target = start_pos
            swing(swing_target, current_item, "swing_back")
        
        # Record the placed position.
        placed_positions.append(placed_pos)
        time.sleep(0.5)
        update_continue(n_placed)
        n_placed += 1
            
    print("Reconstruction process completed!")
    return item_list, block_list, placed_positions

def test_function():
    pass

def create_indexed_dict(item_list, pl_pos):
    new_item_list = []
    bathroom_count = 1
    # Define the conditions in a tuple for easy extension.
    conditions = ("foundation", "floor")
    
    for item in item_list:
        new_item_list.append(item)
        # Lowercase the item once and check for the conditions.
        lower_item = item.lower()
        if any(cond in lower_item for cond in conditions):
            new_item_list.append(f"bathroom_module_{bathroom_count}")
            bathroom_count += 1

    # Validate the lengths.
    if len(new_item_list) != len(pl_pos):
        raise ValueError("Length mismatch: new_item_list and pl_pos must be the same length")

    # Use enumerate and zip for a cleaner dictionary comprehension.
    indexed_dict = {
        index: {"name": name, "pos": pos}
        for index, (name, pos) in enumerate(zip(new_item_list, pl_pos))
    }
    return indexed_dict

def disassembly(_item_list, _pl_pos,_block_list, _framehandler, _dis_pos = dis_pos):
    indexed_dict = create_indexed_dict(_item_list, _pl_pos)
    i = len(_pl_pos)-1
    bm_idx = 0
    n_placed = 0 # For non-bathroom_module elements
    elements = 'scanning_site'
    activity = '-'

    _, _ct_pos, _st_pos = scan_site(_framehandler,True)

    with construction_status.state_lock:
        construction_status.state["current_element"] = elements
        construction_status.state["current_state"] = activity

    moveJ(_dis_pos, ACC, VEL)
    _dis_pos = find_part(MARKER_DICT["deconstruction"], _framehandler)

    moveJ(_ct_pos, ACC, VEL)

    while i > -1:
        item = indexed_dict[i]['name']
        pickup_position_data = indexed_dict[i]['pos']

        if item != "Foundation":
            gripper_width(100)

            if i != (len(_pl_pos)-1):
                swing(_ct_pos, item,"swing_back") # Go to construction site

            elements = item
            activity = 'pick'

            with construction_status.state_lock:
                construction_status.state["current_element"] = elements
                construction_status.state["current_state"] = activity

            # Go to location in construction site
            pos = pickup_position_data['coords'].copy()
            rotation_vector = rotation_matrix_to_vector(np.array(pickup_position_data['orientation']))
            pos.extend(rotation_vector.tolist())
            moveL(pos, ACC, VEL)
            time.sleep(0.75)

            if item.startswith("Floor"):
                translate((0, 0, 0.0055), ACC, VEL)

            # Close gripper
            for k in range(len(elements_gripper_width[item[:-2]])):
                if elements_gripper_width[item[:-2]][k] is not None:
                    gripper_width(elements_gripper_width[item[:-2]][k])
                    time.sleep(1)

            time.sleep(2.5)
            translate((0, 0, -0.09), ACC, VEL)


            if item.startswith("bathroom_module"):
                swing(_st_pos, item, "swing")  ###

                activity = 'place'
                with construction_status.state_lock:
                    construction_status.state["current_state"] = activity

                pos = _block_list[(len(_block_list)-1 -bm_idx)]['coords'].copy()
                rotation_vector = rotation_matrix_to_vector(np.array(_block_list[(len(_block_list)-1 -bm_idx)]['orientation'])) 
                pos.extend(rotation_vector.tolist())
                moveL(pos, ACC, VEL)
                bm_idx += 1
                
                time.sleep(1)
                gripper_width(100)
                time.sleep(2)
                translate((0, 0, -0.06), ACC, VEL)
            else:
                swing(_dis_pos, item, "swing")
                
                activity = 'place'
                with construction_status.state_lock:
                    construction_status.state["current_state"] = activity

                if item[:-2] == "Wall":  # Fine-tuning for wall
                    z_extra = 0.015
                    z_down = mdl.get_EE_coords()[2] + pickup_offsets["wall"]["z"] + z_extra
                elif item[:-2] == "Floor":  # Fine-tuning for floor
                    z_extra = -0.0013
                    z_down = mdl.get_EE_coords()[2] + pickup_offsets["floor"]["z"] + z_extra

                x_move = drops[n_placed][0]
                y_move = -drops[n_placed][1]

                translate((x_move, y_move, 0), ACC, VEL)
                translate((0, 0, z_down), ACC, VEL)
                # print(x_move, y_move, z_down)
                time.sleep(1)
                gripper_width(100)
                time.sleep(2)
                translate((0, 0, -0.06), ACC, VEL)
                n_placed += 1
            
        i -=1

