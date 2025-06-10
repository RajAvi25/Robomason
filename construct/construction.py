# construct/construction.py
import os
import time
from datetime import datetime
from func_timeout import FunctionTimedOut, func_timeout
import pickle

import pandas as pd
# import cv2

from math import pi

from collections import defaultdict, Counter

from configs.system_config import *
from configs.construction_config import *

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

        # Determine Pickup Position using the storage marker:
        pu_pos = find_part(MARKER_DICT["pick_up"], frame_handler)
        
        translate((-0.11, 0.1, 0.04), ACC, VEL)
        pu_pos = mdl.get_joints()
        
        # Determine Construction Site Position:
        moveJ(camera_pos2, ACC, VEL)
        ct_pos = find_part(MARKER_DICT["construction"], frame_handler)
        translate((0.075, 0.025, 0), ACC, VEL)
        ct_pos = mdl.get_joints()
        
        # Determine Storage Position:
        moveJ(drop_pos, ACC, VEL)
        st_pos = find_part(MARKER_DICT["storage"], frame_handler)
        translate((0.1,0.05, 0), ACC, VEL)
        st_pos = mdl.get_joints()

        update_site_positions(pu_pos, ct_pos, st_pos)
        
    else:
        print('Using stub file for site scan')
        path = "/home/avi/Desktop/robomason/_workingdata/_siteinfo/saved_positions.pkl"
        with open(path, "rb") as f:
            pu_pos, ct_pos, st_pos = pickle.load(f)

    moveJ(pu_pos, ACC, VEL)
    print("Finished scanning")

    return pu_pos, ct_pos, st_pos

def update_site_positions(pu_pos, ct_pos, st_pos, path = "/home/avi/Desktop/robomason/_workingdata/_siteinfo/saved_positions.pkl"):
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
        gripper_width(20) #adjusted floor    plt.show()

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

            # print('second floor')

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

def pick_random_element(_part_id, _length_mark, _x_offset, _y_offset,_z_offset, _gripper_width_, _framehandler):

    # #go to pickup location
    # pu_pos, ct_pos, st_pos = scan_site(_framehandler,True)
    # pickup_loaction = pu_pos
    # moveJ(pickup_loaction, ACC, VEL)
    #Orient
    set_orientation(orientations['pu'],ACC, VEL)
    #find object
    align_with_random_part(_part_id, _length_mark, _framehandler)   #this centers over the foundation
    gripper_width(100)
    #Align 
    translate((_x_offset,_y_offset,0),ACC,VEL)
    #Go down
    c = mdl.get_EE_coords()
    c[2] =  _z_offset
    o = mdl.get_orientation()
    rotation_vector = rotation_matrix_to_vector(np.array(o))
    c.extend(rotation_vector.tolist())
    #Go-to new position
    moveL(c, ACC, VEL)
    #Close_gripper
    gripper_width(_gripper_width_)
    time.sleep(2)
    #Move up
    translate((0,0,-0.1),ACC, VEL)

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

def swing(_pos, item,activity):
    elements = item
    # activity= 'swing' or 'swing_back'

    with construction_status.state_lock:
        construction_status.state["current_element"] = elements
        construction_status.state["current_state"] = activity
        
    moveJ(_pos, ACC, VEL)

def search_element(item,_framehandler, pos = 'pu', marker_dict = MARKER_DICT):
    clarity_shift = 0.035
    elements = item
    activity= 'search'
    print(f"Searching for {item}") 

    if item != "Foundation":
        item = item.split("_")[0]

    with construction_status.state_lock:
        construction_status.state["current_element"] = elements
        construction_status.state["current_state"] = activity

    if not item.startswith("Wall") :
        translate((0,0,clarity_shift),ACC, VEL) # Move down a bit to read marker better.

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

def pick_element(item, _framehandler, _pu_pos,  marker_dict = MARKER_DICT):
    elements = item
    activity= 'pick'

    if item != "Foundation":
        item = item.split("_")[0]

    with construction_status.state_lock:
        construction_status.state["current_element"] = elements
        construction_status.state["current_state"] = activity

    if item == "Foundation":
        pick_random_element(
            marker_dict.get(item),
            ELEMENT_MARKER_SIZE,
            pickup_offsets["foundation"]["x"], 
            pickup_offsets["foundation"]["y"],
            pickup_offsets["foundation"]["z"],
            50,
            _framehandler,
            )
    elif item.startswith("Wall"):
        # grab(_pu_pos, MARKER_DICT.get("Wall"), _framehandler)
        pick_random_element(
            marker_dict.get(item),
            ELEMENT_MARKER_SIZE,
            pickup_offsets["wall"]["x"], 
            pickup_offsets["wall"]["y"],
            pickup_offsets["wall"]["z"],
            70,
            _framehandler,
            )
    elif item.startswith("Floor"):
        # grab(_pu_pos, MARKER_DICT.get("Wall"), _framehandler)
        pick_random_element(
            marker_dict.get(item),
            ELEMENT_MARKER_SIZE,
            pickup_offsets["floor"]["x"], 
            pickup_offsets["floor"]["y"],
            pickup_offsets["floor"]["z"],
            50,
            _framehandler,
            )          
    else:
        print("Invalid item (Error)")       


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
        current_posj[5] += pi/2 - 0.0125
        moveJ(current_posj, ACC, VEL)
        # translate((0, 0, z_offset_fund_place), ACC, VEL)
        #Go down
        c = mdl.get_EE_coords()
        c[2] =  z_offset_fund_place
        o = mdl.get_orientation()
        rotation_vector = rotation_matrix_to_vector(np.array(o))
        c.extend(rotation_vector.tolist())
        moveL(c, ACC, VEL)

        gripper_width(100)
        time.sleep(0.5)

    elif marker_dict.get(item) == 10:
        translate((x_move + x_offset_wall_place, y_move + y_offset_wall_place, 0), ACC, VEL)
        set_orientation(orientations['ct'], ACC, VEL)
        if floorpartid == 3:
            # print(f"I am at (Wall_2): {mdl.get_EE_coords()}")  
            translate(
                (x_offset_wall_place_finetune_2, 
                 y_offset_wall_place_finetune_2,
                 0 
                ), ACC, VEL)
            # print(f"Now, I am at (Wall_2): {mdl.get_EE_coords()}")
            #Rotates to adjust
            temp_pos =mdl.get_joints()
            temp_pos[5] = temp_pos[5]-wall_place_rotation_finetune
            moveJ(temp_pos, ACC, VEL)

            c = mdl.get_EE_coords()
            c[2] =  + z_offset_wall_place + z_offset_wall_place_finetune_2
            # print(f"ground: {+ z_offset_wall_place + z_offset_wall_place_finetune_2}")
            o = mdl.get_orientation()
            rotation_vector = rotation_matrix_to_vector(np.array(o))
            c.extend(rotation_vector.tolist())
            moveL(c, ACC, VEL)

            gripper_width(100)
            time.sleep(0.5)

        else:            
            translate(
                (x_offset_wall_place_finetune_1,
                 y_offset_wall_place_finetune_1,
                 0
                 ), ACC, VEL) 
            
            temp_pos =mdl.get_joints()
            temp_pos[5] = temp_pos[5]-wall_place_rotation_finetune
            moveJ(temp_pos, ACC, VEL)
                          
            c = mdl.get_EE_coords()
            c[2] =  + z_offset_wall_place + z_offset_wall_place_finetune_1
            o = mdl.get_orientation()
            rotation_vector = rotation_matrix_to_vector(np.array(o))
            c.extend(rotation_vector.tolist())
            moveL(c, ACC, VEL)

            gripper_width(100)
            time.sleep(0.5)

    elif marker_dict.get(item) == 12:   #For the floor                
        set_orientation(orientations['ct'],ACC, VEL)
        translate(
            (x_move + x_offset_floor_place,  
             y_move + y_offset_floor_place, 
             0
             ), ACC, VEL)                  

        if floorpartid == 2:
            temp_pos =mdl.get_joints()
            temp_pos[5] = temp_pos[5] - floor_place_rotation_1
            moveJ(temp_pos, ACC, VEL)

            # translate((0.05,0,0),ACC,VEL)
            # print(f"Moved in x, now I am at: {mdl.get_EE_coords()}")
            # translate((0,-0.03,0),ACC,VEL)
            # print(f"Moved in y, now I am at: {mdl.get_EE_coords()}")
    
            translate(
                (x_offset_floor_place_finetune_1, 
                 y_offset_floor_place_finetune_1,
                 0
                 ), ACC, VEL)    

            
            c = mdl.get_EE_coords()
            c[2] =  + z_offset_floor_place + z_offset_floor_place_finetune_1
            o = mdl.get_orientation()
            rotation_vector = rotation_matrix_to_vector(np.array(o))
            c.extend(rotation_vector.tolist())
            moveL(c, ACC, VEL) 
            
        elif floorpartid == 4:
            # print('second floor')
            temp_pos =mdl.get_joints()
            temp_pos[5] = temp_pos[5] - floor_place_rotation_2
            moveJ(temp_pos, ACC, VEL)  

            translate(
                (x_offset_floor_place_finetune_2, 
                 y_offset_floor_place_finetune_2,
                 0
                 ), ACC, VEL)
            
            c = mdl.get_EE_coords()
            c[2] =  + z_offset_floor_place + z_offset_floor_place_finetune_2
            o = mdl.get_orientation()
            rotation_vector = rotation_matrix_to_vector(np.array(o))
            c.extend(rotation_vector.tolist())
            moveL(c, ACC, VEL)
                    
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


from collections import defaultdict, Counter 

def create_element_list(_IFC_sorted):
    raw_items = []
    bathroom_count = 0
    for row in _IFC_sorted:
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
    return ordered_items

def trigger_resume(new_state):
    """
    Sets the resume state to True or False in the file.

    Parameters:
    - new_state: bool, the value to write into the resume state file
    """
    filepath = '_workingdata/_siteinfo/resume_state.txt'
    state_str = 'True' if new_state else 'False'
    with open(filepath, 'w') as f:
        f.write(state_str)
    # print(f"Saved resume state: {state_str}")

def update_state(new_state):
    """
    Updates the file with the given new_state string.

    Parameters:
    - new_state: str, the new state to write into the file
    """
    filepath = '_workingdata/_siteinfo/_construction_state/_stage.txt'
    with open(filepath, 'w') as f:
        f.write(str(new_state))

def adjust_index(lst):
    """
    Reads the element from the file and returns its index in the list.

    Parameters:
    - lst: list of elements

    Returns:
    - int: index of the element read from the file

    Raises:
    - ValueError: if the element is not in the list
    - FileNotFoundError: if the file path is invalid
    """
    filepath = '_workingdata/_siteinfo/_construction_state/_stage.txt'
    with open(filepath, 'r') as f:
        element = f.read().strip()
    return lst.index(element)


def check_resume():
    """
    Checks and returns the resume state from the file.

    Returns:
    - bool: True if resume is enabled, False otherwise
    """
    filepath = '_workingdata/_siteinfo/resume_state.txt'
    try:
        with open(filepath, 'r') as f:
            content = f.read().strip()
            return content == 'True'
    except FileNotFoundError:
        return False  # Default to False if the file is missing
    
def verify(_board, timeout=5) -> bool:
    ok = _board.ask("Is the placement okay?", ("Yes", "No"), timeout)
    return True if ok == "Yes" else False

def perform_construction_routine(
        _board, 
        element, 
        _frame_handler, 
        _ct_pos, 
        _pu_pos, 
        _st_pos, 
        _IFC_sorted, 
        marker_dict = MARKER_DICT):
    grab_pos = None
    placed_pos = None
    if element == "Foundation":
        placed_pos = install_Foundation(_ct_pos,_st_pos,_frame_handler,marker_dict,_IFC_sorted)
    elif element.startswith("bathroom_module"):
        index = int(element[-1])

        from .bathroom import routine_for_bathroom
        grab_pos,placed_pos = routine_for_bathroom(
            _board,
            _frame_handler,
            index,
            place_pos_bathroom=_ct_pos,    # use construction position
            _IFC_sorted=_IFC_sorted,
                                )
    elif element.startswith("Wall"):
        placed_pos = install_Walls(element,_ct_pos,_pu_pos,_frame_handler,marker_dict,_IFC_sorted)
    elif element.startswith("Floor"):
        placed_pos = install_Floors(element,_ct_pos,_st_pos,_pu_pos,_frame_handler,marker_dict,_IFC_sorted)
    else:
        print("Invalid element")
    return grab_pos, placed_pos

def install_Foundation(_ct_pos,_st_pos,_frame_handler,_marker_dict,_IFC_sorted):
    search_element("Foundation", _frame_handler, marker_dict=_marker_dict)
    # 2. Pick the item.
    pick_element("Foundation", _frame_handler,_st_pos,marker_dict=_marker_dict )
    
    # 3. Swing to the construction (placement) site.
    swing(_ct_pos, "Foundation", "swing")

    # 4. Set placement coordinates from the .IFC and place the item 
    place_coords = _IFC_sorted[0][1:4]
 
    placed_pos = place_element("Foundation", 0, place_coords, marker_dict=_marker_dict )
    return placed_pos

def install_Walls(element,_ct_pos,_pu_pos,_frame_handler,_marker_dict,_IFC_sorted):
    search_element(element, _frame_handler, marker_dict=_marker_dict)
    # 2. Pick the item.
    pick_element(element, _frame_handler, _pu_pos, marker_dict=_marker_dict )
    
    # 3. Swing to the construction (placement) site.
    swing(_ct_pos, element, "swing")

    # 4. Set placement coordinates from the .IFC and place the item
    index = (int(element[-1])*2)-1
    place_coords = _IFC_sorted[index][1:4]

    # 5. Place the element
    placed_pos = place_element(element, index, place_coords, marker_dict=_marker_dict)

    return placed_pos

def install_Floors(element,_ct_pos,_st_pos,_pu_pos,_frame_handler,_marker_dict,_IFC_sorted):
    search_element(element, _frame_handler, marker_dict=_marker_dict)
    # 2. Pick the item.
    pick_element(element,_frame_handler, _pu_pos, marker_dict=_marker_dict )
    
    # 3. Swing to the construction (placement) site.
    swing(_ct_pos, element, "swing")

    # 4. Set placement coordinates from the .IFC and place the item
    index = (int(element[-1])*2)
    place_coords = _IFC_sorted[index][1:4]

    # 5. Place the element
    placed_pos = place_element(element, index, place_coords, marker_dict=_marker_dict)

    return placed_pos
        
def construct_and_verify(_board,_frame_handler,_IFC_sorted):
    full_list = create_element_list(_IFC_sorted)
    start_idx = adjust_index(full_list) + 1 if check_resume() else 0
    working_list = full_list[start_idx:]
    frame = None
    
    # Update state.
    if not check_resume():
        with construction_status.state_lock:
            construction_status.state["current_element"] = "scanning_site"
            construction_status.state["current_state"] = "-"
    pu_pos, ct_pos, st_pos = scan_site(_frame_handler, use_stub=check_resume())
    # pu_pos, ct_pos, st_pos = scan_site(_frame_handler, use_stub=True)

    pickup_positions = []        # For extra data (e.g. toilet block information).
    placed_positions = []  # To store the end-effector positions after placement.
    all_ok = True

    if not working_list:
        print("Finished construction.")
        return [], [], True, None

    if len(working_list) == 0:
        working_list = full_list[-1]
    
    if working_list[0].startswith("bathroom_module"):
        moveJ(st_pos, ACC, VEL)

    for item in working_list:
        pickup_postion,placed_position = perform_construction_routine(
                _board,
                item,
                _frame_handler,
                ct_pos, 
                pu_pos, 
                st_pos,
                _IFC_sorted,
        )
        if pickup_postion is not None:
            pickup_positions.append(pickup_postion)
        if placed_position is not None:
            placed_positions.append(placed_position)

        verified = verify(_board)
        # print(f"The verfication is: {verified}")
        if not verified:
            update_state(item)
            trigger_resume(True)
            all_ok = False
            translate((0, 0.05, -0.05), ACC, VEL)
            frame = _frame_handler.get_latest_frame()
            break
        else:
            trigger_resume(False) 
        if item.startswith("Foundation") or item.startswith("Floor"):
            swing(st_pos, item, "swing_back")
        else:
            swing(pu_pos,item,"swing_back")

    if all_ok:
        print("Finished construction.")

    return pickup_positions,placed_positions,all_ok,frame

def reconstruct_and_verify(_board,_dis_pos,_frame_handler,_IFC_sorted):
    full_list = create_element_list(_IFC_sorted)
    start_idx = adjust_index(full_list) + 1 if check_resume() else 0
    working_list = full_list[start_idx:]
    working_list = [item for item in working_list if item != "Foundation"]
    frame = None

    global x_offset_wall_place_finetune_2
    x_offset_wall_place_finetune_2 = 0.005  # <- Bigger integer for ifc2

    
    # Update state.
    if not check_resume():
        with construction_status.state_lock:
            construction_status.state["current_element"] = "scanning_site"
            construction_status.state["current_state"] = "-"

    _, ct_pos, st_pos = scan_site(_frame_handler, use_stub=True)

    pickup_positions = []        # For extra data (e.g. toilet block information).
    placed_positions = []  # To store the end-effector positions after placement.
    all_ok = True

    if not working_list:
        print("Finished reconstruction.")
        return [], [], True, None

    if len(working_list) == 0:
        working_list = full_list[-1]
    
    if working_list[0].startswith("bathroom_module"):
        moveJ(st_pos, ACC, VEL)

    for item in working_list:
        pickup_postion,placed_position = perform_construction_routine(
                _board,
                item,
                _frame_handler,
                ct_pos, 
                _dis_pos, 
                st_pos,
                _IFC_sorted,
        )
        if pickup_postion is not None:
            pickup_positions.append(pickup_postion)
        if placed_position is not None:
            placed_positions.append(placed_position)

        verified = verify(_board)
        # print(f"The verfication is: {verified}")
        if not verified:
            update_state(item)
            trigger_resume(True)
            all_ok = False
            translate((0, 0.05, -0.05), ACC, VEL)
            frame = _frame_handler.get_latest_frame()
            break
        else:
            trigger_resume(False) 
        if item.startswith("Foundation") or item.startswith("Floor"):
            swing(st_pos, item, "swing_back")
        else:
            swing(_dis_pos,item,"swing_back")

    if all_ok:
        print("Finished reconstruction.")

    return pickup_positions,placed_positions,all_ok,frame

def create_indexed_dict(item_list, pl_pos):
    indexed_dict = {
        index: {"name": name, "pos": pos}
        for index, (name, pos) in enumerate(zip(item_list, pl_pos))
    }
    return indexed_dict

def load_saved_runs(saved_paths):
    """
    Load block_list and pl_pos from saved JSON files in order.
    Args:
        saved_paths (list of str): File paths to JSON payloads.
    Returns:
        tuple: (combined_block_list, combined_pl_pos)
    """
    combined_bl = []
    combined_pp = []
    for path in sorted(saved_paths):
        with open(path, 'r') as f:
            data = json.load(f)
        combined_bl.extend(data.get('block_list', []))
        combined_pp.extend(data.get('pl_pos', []))
    return combined_bl, combined_pp

def disassembly(_pl_pos,_block_list, _framehandler, _dis_pos = dis_pos,saved_paths=None):
    _item_list = [
        "Foundation",
        "bathroom_module_1",
        "Wall_1",
        "Floor_1",
        "bathroom_module_2",
        "Wall_2",
        "Floor_2",
        "bathroom_module_2"
    ]

    if len(_item_list) != len(_pl_pos):
        if not saved_paths:
            raise ValueError(
                "Length mismatch between item_list and pl_pos, and no saved_paths provided."
            )
        _block_list, _pl_pos = load_saved_runs(saved_paths)
        if len(_item_list) != len(_pl_pos):
            raise ValueError(
                f"After loading saved runs, lengths still mismatch: {len(_item_list)} items vs {len(_pl_pos)} positions."
            )

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
    _ = find_part(MARKER_DICT["deconstruction"], _framehandler)
    translate((0.1, 0.14, 0.06), ACC, VEL)
    _dis_pos = mdl.get_joints()

    moveJ(_ct_pos, ACC, VEL)

    while i > -1:
        item = indexed_dict[i]['name']
        pickup_position_data = indexed_dict[i]['pos']

        if item != "Foundation":
            gripper_width(100)

            if i != (len(_pl_pos)-1):
                swing(_ct_pos, item,"swing_back") # Go to construction site
                translate((0.05,0,0),ACC,VEL)

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

            # if item.startswith("Floor"):
            #     translate((0, 0, 0.0055), ACC, VEL)

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
                translate((0, 0, -0.003), ACC, VEL)
                gripper_width(100)
                time.sleep(2)
                translate((0, 0, -0.06), ACC, VEL)
            else:
                swing(_dis_pos, item, "swing")
                
                activity = 'place'
                with construction_status.state_lock:
                    construction_status.state["current_state"] = activity

                                
                x_move = drops[n_placed][0]
                y_move = -drops[n_placed][1]

                translate((x_move, y_move, 0), ACC, VEL)

                c = mdl.get_EE_coords()
                if item[:-2] == "Wall":  # Fine-tuning for wall
                    z_extra = 0 #0.015
                    c[2] =  pickup_offsets["wall"]["z"] + z_extra


                elif item[:-2] == "Floor":  # Fine-tuning for floor
                    z_extra = 0 #0.015
                    c[2] =  pickup_offsets["floor"]["z"] + z_extra

                o = mdl.get_orientation()
                rotation_vector = rotation_matrix_to_vector(np.array(o))
                c.extend(rotation_vector.tolist())
                #Go-to new position
                moveL(c, ACC, VEL)

                time.sleep(1)
                gripper_width(100)
                time.sleep(2)
                translate((0, 0, -0.06), ACC, VEL)
                n_placed += 1
            
        i -=1
    print("Finished deconstruction!")
    return _dis_pos

