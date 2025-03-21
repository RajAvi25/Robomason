# construct/construction.py

import time
import pandas as pd
from math import pi
from datetime import datetime
import cv2
from system_config import *
from construction_config import *
from ui.mobility import moveJ, translate,set_orientation, gripper_width
import ui.MarkerDetectionLocalization as mdl
import pickle
from .marker_pose import find_part
from detections.marker_detector import MarkerDetector
from .bathroom import *
from func_timeout import FunctionTimedOut, func_timeout
from . import construction_status

# --- Site Scanning Routine ---
def scan_site(frame_handler, use_stub=False):
    """
    Scans the construction site and returns key positions:
      - pu_pos: Pickup position (for picking parts)
      - ct_pos: Construction site position (for placement)
      - st_pos: Storage position
    If use_stub is True, positions are loaded from a file.
    """
    if not use_stub:
        print('Performing site scan (real routine)')
        moveJ(camera_pos1, ACC, VEL)
        
        # Determine Pickup Position using the storage marker:
        pu_pos = find_part(MARKER_DICT["storage"], frame_handler)
        z_offset_value = 0.105
        translate((0.15, 0.1, z_offset_value), ACC, VEL)
        
        # Determine Construction Site Position:
        moveJ(drop_pos, ACC, VEL)
        translate((-0.1, 0, 0), ACC, VEL)
        translate((0, 0, -0.08), ACC, VEL)
        ct_pos = find_part(MARKER_DICT["construction"], frame_handler)
        
        # Determine Storage Position:
        moveJ(drop_pos, ACC, VEL)
        translate((0.1, 0, 0), ACC, VEL)
        _ = find_part(MARKER_DICT["pick_up"], frame_handler)
        st_pos = mdl.get_joints()
        
        moveJ(pu_pos, ACC, VEL)
    else:
        print('Using stub file for site scan')
        path = "/home/avi/Desktop/robomason/_workingdata/_siteinfo/saved_positions.pkl"
        with open(path, "rb") as f:
            pu_pos, ct_pos, st_pos, _ = pickle.load(f)
    return pu_pos, ct_pos, st_pos

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

# --- Construction Routine ---
def perform_construction(IFC_sorted, marker_dict, frame_handler):
    
    item_list = []
    block_list = []
    placed_positions = []
    df = pd.DataFrame({'task': [], 'start date': [], 'end date': []})
    
    start_time = datetime.now()

    elements = 'Scanning site'
    activity= 'grab'

    with construction_status.state_lock:
        construction_status.state["current_element"] = elements
        construction_status.state["current_state"] = activity

    pu_pos, ct_pos, st_pos = scan_site(frame_handler, use_stub = False)

    end_time = datetime.now()
    df.loc[len(df)] = {"task": "Initializing", "start date": start_time, "end date": end_time}
    
    names = IFC_sorted[:, 0]
    n_place = IFC_sorted.shape[0]
    n_placed = 0
    tries = 0
    start_pos = pu_pos
    moveJ(start_pos, ACC, VEL)
    
    while n_placed < n_place:
        start_time = datetime.now()

        moveJ(start_pos, ACC, VEL)

        item = names[n_placed].split(":")[1].split("-")[0].strip()

        print(f"Searching for {item}")
        task = "Placing " + item

        # elements = 'Searching element'
        # activity= 'grab'
        # with construction_status.state_lock:
        #             construction_status.state["current_element"] = elements
        #             construction_status.state["current_state"] = activity
        
        img = frame_handler.get_latest_frame()
        _, ids = MarkerDetector.find_objects(img)
  
        if (ids is not None and marker_dict.get(item) in ids):
            img = frame_handler.get_latest_frame()
            _, ids = MarkerDetector.find_objects(img)
            print("The item is here:")

            elements = item
            activity= 'grab'

            with construction_status.state_lock:
                construction_status.state["current_element"] = elements
                construction_status.state["current_state"] = activity

            # Grab function
            grab(start_pos, marker_dict.get(item), frame_handler)
            print("I have now grabbed it!")

            # Now for the placement functions
            place_coords = IFC_sorted[n_placed, 1:4]

            activity= 'place'

            with construction_status.state_lock:
                construction_status.state["current_state"] = activity

            # Place function
            print("We are now placing:")
            pos_placed = place(ct_pos, place_coords, marker_dict.get(item),n_placed)
            item_list.append(item)
            placed_positions.append(pos_placed)

            end_time = datetime.now()
            new_row = {'task': task, 'start date': start_time, 'end date': end_time}
            new_row_df = pd.DataFrame([new_row])
            df = pd.concat([df, new_row_df], ignore_index=True)

            # Toilet placements
            if item.strip() == "Floor" or item.strip() == "Foundation":
                elements = 'Toilet'
                activity= 'grab'

                with construction_status.state_lock:
                    construction_status.state["current_element"] = elements
                    construction_status.state["current_state"] = activity

                start_time = datetime.now()
                task = "Placing toilet"
                print("We should now place a colored toilet!")
                toilet_found = False

                # Move to storage zone
                moveJ(st_pos,ACC, VEL)

                # Find and grab block of specified color
                red_found, blue_found = check_frame(bathroom_module_color_thresholds, bathroom_module_area_thresholds, frame_handler)
                print("Red found=", red_found)
                print("Blue found=", blue_found)

                # Ask what color toilet to place
                answer = None
                try:
                    answer = func_timeout(5, lambda: input('Input color of toilet you want placed [r/b]:\n'))
                    print(answer)
                except FunctionTimedOut:
                    print("Too slow, I will just pick a color for you!")
                    answer = None

                if answer is None:
                    print("Too slow, I will just pick a color for you!")
                    if red_found:
                        toilet_pos_grab = grab_toilet(bathroom_module_color_thresholds["red"]["LB"], 
                                  bathroom_module_color_thresholds["red"]["UB"], 
                                  frame_handler)
                        toilet_found = True
                    elif blue_found:
                        toilet_pos_grab = grab_toilet(bathroom_module_color_thresholds["blue"]["LB"], 
                                  bathroom_module_color_thresholds["blue"]["UB"], 
                                  frame_handler)
                        toilet_found = True
                elif answer == "r" and red_found:
                    toilet_pos_grab = grab_toilet(bathroom_module_color_thresholds["red"]["LB"], 
                                  bathroom_module_color_thresholds["red"]["UB"], 
                                  frame_handler)
                    toilet_found = True
                elif answer == "b" and blue_found:
                    toilet_pos_grab = grab_toilet(bathroom_module_color_thresholds["blue"]["LB"], 
                                  bathroom_module_color_thresholds["blue"]["UB"], 
                                  frame_handler)
                    toilet_found = True

                if toilet_found:
                    block_list.append(toilet_pos_grab)

                    activity= 'place'

                    with construction_status.state_lock:
                        construction_status.state["current_state"] = activity

                    # Place toilet
                    time.sleep(0.5)
                    moveJ(ct_pos,ACC, VEL)
                    time.sleep(0.5)
                    print("n_placed is:", n_placed)
                    toilet_pos = place_toilet(IFC_sorted, n_placed)

                    placed_positions.append(toilet_pos)
                    print("Toilet is now placed!")
                    item_list.append("Toilet")
                else:
                    print("Could not find the correct toilet, moving on...")

                end_time = datetime.now()
                new_row = {'task': task, 'start date': start_time, 'end date': end_time}
                new_row_df = pd.DataFrame([new_row])
                df = pd.concat([df, new_row_df], ignore_index=True)

            # After successful placement
            print("Now placed all good!")
            n_placed += 1
            tries = 0
        else:
            elements = 'Searching element'
            activity= 'grab'
            print("We can't see the item!")

            with construction_status.state_lock:
                construction_status.state["current_element"] = elements
                construction_status.state["current_state"] = activity

            time.sleep(3)
            tries += 1
            if tries > 5:
                    userAssist = None
                    try:
                        userAssist = func_timeout(5, lambda: input('Please help me. Should I continue [y/n]:\n'))
                        print(userAssist)
                    except FunctionTimedOut:
                        print("Too slow, I will try again!")
                        userAssist = 'y'
                        if userAssist == 'y':
                            tries += 1
                            continue
                        if userAssist == 'n':
                            break

    elements = 'Scanning site'
    activity= 'grab'

    with construction_status.state_lock:
        construction_status.state["current_element"] = elements
        construction_status.state["current_state"] = activity

    start_time = datetime.now()
    # Move to camera position 1
    moveJ(camera_pos1,ACC, VEL)
    time.sleep(1)

    # # Take image
    # img = frame_handler.get_latest_frame()
    # img_name1 = item + str(n_placed) + "view1" + ".jpg"
    # cv2.imwrite(img_name1, img)

    # Move back
    moveJ(ct_pos,ACC, VEL)
    time.sleep(2)

    end_time = datetime.now()
    new_row = {'task': task, 'start date': start_time, 'end date': end_time}
    new_row_df = pd.DataFrame([new_row])
    df = pd.concat([df, new_row_df], ignore_index=True)

    if tries < 5:
        print("Placement done!")

    return item_list, block_list, placed_positions, df

#Reconstruction routine
def perform_reconstruction(IFC_sorted, marker_dict, frame_handler):
    
    item_list = []
    block_list = []
    placed_positions = []
    df = pd.DataFrame({'task': [], 'start date': [], 'end date': []})
    
    start_time = datetime.now()

    # elements = 'Scanning site'
    # activity= 'grab'

    # with construction_status.state_lock:
    #     construction_status.state["current_element"] = elements
    #     construction_status.state["current_state"] = activity

    _, ct_pos, st_pos = scan_site(frame_handler, use_stub=True)

    end_time = datetime.now()
    df.loc[len(df)] = {"task": "Initializing", "start date": start_time, "end date": end_time}
    
    names = IFC_sorted[:, 0]
    n_place = IFC_sorted.shape[0]
    n_placed = 0
    tries = 0
    start_pos = dis_pos
    moveJ(start_pos, ACC, VEL)
    
    while n_placed < n_place:
        start_time = datetime.now()

        moveJ(start_pos, ACC, VEL)
        translate((-0.1,0.08,0),ACC,VEL)

        item = names[n_placed].split(":")[1].split("-")[0].strip()
        print(item)

        print(f"Searching for {item}")
        task = "Placing " + item

        # If the item is "Foundation", skip the grab and place routine
        if item == "Foundation":
            print("Skipping placement for Foundation but continuing with the toilet routine.")
        else:
            img = frame_handler.get_latest_frame()
            _, ids = MarkerDetector.find_objects(img)

            if ids is not None and marker_dict.get(item) in ids:
                img = frame_handler.get_latest_frame()
                _, ids = MarkerDetector.find_objects(img)
                print("The item is here:")

                elements = item
                activity = 'grab'

                with construction_status.state_lock:
                    construction_status.state["current_element"] = elements
                    construction_status.state["current_state"] = activity

                # Grab function
                grab(start_pos, marker_dict.get(item), frame_handler)
                print("I have now grabbed it!")

                # Now for the placement functions
                place_coords = IFC_sorted[n_placed, 1:4]

                activity = 'place'

                with construction_status.state_lock:
                    construction_status.state["current_state"] = activity

                # Place function
                print("We are now placing:")
                pos_placed = place(ct_pos, place_coords, marker_dict.get(item), n_placed)
                item_list.append(item)
                placed_positions.append(pos_placed)

            else:
                elements = 'Searching element'
                activity = 'grab'
                print("We can't see the item!")

                with construction_status.state_lock:
                    construction_status.state["current_element"] = elements
                    construction_status.state["current_state"] = activity

                time.sleep(3)
                tries += 1
                if tries > 5:
                    userAssist = None
                    try:
                        userAssist = func_timeout(5, lambda: input('Please help me. Should I continue [y/n]:\n'))
                        print(userAssist)
                    except FunctionTimedOut:
                        print("Too slow, I will try again!")
                        userAssist = 'y'
                    
                    if userAssist == 'y':
                        tries += 1
                        continue
                    if userAssist == 'n':
                        break

        # **Toilet Routine Runs Even If Item is Foundation**
        if item.strip() == "Floor" or item.strip() == "Foundation":
            elements = 'Toilet'
            activity = 'grab'

            with construction_status.state_lock:
                construction_status.state["current_element"] = elements
                construction_status.state["current_state"] = activity

            start_time = datetime.now()
            task = "Placing toilet"
            print("We should now place a colored toilet!")
            toilet_found = False

            # Move to storage zone
            moveJ(st_pos, ACC, VEL)

            # Find and grab block of specified color
            red_found, blue_found = check_frame(bathroom_module_color_thresholds, bathroom_module_area_thresholds, frame_handler)
            print("Red found=", red_found)
            print("Blue found=", blue_found)

            # Ask what color toilet to place
            answer = None
            try:
                answer = func_timeout(5, lambda: input('Input color of toilet you want placed [r/b]:\n'))
                print(answer)
            except FunctionTimedOut:
                print("Too slow, I will just pick a color for you!")
                answer = None

            if answer is None:
                print("Too slow, I will just pick a color for you!")
                if red_found:
                    toilet_pos_grab = grab_toilet(bathroom_module_color_thresholds["red"]["LB"], 
                              bathroom_module_color_thresholds["red"]["UB"], 
                              frame_handler)
                    toilet_found = True
                elif blue_found:
                    toilet_pos_grab = grab_toilet(bathroom_module_color_thresholds["blue"]["LB"], 
                              bathroom_module_color_thresholds["blue"]["UB"], 
                              frame_handler)
                    toilet_found = True
            elif answer == "r" and red_found:
                toilet_pos_grab = grab_toilet(bathroom_module_color_thresholds["red"]["LB"], 
                              bathroom_module_color_thresholds["red"]["UB"], 
                              frame_handler)
                toilet_found = True
            elif answer == "b" and blue_found:
                toilet_pos_grab = grab_toilet(bathroom_module_color_thresholds["blue"]["LB"], 
                              bathroom_module_color_thresholds["blue"]["UB"], 
                              frame_handler)
                toilet_found = True

            if toilet_found:
                block_list.append(toilet_pos_grab)

                activity = 'place'

                with construction_status.state_lock:
                    construction_status.state["current_state"] = activity

                # Place toilet
                time.sleep(0.5)
                moveJ(ct_pos, ACC, VEL)
                time.sleep(0.5)
                print("n_placed is:", n_placed)
                toilet_pos = place_toilet(IFC_sorted, n_placed)

                placed_positions.append(toilet_pos)
                print("Toilet is now placed!")
                item_list.append("Toilet")
            else:
                print("Could not find the correct toilet, moving on...")

            end_time = datetime.now()
            new_row = {'task': task, 'start date': start_time, 'end date': end_time}
            new_row_df = pd.DataFrame([new_row])
            df = pd.concat([df, new_row_df], ignore_index=True)

        # After successful placement or skipping foundation
        print("Now placed all good!")
        n_placed += 1
        tries = 0

    elements = 'Scanning site'
    activity = 'grab'

    with construction_status.state_lock:
        construction_status.state["current_element"] = elements
        construction_status.state["current_state"] = activity

    start_time = datetime.now()
    moveJ(dis_pos, ACC, VEL)
    time.sleep(1)
    moveJ(ct_pos, ACC, VEL)
    time.sleep(2)

    end_time = datetime.now()
    new_row = {'task': task, 'start date': start_time, 'end date': end_time}
    new_row_df = pd.DataFrame([new_row])
    df = pd.concat([df, new_row_df], ignore_index=True)

    if tries < 5:
        print("Placement done!")

    return item_list, block_list, placed_positions, df


#Disassembly routine
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

def create_maps(item_list, pl_pos, block_list):
    # Count occurrences for numbering.
    counts = {"Toilet": 0, "Floor": 0, "Wall": 0, "Foundation": 0}
    
    for item in item_list:
        capitalized_item = item.capitalize()  # Ensure consistent capitalization
        if capitalized_item in counts:
            counts[capitalized_item] += 1

    # Set counters starting at the total count (highest number first)
    toilet_counter = counts["Toilet"]
    floor_counter  = counts["Floor"]
    wall_counter   = counts["Wall"]
    
    pl_mapping = {}     # Mapping for every item using pl_pos
    toilet_mapping = {} # Additional mapping for toilets using block_list

    # Create copies to pop from in reverse order.
    available_pl = pl_pos.copy()
    available_blocks = block_list.copy()

    # Process the item_list in reverse order.
    for i in range(len(item_list) - 1, -1, -1):
        item = item_list[i]
        capitalized_item = item.capitalize()  # Ensure first letter is capitalized
        # Since pl_pos and item_list are the same length, pop an entry from available_pl.
        pl_entry = available_pl.pop()
        
        if capitalized_item == "Toilet":
            key = f"Toilet_{toilet_counter}"
            toilet_counter -= 1
            pl_mapping[key] = pl_entry
            # For toilets, also assign an entry from block_list (in reverse order).
            block_entry = available_blocks.pop() if available_blocks else None
            toilet_mapping[key] = block_entry
        elif capitalized_item == "Floor":
            key = f"Floor_{floor_counter}"
            floor_counter -= 1
            pl_mapping[key] = pl_entry
        elif capitalized_item == "Wall":
            key = f"Wall_{wall_counter}"
            wall_counter -= 1
            pl_mapping[key] = pl_entry
        elif capitalized_item == "Foundation":
            # Foundation is unique.
            key = "Foundation"
            pl_mapping[key] = pl_entry
        else:
            # For any other type, simply use the item name.
            pl_mapping[item] = pl_entry

    return pl_mapping, toilet_mapping

def disassembly(_item_list, _pl_pos, _block_list, _framehandler, _ct_pos,_st_pos ,_dis_pos = dis_pos):
    map1, map2 = create_maps(_item_list, _pl_pos, _block_list)
    i = 0
    j = 0

    # map1 = pl_mapping
    # map2 = toilet_mapping

    elements = 'Scanning site'
    activity = 'grab'

    with construction_status.state_lock:
        construction_status.state["current_element"] = elements
        construction_status.state["current_state"] = activity

    moveJ(_dis_pos, ACC, VEL)
    _dis_pos = find_part(MARKER_DICT["deconstruction"], _framehandler)

    while i < len(_item_list):
        gripper_width(100)
        moveJ(_ct_pos, ACC, VEL)
        keys_list = list(map1.keys())
        item = keys_list[i]

        if item == "Foundation":
            print('Disassembly completed.')
            break
        else:
            elements = item[:-2]
            activity = 'grab'

            with construction_status.state_lock:
                construction_status.state["current_element"] = elements
                construction_status.state["current_state"] = activity

            # Move to the element at the construction site
            pos = map1[keys_list[i]]['coords'].copy()
            rotation_vector = rotation_matrix_to_vector(np.array(map1[keys_list[i]]['orientation']))
            pos.extend(rotation_vector.tolist())
            moveL(pos, ACC, VEL)
            time.sleep(0.75)

            if item[:-2] == "Floor":
                translate((0, 0, 0.0055), ACC, VEL)

            # if item[:-2] == "Floor":
            #     translate((0, 0, 0.0035), ACC, VEL)

            # Close gripper to hold object
            for k in range(len(elements_gripper_width[item[:-2]])):
                if elements_gripper_width[item[:-2]][k] is not None:
                    gripper_width(elements_gripper_width[item[:-2]][k])
                    time.sleep(1)

            time.sleep(2.5)
            translate((0, 0, -0.09), ACC, VEL)

            activity = 'place'
            with construction_status.state_lock:
                construction_status.state["current_state"] = activity

            if item[:-2] == 'Toilet':
                # Move to storage zone
                moveJ(_st_pos,ACC, VEL)
                pos = map2[keys_list[i]]['coords'].copy()
                rotation_vector = rotation_matrix_to_vector(np.array(map2[keys_list[i]]['orientation']))
                pos.extend(rotation_vector.tolist())
                moveL(pos, ACC, VEL)
                time.sleep(1)
                gripper_width(100)
                time.sleep(2)
                translate((0, 0, -0.06), ACC, VEL)
            else:
                # Moves to disassembly position
                moveJ(_dis_pos, ACC, VEL)

                if item[:-2] == "Wall":  # Fine-tuning for wall
                    z_extra = 0.015
                    z_down = mdl.get_EE_coords()[2] + pickup_offsets["wall"]["z"] + z_extra
                elif item[:-2] == "Floor":  # Fine-tuning for floor
                    z_extra = -0.0013
                    z_down = mdl.get_EE_coords()[2] + pickup_offsets["floor"]["z"] + z_extra

                x_move = drops[j][0]
                y_move = -drops[j][1]

                translate((x_move, y_move, 0), ACC, VEL)
                translate((0, 0, z_down), ACC, VEL)
                # print(x_move, y_move, z_down)
                time.sleep(1)
                gripper_width(100)
                time.sleep(2)
                translate((0, 0, -0.06), ACC, VEL)  # Moves up
                j += 1

        i += 1