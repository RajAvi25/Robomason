# construct/construction_config.py

import numpy as np

WORKER_COLORS = {
    5: 'cyan',
    6: 'green',
    7: 'blue',
    8: 'orange',
    9: 'purple',
    13: 'brown',
    14: 'pink',
    15: 'lime',
    16: 'red',
    17: 'gold',
    18: 'magenta',
    19: 'teal'
}

worker_marker_styles = {
    5: 'x',     
    6: 'x',      
    7: 'x',      
    8: 'x',     
    9: 'x',      
    13: 'x',    
    14: 'x',     
    15: 'x',     
    16: 'zone',     
    17: 'x',    
    18: 'x',     
    19: 'x'      
}

SITE_MARKER = [0,1,2,3]
ELEMENT_MARKER = [10,11,12,13]

# Axes limits
z_level = -0.155
x_limits = (-0.35, 0.81)
y_limits = (-0.25, 0.75)
z_limits = (z_level, 0.49)

# Dimensions for environment "boxes"
b1_dims = (29.5 / 100, 21 / 100)  # pickup site
b2_dims = (21 / 100, 29.5 / 100)  # bathroom module site
b3_dims = (21 / 100, 29.5 / 100)  # construction site
b4_dims = (45 / 100, 29.5 / 100)  # deconstruction site

box_positions = [
    (-0.1522, -0.0485),
    (0.5802, 0.2528),
    (0.5802, 0.5669),
    (0.242,  -0.0485)
]
box_dims = [b1_dims, b2_dims, b3_dims, b4_dims]
box_color = 'darkblue'
box_alpha = 0.5

MARKER_DICT = {
    "pick_up": 2,
    "storage": 1,
    "construction": 0,
    "deconstruction": 3,
    "Foundation": 11, 
    "Wall": 10,
    "Floor": 12
}

MARKER_SIZE = 0.023
SITE_MARKER_SIZE = 0.02
ELEMENT_MARKER_SIZE = 0.01
WORKER_SQUARE_SIZE = 0.1

camera_pos1 = [0.9408252239227295,
 -1.7368041477599085,
 1.8473270575152796,
 -1.7449795208373011,
 -1.5089376608477991,
 -0.58673602739443]

drop_pos = [2.8157572746276855,
 -1.6738444767394007,
 1.7824528853045862,
 -1.640379091302389,
 -1.5882733503924769,
 -0.3120625654803675]

dis_pos = [1.6085065603256226,
 -1.6148439846434535,
 1.8089807669269007,
 -1.7913366756834925,
 -1.5477584044085901,
 -0.019556824360982716]

python_executable_IFC = "/home/avi/anaconda3/envs/learningFactory-2/bin/python"
IFC_function_path = "/home/avi/Desktop/robomason/ifc/IFC_functions.py"  

orientations = {
    "ct":[[-0.06447076, -0.99791238,  0.00379492],
       [-0.9977449 ,  0.06453016,  0.01846535],
       [-0.01867169, -0.00259589, -0.9998223 ]],

    "floor": [[ 0.99960182, -0.020567  , -0.01931826],
              [ 0.01895897,  0.99661267, -0.08002336],
              [ 0.02089866,  0.07962524,  0.99660577]],

    "pu": [[ -0.99981043,  0.00480936,  0.01886716],
           [ 0.00432581,  0.99966324, -0.02558695],
           [-0.01898387, -0.02550048, -0.99949454]]
}

# Element pickup related constants
pickup_offsets = {
    "foundation": {"x": -0.038, "y": -0.107, "z": -0.185},
    "wall": {"x": -0.0015, "y": -0.087, "z": -0.182},  # Alternative z: -0.192
    "floor": {"x": -0.038, "y": -0.0995, "z": -0.205}
}

# Disassembly related constants
dx = -0.15
dy = -0.15
d_offset = -0.01
drops = np.array([[0, d_offset], [dx, d_offset] , [0, d_offset+dy] , [dx, d_offset+dy]])

elements_gripper_width = {
    "Wall": (None, 95),
    "Floor": (50, 20),
    "Toilet": (50, 0)
}

# Dictionary for color thresholds
bathroom_module_color_thresholds = {
    "red": {
        "LB": np.array([[0, 100, 100], [160, 100, 100]]),
        "UB": np.array([[10, 255, 255], [179, 255, 255]])
    },
    "blue": {
        "LB": np.array([97, 156, 99]),
        "UB": np.array([116, 255, 255])
    },
}

# Dictionary for area thresholds
bathroom_module_area_thresholds = {
    "min": 300,
    "max": 3000
}

bathroom_module_radius = 0.025

x_offest_bathroom_module_place_1 =  0.0025 # +ve is right
y_offest_bathroom_module_place_1 = 0.0025
z_offest_bathroom_module_place_1 = -0.21 #-0.22

x_offest_bathroom_module_place_2 =  0.0025#-0.005
y_offest_bathroom_module_place_2 = -0.005
z_offest_bathroom_module_place_2 = -0.22 

# Offsets for placing Foundation
x_offset_fund_place = 0.0025
y_offset_fund_place = 0.0 
z_offset_fund_place = -0.005

# Offsets for placing wall parts
x_offset_wall_place = 0.00
y_offset_wall_place = -0.0017  
z_offset_wall_place = 0.0256  

# Offsets for placing floor parts
x_offset_floor_place = +0.0017
y_offset_floor_place = 0.0085  
z_offset_floor_place = -0.024412 

# Fine-tuning offsets for wall placement (first set)
x_offset_wall_place_finetune_1 = -0.0028
y_offset_wall_place_finetune_1 = -0.005
z_offset_wall_place_finetune_1 = 0.002

# Fine-tuning offsets for floor placement (first set)
x_offset_floor_place_finetune_1 = -0.005
y_offset_floor_place_finetune_1 = -0.005
z_offset_floor_place_finetune_1 = 0.0095

# Fine-tuning offsets for wall placement (second set)
x_offset_wall_place_finetune_2 = -0.0035 #0.0
y_offset_wall_place_finetune_2 = 0.0025 # positive would move it backwards
z_offset_wall_place_finetune_2 = 0.002

# Fine-tuning offsets for floor placement (second set)
x_offset_floor_place_finetune_2 = -0.0025
y_offset_floor_place_finetune_2 = -0.0035
z_offset_floor_place_finetune_2 = -0.0150

# Rotation adjustments
wall_place_rotation_finetune = 0.022
floor_place_rotation_1 = 0.015
floor_place_rotation_2 = 0.019


