# /construction_config.py

import numpy as np

#####################################################################
################# Plotting related constants ########################
#####################################################################

ELEMENT_COLORS = {
    "Scanning site":        "royalblue",
    "Foundation":           "forestgreen",
    "Searching element":    "crimson",

    "Wall 1":               "gold",
    "Wall 2":               "darkgoldenrod",

    "Floor 1":              "indigo",
    "Floor 2":              "orchid",

    "Bathroom module 1":    "chocolate",
    "Bathroom module 2":    "sienna",
    "Bathroom module 3":    "tan",
}

# Define a base dictionary for states common to both modes.
BASE_STATE_STYLES = {
    "pick":       {"linewidth": 2.0, "linestyle": "--"},
    "swing":      {"linewidth": 2.0, "linestyle": "-."},
    "place":      {"linewidth": 5.0, "linestyle": "-"},
    "swing_back": {"linewidth": 5.0, "linestyle": ":"}
}

#######

# For disassembly, these are exactly the base styles.
STATE_STYLES_DISASSEMBLY = BASE_STATE_STYLES.copy()
ALLOWED_STATES_DISASSEMBLY = list(BASE_STATE_STYLES.keys())

# For assembly (or reassembly), add an extra state "search".
STATE_STYLES_ASSEMBLY = BASE_STATE_STYLES.copy()
STATE_STYLES_ASSEMBLY["search"] = {"linewidth": 2.0, "linestyle": "-"}
ALLOWED_STATES_ASSEMBLY = ["search"] + list(BASE_STATE_STYLES.keys())

# For a scanning site (if needed):
SCANNING_STYLE = {"linewidth": 3.0, "linestyle": "-"}

SITE_COLOR = 'darkblue'
SITE_ALPHA = 0.5

#######

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

#  Axes limits
z_level = -0.155

# x_limits = (-0.45, 1.05)
# y_limits = (-0.35, 1.15)
# z_limits = (z_level, 0.5)

SCALING_FACTOR = 0.75

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

#####################################################################
###################### IFC related constants ########################
#####################################################################

python_executable_IFC = "/home/avi/anaconda3/envs/learningFactory-2/bin/python"
IFC_function_path = "/home/avi/Desktop/robomason/ifc/IFC_functions.py"  

#####################################################################
###################### Site related constants #######################
#####################################################################

MARKER_DICT = {
    "pick_up": 1,
    "storage": 2,
    "construction": 0,
    "deconstruction": 3,
    "Foundation": 11, 
    "Wall": 10,
    "Floor": 12
}

SITE_MARKER = [0,1,2,3]
ELEMENT_MARKER = [10,11,12,13]

MARKER_SIZE = 0.023
SITE_MARKER_SIZE = 0.02
ELEMENT_MARKER_SIZE = 0.01
WORKER_SQUARE_SIZE = 0.1

camera_pos1 = [0.9474727511405945,
 -1.7047339878477992,
 1.94969350496401,
 -1.8798023662962855,
 -1.5095894972430628,
 -0.5794580618487757]

drop_pos = [2.7558717727661133,
 -1.478697122340538,
 1.5961135069476526,
 -1.6505519352354945,
 -1.5904715696917933,
 -0.3027423063861292]

dis_pos = [1.2044146060943604,
 -1.6227685413756312,
 1.875298802052633,
 -1.8967586956419886,
 -1.5713561216937464,
 -0.4224126974688929]

camera_pos2 = [3.3231146335601807,
 -1.2728570264628907,
 1.3248704115497034,
 -1.6234127483763636,
 -1.5135791937457483,
 0.2665298879146576]

#####################################################################
################# Construction related constants ####################
#####################################################################

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
    "foundation": {"x": -0.037 , "y": -0.095, "z": 0.195},
    "wall": {"x": -0.016, "y": -0.087, "z": 0.20},  
    "floor": {"x": -0.050, "y": -0.105, "z": 0.195}
}

# Disassembly related constants
drops = np.array([
    [ 0.1,        0.17      ],  # upper-right (n_place = 0)
    [0.1,        0.05      ],  # upper-left  (n_place = 1)
    [-0.15,       0.17       ],  # lower-right (n_place = 2)
    [-0.12,      0.05],        # lower-left  (n_place = 3)
])

elements_gripper_width = {
    "Wall": (None, 70),
    "Floor": (50, 20),
    "bathroom_module": (50, 0)
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

#Offsets for placing Bathroom parts
# Dictionary for area thresholds
bathroom_module_area_thresholds = {
    "min": 300,
    "max": 3000
}

bathroom_module_radius = 0.025

x_offest_bathroom_module_place_1 = -0.0025 # +ve is right
y_offest_bathroom_module_place_1 = -0.0125
z_offest_bathroom_module_place_1 = -0.22

x_offest_bathroom_module_place_2 = 0.0005
y_offest_bathroom_module_place_2 = -0.01
z_offest_bathroom_module_place_2 = -0.232

# Offsets for placing Foundation
x_offset_fund_place = 0.0025
y_offset_fund_place = 0.0 
z_offset_fund_place = 0.20

# Offsets for placing wall parts
x_offset_wall_place = 0.008
y_offset_wall_place = -0.0045
z_offset_wall_place = 0.21 

# Fine-tuning offsets for wall placement (first set)
x_offset_wall_place_finetune_1 = -0.01
y_offset_wall_place_finetune_1 = 0.001
z_offset_wall_place_finetune_1 = 0.005

# Fine-tuning offsets for wall placement (second set)
x_offset_wall_place_finetune_2 = -0.015  # <- bigger (right) integer for ifc2
y_offset_wall_place_finetune_2 = -0.001 # positive would move it backwards
z_offset_wall_place_finetune_2 = 0.078

# Offsets for placing floor parts
x_offset_floor_place = -0.02
y_offset_floor_place = 0.0085  
z_offset_floor_place = 0.272

# Fine-tuning offsets for floor placement (first set)
x_offset_floor_place_finetune_1 = 0.02
y_offset_floor_place_finetune_1 = -0.013
z_offset_floor_place_finetune_1 = 0

# Fine-tuning offsets for floor placement (second set)
x_offset_floor_place_finetune_2 = 0.02
y_offset_floor_place_finetune_2 = -0.013
z_offset_floor_place_finetune_2 = 0.072

# Rotation adjustments
wall_place_rotation_finetune = 0.02
floor_place_rotation_1 = 0.02
floor_place_rotation_2 = 0.019


