import numpy as np

CAMERA_MATRIX = np.array([
    [616.83925196, 0.0, 326.33664825],
    [0.0, 617.05481514, 234.61266457],
    [0.0, 0.0, 1.0]
])

DIST_COEFFS = np.array([[0.07621379, 0.3656089, 0.00612415, 0.00228554, -1.79890498]])

# R_ZERO defines the zero-degree reference for the system.
R_ZERO = np.array([
    [-9.99998739e-01, -7.55834573e-07, -1.58790778e-03],
    [-7.96316814e-07,  1.00000000e+00,  2.54934437e-05],
    [ 1.58790776e-03,  2.54946760e-05, -9.99998739e-01]
])

frame_length_by_2  = 650/2
frame_height = 250

#Robot movement acceleration and velocity
ACC = 0.2
VEL = 0.2

camera_offsets = {
    "x":-0.028,
    "y":-0.098 ,
    "z": 0.187 
}

ground_z_zero = 0.19225675659682884 #When end-effector[2] is here with closed gripper,we hit ground.

