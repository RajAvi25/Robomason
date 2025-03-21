import cv2
import numpy as np
from detections.aruco_detection import find_aruco_markers
from system_config import *
from construction_config import *

class MarkerDetector:
    """
    Handles marker detection and pose estimation.
    """
    @staticmethod
    def find_objects(img, marker_size=6, total_markers=250):
        return find_aruco_markers(img, marker_size, total_markers)

    @staticmethod
    def get_marker_coordinates(img, rotation_degrees, is_debugging=False):
        marker_size = MARKER_SIZE
        camera_matrix = CAMERA_MATRIX
        dist_coeffs = DIST_COEFFS
        corners, ids = MarkerDetector.find_objects(img)
        detected_markers = []
        if ids is not None:
            obj_points = np.array([
                [-marker_size / 2,  marker_size / 2, 0],
                [ marker_size / 2,  marker_size / 2, 0],
                [ marker_size / 2, -marker_size / 2, 0],
                [-marker_size / 2, -marker_size / 2, 0]
            ], dtype=np.float32)
            theta = np.radians(-rotation_degrees)
            R_rotation = np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta),  np.cos(theta), 0],
                [0, 0, 1]
            ])
            for i, corner in enumerate(corners):
                marker_id = ids[i][0]
                if marker_id in WORKER_COLORS.keys():
                    success, rvec, tvec = cv2.solvePnP(obj_points, corner[0], camera_matrix, dist_coeffs)
                    if success:
                        if is_debugging:
                            print(f"Marker ID: {marker_id}")
                            print(f"x: {-tvec[1][0]*100}[cm], y: {tvec[0][0]*100}[cm], z: {tvec[2][0]*100}[cm]")
                        tvec_rotated = R_rotation @ tvec
                        detected_markers.append(((tvec_rotated[0][0], tvec_rotated[1][0], tvec_rotated[2][0]), ids[i]))
            if detected_markers:
                return detected_markers
        return None
