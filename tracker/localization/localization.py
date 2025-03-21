import json
import base64
import cv2
import numpy as np
from detections.marker_detector import MarkerDetector
from system_config import R_ZERO

class Localizer:
    @staticmethod
    def camera_to_world_coords(c, x_disp, y_disp, in_cm=False):
        x = (-c[0]) + x_disp
        y = c[1] + y_disp
        z = 0.522266 - c[2]
        if in_cm:
            return (x * 100, y * 100, z * 100)
        return (x, y, z)

    @staticmethod
    def calculate_rotation(R_rotated):
        R_rotated = np.array(R_rotated)
        R_relative = np.dot(R_rotated, R_ZERO.T)
        angle = np.arccos((np.trace(R_relative) - 1) / 2)
        angle_degrees = np.degrees(angle)
        axis = np.array([
            R_relative[2, 1] - R_relative[1, 2],
            R_relative[0, 2] - R_relative[2, 0],
            R_relative[1, 0] - R_relative[0, 1]
        ])
        if axis[2] < 0:
            angle_degrees = -angle_degrees
        return angle_degrees

    @staticmethod
    def calculate_offsets(ang):
        h, k = 0.01560, 0.00041
        A = np.array([-0.021 - h, 0.107 - k])
        angle_radians = np.radians(ang)
        rotation_matrix = np.array([
            [np.cos(angle_radians), -np.sin(angle_radians)],
            [np.sin(angle_radians),  np.cos(angle_radians)]
        ])
        rotated_vector = np.dot(rotation_matrix, A)
        B = rotated_vector + np.array([h, k])
        return B[0], B[1]

    @staticmethod
    def process_image(message, save_path, is_debugging=False):
        try:
            ref_x = 0.0
            ref_y = -0.45
            data = json.loads(message)
            img_base64 = data['image']
            img_data = base64.b64decode(img_base64)
            img_array = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
            coordinates = data['coordinates']
            orient = data['orientation']
            ang = Localizer.calculate_rotation(orient)
            x_shift, y_shift = Localizer.calculate_offsets(ang)
            x_offset = ref_x + x_shift
            y_offset = ref_y + y_shift
            x, y, z = coordinates['x'], coordinates['y'], coordinates['z']
            ground_x = round(x - x_offset, 4)
            ground_y = round(y - y_offset, 4)
            markers = MarkerDetector.get_marker_coordinates(img_array, ang, is_debugging)
            results = []
            if markers is not None:
                for (c, marker_id) in markers:
                    c_new = Localizer.camera_to_world_coords(c, ground_x, ground_y, in_cm=False)
                    c_new = tuple(round(coord, 4) for coord in c_new)
                    results.append((c_new, marker_id))
                return img_array, results
            else:
                return img_array, None
        except Exception as e:
            print(f"Error processing image: {e}")
            return None, None
