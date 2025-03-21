import cv2

def find_aruco_markers(img, marker_size=6, total_markers=250, draw=True):
    """
    Detects Aruco markers in the given image and returns marker corners and marker IDs.
    """
    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, f'DICT_{marker_size}X{marker_size}_{total_markers}')
    )
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    marker_corners, marker_ids, _ = detector.detectMarkers(img)
    return marker_corners, marker_ids
