#camera/annotate.py
import cv2
import numpy as np
import time

def showFrame(frame, scale=1.5):
    """
    Resize the frame by a given scale factor.
    """
    height, width = frame.shape[:2]
    new_dimensions = (int(width * scale), int(height * scale))
    return cv2.resize(frame, new_dimensions, interpolation=cv2.INTER_LINEAR)

def annotateTime(frame,
                 position=(10, 30),
                 font=cv2.FONT_HERSHEY_SIMPLEX,
                 fontScale=1.0,
                 color=(255, 255, 255),
                 thickness=2):
    """
    Draws a timestamp (YYYY-MM-DD HH:MM:SS) in the top-left corner of the frame.
    :param frame: image to annotate in-place
    :param position: bottom-left corner of the text (x, y)
    :param font, fontScale, color, thickness: styling parameters
    """
    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    cv2.putText(frame, ts, position, font, fontScale, color, thickness, cv2.LINE_AA)
    return frame

def findArucoMarkers(img, markerSize=6, totalMarkers=250, draw=True):
    """
    Detect ArUco markers in an image.
    
    :param img: Input image.
    :param markerSize: Marker size (e.g. 6).
    :param totalMarkers: Total markers in the dictionary.
    :param draw: (Unused) Option to draw markers.
    :return: (markerCorners, markerIds)
    """
    dictionary = cv2.aruco.getPredefinedDictionary(
        getattr(cv2.aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    )
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    markerCorners, markerIds, _ = detector.detectMarkers(img)
    return markerCorners, markerIds

def annotateMarkersBasic(frame, bboxs, ids):
    """
    Annotate ArUco markers by:
      - Shading them with a translucent green overlay,
      - Drawing a green border,
      - Placing the text so that its top-right corner is at (min_x, min_y) of the marker.
    """
    FONT_SCALE = 1.0
    alpha = 0.3

    overlay = frame.copy()
    for i in range(len(bboxs)):
        marker_corners = bboxs[i][0]
        marker_id = ids[i][0]

        marker_contour = np.array(marker_corners, dtype=np.int32)
        epsilon = 0.01 * cv2.arcLength(marker_contour, True)
        approx = cv2.approxPolyDP(marker_contour, epsilon, True)

        cv2.drawContours(overlay, [approx], 0, (0, 255, 0), -1)

        rect = cv2.minAreaRect(approx)
        box = cv2.boxPoints(rect)
        box = np.int_(box)
        cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    for i in range(len(bboxs)):
        marker_corners = bboxs[i][0]
        marker_id = ids[i][0]
        text = str(marker_id)

        xs = marker_corners[:, 0]
        ys = marker_corners[:, 1]
        anchor_x = int(np.min(xs))
        anchor_y = int(np.min(ys))

        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(text, font, FONT_SCALE, thickness)

        text_x = anchor_x - text_width
        text_y = anchor_y + text_height

        text_x = max(0, text_x)
        text_y = min(frame.shape[0], text_y)

        cv2.putText(frame, text, (text_x, text_y),
                    font, FONT_SCALE, (255, 255, 255), thickness, cv2.LINE_AA)

    return frame


def annotateDangerZone(frame, tint, tint_alpha=0.15):
    """
    Annotate the danger zone:
      - Draw a horizontal red line at 70% of the frame height,
      - Optionally apply a red tint to the bottom 30% region.
    """
    height, width = frame.shape[:2]
    line_y = int(height * 0.7)

    cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 2)

    if tint:
        red_color = (0, 0, 255)
        bottom_region = frame[line_y:height, :]
        red_overlay = bottom_region.copy()
        red_overlay[:] = red_color
        tinted_region = cv2.addWeighted(red_overlay, tint_alpha, bottom_region, 1 - tint_alpha, 0)
        frame[line_y:height, :] = tinted_region

    return frame

def liveFeedAruco(frame_handler):
    """
    Display a live feed with annotated ArUco markers.
    
    :param frame_handler: Instance of FrameHandler.
    """
    while not frame_handler.stopped:
        frame = frame_handler.get_latest_frame()
        if frame is not None:
            frame_copy = frame.copy()
            bboxs, ids = findArucoMarkers(frame)

            # Add timestamp
            annotateTime(frame_copy)

            # Draw danger zone
            annotateDangerZone(frame_copy,True)  # Set to True to see the red tint. 

            # If any markers are found, annotate them
            if ids is not None and ids.any():
                annotateMarkersBasic(frame_copy, bboxs, ids)

            # Show enlarged frame
            enlarged = showFrame(frame_copy, scale=1.5)
            cv2.imshow('Live Feed', enlarged)
                
        if cv2.waitKey(1) & 0xFF == ord('q'):
            frame_handler.stopped = True
            break
    cv2.destroyAllWindows()



def transmitFeed(frame_handler):
    """
    Annotate each frame and let FrameHandler send it over WebSocket.
    No local imshow or waitKey calls here.
    """
    
    while not frame_handler.stopped:
        frame = frame_handler.get_latest_frame()
        if frame is not None:
            # Make a copy so we don't modify the original in place
            frame_copy = frame.copy()

            # 1) Danger zone annotation
            annotateDangerZone(frame_copy, True)  # e.g. red tint on bottom 30%

            # 2) ArUco detection + annotation
            bboxs, ids = findArucoMarkers(frame_copy)
            if ids is not None and ids.any():
                annotateMarkersBasic(frame_copy, bboxs, ids)

            # Now store the annotated frame back so the publisher sends it
            # We'll lock and replace frame_handler.frame with our annotated version
            with frame_handler.lock:
                frame_handler.frame = frame_copy

        # Sleep a bit so we don't slam the CPU
        time.sleep(0.005)
    
    # Optionally do any cleanup here
    print("[transmitFeed] Stopped.")

