
#camera/annotate.py
import cv2
import numpy as np

def showFrame(frame, scale=1.5):
    """
    Resize the frame by a given scale factor.
    """
    height, width = frame.shape[:2]
    new_dimensions = (int(width * scale), int(height * scale))
    return cv2.resize(frame, new_dimensions, interpolation=cv2.INTER_LINEAR)


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

#######################################################################################
############################# ORIGINAL FUNCTION #######################################
#######################################################################################
# def annotateMarker(frame, bboxs, ids):
#     """
#     Annotate detected markers by drawing a rectangle and putting text (marker ID).
    
#     :param frame: Image frame to annotate.
#     :param bboxs: List of marker bounding boxes.
#     :param ids: List of marker IDs.
#     """
#     for i in range(len(bboxs)):
#         marker_corners = bboxs[i][0]
#         marker_id = ids[i][0]
        
#         # Approximate the marker contour
#         marker_contour = np.array(marker_corners, dtype=np.int32)
#         epsilon = 0.01 * cv2.arcLength(marker_contour, True)
#         approx = cv2.approxPolyDP(marker_contour, epsilon, True)
        
#         # Draw rectangle around the marker
#         rect = cv2.minAreaRect(approx)
#         box = cv2.boxPoints(rect)
#         box = np.int_(box)
#         cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
        
#         # Get top-left corner for text (with an offset)
#         marker_corner_tuple = tuple(marker_corners[0].astype(int))
#         text_offset = 30 
#         cv2.putText(frame, str(marker_id), 
#                     (marker_corner_tuple[0], marker_corner_tuple[1] + text_offset), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

######################################################################################

# def annotateMarker(frame, bboxs, ids):
#     """
#     Annotate detected markers by:
#       - Shading them with a translucent green overlay,
#       - Drawing a green border,
#       - Placing the text so that its top-right corner is at (min_x, min_y) of the marker,
#       - Drawing a horizontal red line at 70% of the frame height,
#       - Adding a red tint to the remaining 30% region,
#       - Resizing the frame by a scale factor (set internally).
#     """
#     FONT_SCALE = 1.0  # Internal font scale variable.
#     alpha = 0.3       # Transparency factor for the green overlay.
#     redtint = True   # Toggle tinting
#     tint_alpha = 0.15  # Tint intensity (30% red overlay)

#     # ---- First loop: Shading and borders ----
#     overlay = frame.copy()
#     for i in range(len(bboxs)):
#         marker_corners = bboxs[i][0]
#         marker_id = ids[i][0]

#         # Approximate the marker contour
#         marker_contour = np.array(marker_corners, dtype=np.int32)
#         epsilon = 0.01 * cv2.arcLength(marker_contour, True)
#         approx = cv2.approxPolyDP(marker_contour, epsilon, True)

#         # Draw filled contour (shaded overlay) on the overlay image
#         cv2.drawContours(overlay, [approx], 0, (0, 255, 0), -1)

#         # Draw the green border on the original frame for crisp edges
#         rect = cv2.minAreaRect(approx)
#         box = cv2.boxPoints(rect)
#         box = np.int_(box)
#         cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

#     # Blend the overlay with the original frame for translucent shading
#     cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

#     # ---- Second loop: Draw marker ID text ----
#     for i in range(len(bboxs)):
#         marker_corners = bboxs[i][0]
#         marker_id = ids[i][0]
#         text = str(marker_id)

#         # Find anchor: top-right is defined as the minimum x and minimum y among the corners
#         xs = marker_corners[:, 0]
#         ys = marker_corners[:, 1]
#         anchor_x = int(np.min(xs))
#         anchor_y = int(np.min(ys))

#         # Compute text size and baseline
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         thickness = 2
#         (text_width, text_height), baseline = cv2.getTextSize(text, font, FONT_SCALE, thickness)

#         # Place text so its top-right corner is at (anchor_x, anchor_y)
#         text_x = anchor_x - text_width  # text extends to the left
#         text_y = anchor_y + text_height  # text extends downward

#         # Clamp if necessary so text doesn't go off-frame
#         if text_x < 0:
#             text_x = 0
#         if text_y > frame.shape[0]:
#             text_y = frame.shape[0]

#         cv2.putText(frame, text, (text_x, text_y),
#                     font, FONT_SCALE, (255, 255, 255), thickness, cv2.LINE_AA)

#     # ---- Draw red danger zone ----
#     height, width = frame.shape[:2]
#     line_y = int(height * 0.7)
#     cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 2)

#     # Add a red tint to the remaining 30% region (from line_y to bottom)
#     if redtint:
#         red_color = (0, 0, 255)  # Red in BGR
#         # Create an overlay for the bottom region
#         bottom_region = frame[line_y:height, :]
#         red_overlay = bottom_region.copy()
#         red_overlay[:] = red_color
#         # Blend the red overlay with the bottom region
#         tinted_region = cv2.addWeighted(red_overlay, tint_alpha, bottom_region, 1 - tint_alpha, 0)
#         frame[line_y:height, :] = tinted_region
    
#     return frame

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

            # Always draw danger zone
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