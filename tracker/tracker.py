import json
import time
import os
import threading
from .communication import CommunicationHandler
from .localization import Localizer
from .utils import Utils

def tracker_node():
    base_path = "/home/avi/Desktop/Robot/random_detection"
    next_folder = Utils.get_next_folder(base_path)
    os.makedirs(next_folder, exist_ok=True)
    
    comms = CommunicationHandler()
    tracking_thread = threading.Thread(target=comms.listen_for_tracking_commands)
    tracking_thread.daemon = True
    tracking_thread.start()
    
    print("Tracker node waiting for messages...")
    
    while True:
        try:
            with comms.tracking_lock:
                if comms.is_tracking:
                    message = comms.receive_message()
                    if message is None:
                        print('None received')
                        continue
                    data_in = json.loads(message)
                    # print(f'message from tracking {data_in}')
                    img_array, markers = Localizer.process_image(message, next_folder)
                    worker_spotted_status = False
                    worker_coordinates = None
                    worker_id = None
                    if markers is not None:
                        worker_spotted_status = True
                        for (w_coords, marker_id) in markers:
                            print('Marker/s detected')
                            worker_id = int(marker_id[0])
                            worker_coordinates = w_coords
                    else:
                        print("No detections")
                    
                    coords = data_in.get("coordinates", {})
                    formatted_coords = [
                        round(coords.get("x", 0), 3),
                        round(coords.get("y", 0), 3),
                        round(coords.get("z", 0), 3)
                    ]
                    
                    data_out = {
                        "coordinates": formatted_coords,
                        "element": data_in.get("element"),
                        "state": data_in.get("state"),
                        "joints": data_in.get("joints"),
                        "orientation": data_in.get("orientation"),
                        "worker spotted": worker_spotted_status,
                        "timestamp_send": time.time(),
                        "obstacle_maneuver":data_in.get("obstacle_maneuver")
                    }
                    if worker_spotted_status and worker_coordinates is not None:
                        data_out["worker coordinates"] = worker_coordinates
                        data_out["worker id"] = worker_id
                    
                    comms.publish_message(data_out)
        except Exception as e:
            print(f"Error in tracker node: {e}")

if __name__ == "__main__":
    tracker_node()
