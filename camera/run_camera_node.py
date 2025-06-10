import threading
import subprocess
import time
import sys
from frame_handler import FrameHandler
from annotate import liveFeedAruco  # or use transmitFeed

def main():
    ws_url = "ws://localhost:9090"
    camera_index = 4
    frame_rate = 30

    # Start rosbridge server as subprocess
    rosbridge_process = None
    try:
        print("Launching rosbridge server...")
        rosbridge_process = subprocess.Popen([
            'ros2', 'launch', 'rosbridge_server', 'rosbridge_websocket_launch.xml'
        ])
        time.sleep(3)  # Give rosbridge time to come up
        print("rosbridge launched.")

        # Start camera FrameHandler
        publisher = FrameHandler(
            ws_url=ws_url,
            camera_index=camera_index,
            frame_rate=frame_rate,
            is_sender=True
        )

        # Start annotated camera feed in a separate thread
        camera_thread = threading.Thread(target=liveFeedAruco, args=(publisher,))
        camera_thread.daemon = True
        camera_thread.start()

        # Start frame publishing
        publisher.start_streaming()

    except KeyboardInterrupt:
        print("\n Interrupted by user.")
    finally:
        if rosbridge_process:
            print("Shutting down rosbridge server...")
            rosbridge_process.terminate()
            rosbridge_process.wait()
            print(" rosbridge stopped.")

if __name__ == "__main__":
    main()
