#camera/live_camera.py
import threading
from .frame_handler import FrameHandler
from .annotate import liveFeedAruco, transmitFeed

def main():
    ws_url = "ws://localhost:9090"   
    camera_index = 4                
    frame_rate = 30         

    # Create a FrameHandler in publisher (sender) mode
    publisher = FrameHandler(
        ws_url=ws_url,
        camera_index=camera_index,
        frame_rate=frame_rate,
        is_sender=True
    )
    
    camera_thread = threading.Thread(target=liveFeedAruco, args=(publisher,)) #Run this line for general camera feed display
    # camera_thread = threading.Thread(target=transmitFeed, args=(publisher,))  #Run this line for integrated camera feed display
    camera_thread.daemon = True  
    camera_thread.start()

    # Begin streaming frames over WebSocket
    publisher.start_streaming()

if __name__ == "__main__":
    main()
