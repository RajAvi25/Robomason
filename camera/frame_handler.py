# camera/frame_handler.py
import cv2
import base64
import json
import threading
import time
import zmq
import websocket
import numpy as np

class FrameHandler:
    def __init__(self, ws_url, camera_index=0, topic="/image", frame_rate=30, is_sender=True):
        self.ws_url = ws_url
        self.camera_index = camera_index
        self.topic = topic
        self.frame_rate = frame_rate
        self.is_sender = is_sender
        self.lock = threading.Lock()
        self.frame = None
        self.stopped = False
        self.ws = None

        if self.is_sender:
            self.capture = cv2.VideoCapture(self.camera_index)
            self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not self.capture.isOpened():
                print(f"Error: Could not open video device at index {self.camera_index}.")
                return
            self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
            self.capture_thread.start()
        else:
            self.capture = None
            self.capture_thread = None

    def _capture_frames(self):
        while not self.stopped:
            if self.capture.isOpened():
                ret, frame = self.capture.read()
                if not ret:
                    print("Error: Cannot read from camera stream.")
                    break
                with self.lock:
                    self.frame = frame
            time.sleep(0.01)
        if self.capture:
            self.capture.release()

    def _connect_websocket(self):
        try:
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_open=self._on_open,
                on_message=self._on_message if not self.is_sender else None,
                on_error=self._on_error,
                on_close=self._on_close
            )
            ws_thread = threading.Thread(target=self.ws.run_forever, daemon=True)
            ws_thread.start()
            print(f"[FrameHandler] Connecting to {self.ws_url} ...")
        except Exception as e:
            print(f"[FrameHandler] WebSocket connection error: {e}")
            self.stop()

    def _on_open(self, ws):
        print("[FrameHandler] WebSocket connection opened.")
        if not self.is_sender:
            subscribe_msg = {"op": "subscribe", "topic": self.topic, "qos": {"durability": "volatile", "reliability": "reliable", "depth": 10}}
            ws.send(json.dumps(subscribe_msg))
            print(f"[FrameHandler] Subscribed to topic: {self.topic}")

    def _on_message(self, ws, message):
        data = json.loads(message)
        if 'msg' in data and 'data' in data['msg']:
            img_base64 = data['msg']['data']
            img_bytes = base64.b64decode(img_base64)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is not None:
                with self.lock:
                    self.frame = img
            else:
                print("Error: Failed to decode image from received message.")

    def _on_error(self, ws, error):
        print(f"[FrameHandler] WebSocket error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        print("[FrameHandler] WebSocket connection closed.")

    def _read_frame(self):
        with self.lock:
            return self.frame

    def _send_frame(self, frame):
        if frame is None:
            return
        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            print("Error: Frame encoding failed.")
            return
        img_str = base64.b64encode(buffer).decode('utf-8')
        msg = {"op": "publish", "topic": self.topic, "msg": {"data": img_str}}
        try:
            self.ws.send(json.dumps(msg))
        except Exception as e:
            print(f"Error sending frame: {e}")

    def start_streaming(self):
        self._connect_websocket()
        if self.is_sender:
            try:
                while not self.stopped:
                    frame = self._read_frame()
                    if frame is not None and self.ws:
                        self._send_frame(frame)
                    time.sleep(1 / float(self.frame_rate))
            except KeyboardInterrupt:
                print("[FrameHandler] Publisher interrupted by user.")
            finally:
                self.stop()
        else:
            try:
                while not self.stopped:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("[FrameHandler] Subscriber interrupted by user.")
            finally:
                self.stop()

    def get_latest_frame(self):
        return self._read_frame()

    def stop(self):
        self.stopped = True
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join()
        if self.ws:
            self.ws.close()
        print("[FrameHandler] Stopped and resources released.")

def cam_init():
    ws_url = "ws://localhost:9090"
    subscriber = FrameHandler(ws_url=ws_url, is_sender=False)
    sub_thread = threading.Thread(target=subscriber.start_streaming, daemon=True)
    sub_thread.start()
    # Import the marker trigger function from robot_controller (or ui, whichever is applicable)
    from robot_controller.marker_trigger import markerTrigger_30
    markerTrigger_30(subscriber)
    subscriber.stop()
    if sub_thread.is_alive():
        sub_thread.join()
