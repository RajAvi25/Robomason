import json
import time
import threading
import zmq

class CommunicationHandler:
    def __init__(self):
        self.context = zmq.Context()
        self.subscriber_socket = self.context.socket(zmq.SUB)
        self.subscriber_socket.connect("tcp://127.0.0.1:5550")
        self.subscriber_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        
        self.publisher_socket = self.context.socket(zmq.PUB)
        self.publisher_socket.bind("tcp://127.0.0.1:5552")
        
        self.command_socket = self.context.socket(zmq.SUB)
        self.command_socket.connect("tcp://127.0.0.1:5551")
        self.command_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        
        self.is_tracking = False
        self.tracking_lock = threading.Lock()

    def listen_for_tracking_commands(self):
        while True:
            try:
                message = self.command_socket.recv_string()
                print(f"Second port message: {message}")
                with self.tracking_lock:
                    if message == 'start_tracking':
                        self.is_tracking = True
                        print('Tracking started')
                    elif message == 'stop_tracking':
                        self.is_tracking = False
                        print('Tracking stopped')
                    else:
                        print(f"Unknown message: {message}")
            except Exception as e:
                print(f"Error on second port: {e}")

    def receive_message(self):
        try:
            return self.subscriber_socket.recv_string()
        except Exception as e:
            print(f"Error receiving message: {e}")
            return None

    def publish_message(self, data):
        try:
            self.publisher_socket.send_string(json.dumps(data))
        except Exception as e:
            print(f"Error publishing message: {e}")
