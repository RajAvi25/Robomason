# plotting/zmq_receiver.py
import zmq
import msgpack
from .config import ZMQ_ADDRESS, SUBSCRIBE_TOPIC, PACKET_SKIP

# Use a global counter if needed
packet_counter = 0

def zmq_receiver(queue):
    """
    ZMQ receiver function.
    Receives data, unpacks it with msgpack, and puts it into a multiprocessing Queue.
    """
    global packet_counter
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(ZMQ_ADDRESS)
    socket.setsockopt_string(zmq.SUBSCRIBE, SUBSCRIBE_TOPIC)
    socket.setsockopt(zmq.RCVTIMEO, 1000)

    while True:
        try:
            packed_data = socket.recv(flags=0)
            packet_counter += 1
            if packet_counter % PACKET_SKIP == 0:
                data = msgpack.unpackb(packed_data, raw=False)
                queue.put(data)
        except zmq.Again:
            continue
        except zmq.ZMQError:
            break
