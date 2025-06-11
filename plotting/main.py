#plotting/main.py
import math
import psutil
import multiprocessing as mp
import sys
import os
import time
from multiprocessing import Barrier
from plotting.worker import plot_view_worker
from plotting.gui import MainWindow
from plotting.worker import zmq_receiver, data_dispatcher_func
from PyQt5.QtWidgets import QApplication

import math
import numpy as np
from numba import njit

# 1) CPU‐affinity & priority for THIS process
p = psutil.Process()
p.cpu_affinity([4,5,6,7])       # choose any idle cores
try:
    os.nice(-5)                 # negative nice → higher priority
except PermissionError:
    pass
# ————————————————————————————————

# ————————————————————————————————
# 2) Dispatch‐tuning knobs
REFRESH_INTERVAL = 10.0   # secs between allowed redraws
POINT_BATCH      = 100   # max points before forcing a redraw
EPS              = 0.02  # RDP epsilon for decimation
# ————————————————————————————————

# def rdp(pts, eps=EPS):
#     """Ramer–Douglas–Peucker decimation (keeps only the 'corners')."""
#     if len(pts) < 3:
#         return pts
#     a, b = pts[0], pts[-1]
#     idx, maxd = 0, 0.0
#     for i, p in enumerate(pts[1:-1], start=1):
#         num = abs((b[1]-a[1])*p[0] - (b[0]-a[0])*p[1] + b[0]*a[1] - b[1]*a[0])
#         den = math.hypot(b[1]-a[1], b[0]-a[0])
#         d   = num/den if den else 0.0
#         if d > maxd:
#             idx, maxd = i, d
#     if maxd > eps:
#         left  = rdp(pts[:idx+1], eps)
#         right = rdp(pts[idx:],   eps)
#         return left[:-1] + right
#     return [a, b]

@njit
def _perp_dist_2d(x0, y0, x1, y1, x2, y2):
    num = abs((y2 - y1)*x0 - (x2 - x1)*y0 + x2*y1 - y2*x1)
    den = math.hypot(y2 - y1, x2 - x1)
    return num/den if den > 0 else 0.0

@njit
def rdp_numba(pts, eps):
    n = pts.shape[0]
    if n < 3:
        return pts
    # endpoints
    x1, y1 = pts[0,0], pts[0,1]
    x2, y2 = pts[-1,0], pts[-1,1]
    idx, maxd = 0, 0.0
    for i in range(1, n-1):
        x0, y0 = pts[i,0], pts[i,1]
        d = _perp_dist_2d(x0, y0, x1, y1, x2, y2)
        if d > maxd:
            idx, maxd = i, d
    if maxd > eps:
        left  = rdp_numba(pts[:idx+1], eps)
        right = rdp_numba(pts[idx:],   eps)
        out = np.empty((left.shape[0] + right.shape[0] - 1, 2), dtype=np.float64)
        out[:left.shape[0],:] = left
        out[left.shape[0]:,:] = right[1:,:]
        return out
    else:
        # just endpoints
        out = np.empty((2,2), dtype=np.float64)
        out[0,:] = pts[0,:]
        out[1,:] = pts[-1,:]
        return out

def throttled_dispatcher(zmq_queue, plot_input_queues, plot_output_queues):
    """
    Reads raw points from zmq_queue, buffers them, RDP‐decimates,
    and fans out to each plot_input_queue only once per batch/time.
    """
    buffer    = []
    last_draw = time.time()

    while True:
        data = zmq_queue.get()  # blocking
        buffer.append((data['x'], data['y'], data['z']))
        now = time.time()

        if (now - last_draw) >= REFRESH_INTERVAL or len(buffer) >= POINT_BATCH:
            decimated = rdp_numba(buffer)
            for x, y, z in decimated:
                pkt = {'x': x, 'y': y, 'z': z}
                for q in plot_input_queues.values():
                    q.put(pkt)
            buffer.clear()
            last_draw = now

if __name__ == '__main__':
    mp.set_start_method('spawn')

    zmq_data_queue     = mp.Queue()
    views              = ['3d','top','front','side']
    plot_input_queues  = {v: mp.Queue() for v in views}
    plot_output_queues = {v: mp.Queue() for v in views}
    sync_barrier = mp.Barrier(len(views), timeout=5)   # 5-s safety timeout

    # start ZMQ receiver
    receiver = mp.Process(target=zmq_receiver, args=(zmq_data_queue,))
    receiver.start()

    # start plot workers
    workers = []
    for v in views:
        p = mp.Process(
            target=plot_view_worker,
            args=(plot_input_queues[v], 
                  plot_output_queues[v], 
                  v,
                  sync_barrier)
        )
        p.start()
        workers.append(p)

    # launch Qt app
    app    = QApplication(sys.argv)
    window = MainWindow(
        zmq_data_queue, 
        plot_input_queues, 
        plot_output_queues,
        throttled_dispatcher
        # data_dispatcher_func
        )
    window.show()

    def cleanup():
        receiver.terminate()
        for p in workers:
            p.terminate()
        receiver.join()
        for p in workers:
            p.join()

    app.aboutToQuit.connect(cleanup)
    sys.exit(app.exec_())