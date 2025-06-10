import sys
import multiprocessing as mp
from multiprocessing import Barrier
from plotting.worker import plot_view_worker
from plotting.gui import MainWindow
from plotting.worker import zmq_receiver, data_dispatcher_func
from PyQt5.QtWidgets import QApplication

if __name__ == '__main__':
    mp.set_start_method('spawn')

    zmq_data_queue     = mp.Queue()
    views              = ['3d','top','front','side']
    plot_input_queues  = {v: mp.Queue() for v in views}
    plot_output_queues = {v: mp.Queue() for v in views}

    #One Barrier for the 4 worker processes
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
    window = MainWindow(zmq_data_queue, plot_input_queues, plot_output_queues,data_dispatcher_func)
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