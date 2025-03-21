# plotting/run.py
import sys
import multiprocessing as mp
from PyQt5.QtWidgets import QApplication
from .main_window import MainWindow
from .receiver import zmq_receiver

if __name__ == "__main__":
    mp.set_start_method('spawn')
    data_queue = mp.Queue()
    receiver_process = mp.Process(target=zmq_receiver, args=(data_queue,))
    receiver_process.start()

    app = QApplication(sys.argv)
    window = MainWindow(data_queue)

    def on_exit():
        receiver_process.terminate()
        receiver_process.join()

    app.aboutToQuit.connect(on_exit)
    window.show()
    sys.exit(app.exec_())
