#plotting/gui.py
from PyQt5.QtWidgets import QMainWindow, QLabel, QWidget, QGridLayout, QScrollArea, QHBoxLayout, QVBoxLayout, QSizePolicy
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from .utils import *

from PyQt5.QtWidgets import (
    QMainWindow, QLabel, QWidget,
    QGridLayout, QScrollArea,
    QHBoxLayout, QVBoxLayout,
    QSizePolicy
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.lines import Line2D
from io import BytesIO

from .utils import *

class MainWindow(QMainWindow):
    """
    Main application window for real-time trajectory views.

    **Initialization Parameters:**
    - *zmq_data_queue*: Queue receiving raw tracking JSON packets from the ZeroMQ subscriber.
    - *plot_input_queues*: Dict mapping view names ("3d","top","front","side") to queues
      where raw packets are sent for rendering.
    - *plot_output_queues*: Dict mapping view names to queues where rendered image data
      (PNG bytes or raw RGBA) is received.
    - *data_dispatcher*: Optional dispatcher object (not used directly here).

    **Layout:**
    1. Status bar (QLabel) showing the current element and state.
    2. 2×2 grid of QLabel widgets for the four trajectory views.
    3. Scrollable legend panel displaying line‐style swatches for each element/state.

    **Behavior:**
    - A QTimer fires every 100 ms, triggering `_on_timer()`.
    - `_on_timer()` pulls all available packets from `zmq_data_queue`, forwards them to
      each plotting worker via `plot_input_queues`, then collects the latest rendered
      image from each `plot_output_queue` and updates the corresponding view label.
    - Updates the status bar text based on the metadata (element, state) of the most
      recently displayed view.
    """
    def __init__(
        self, zmq_data_queue,
        plot_input_queues, plot_output_queues,
        data_dispatcher
    ):
        super().__init__()
        self.setWindowTitle("Real-time Trajectory Views")
        self.resize(int(1400 * SCALING_FACTOR), int(800 * SCALING_FACTOR))

        # keep references
        self.zmq_data_queue     = zmq_data_queue
        self.plot_input_queues  = plot_input_queues
        self.plot_output_queues = plot_output_queues
        self.data_dispatcher    = data_dispatcher

        # CENTRAL WIDGET + LAYOUT
        central = QWidget(self)
        self.setCentralWidget(central)
        main_v = QVBoxLayout(central)

        # 1) Status bar at top
        self.status_label = QLabel("Waiting for data…")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(
            "font-size: 18px; padding: 6px; background: #333; color: white;"
        )
        main_v.addWidget(self.status_label)

        # 2) Plot grid + legend side-by-side
        body_h = QHBoxLayout()
        main_v.addLayout(body_h)

        # 2a) 2×2 grid of views
        self.plot_widget = QWidget()
        grid = QGridLayout(self.plot_widget)
        self.labels = {}
        for pos, view in zip(
            [(0,0),(0,1),(1,0),(1,1)],
            ['3d','top','front','side']
        ):
            lbl = QLabel()
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            lbl.setStyleSheet("background-color: lightgray;")
            grid.addWidget(lbl, *pos)
            self.labels[view] = lbl
        body_h.addWidget(self.plot_widget, 4)

        # 2b) Legend on the right
        self.legend_area   = QScrollArea()
        self.legend_area.setWidgetResizable(True)
        self.legend_widget = QWidget()
        self.legend_area.setWidget(self.legend_widget)
        v_leg_layout = QVBoxLayout(self.legend_widget)

        # Legend Title
        title = QLabel("<b>Legend</b>")
        title.setStyleSheet("font-size: 16px; margin-bottom: 10px;")
        v_leg_layout.addWidget(title)

        for element, color in sorted(ELEMENT_COLORS.items()):
            if element.lower() == "scanning site":
                h = QHBoxLayout()
                fig = Figure(figsize=(1*SCALING_FACTOR, 0.2*SCALING_FACTOR))
                ax  = fig.add_axes([0,0,1,1])
                style = SCANNING_STYLE
                line  = Line2D([0,1],[0.5,0.5],
                               color=color,
                               linestyle=style["linestyle"],
                               linewidth=style["linewidth"],
                               alpha=0.75)
                ax.add_line(line); ax.axis("off")
                canvas = FigureCanvasAgg(fig); canvas.draw()
                buf    = BytesIO(); fig.savefig(buf, format="png"); buf.seek(0)
                img_lbl = QLabel();
                pix = QPixmap.fromImage(QImage.fromData(buf.getvalue()))
                img_lbl.setPixmap(pix.scaled(
                    int(50*SCALING_FACTOR), int(20*SCALING_FACTOR),
                    Qt.KeepAspectRatio
                ))
                txt_lbl = QLabel(f"{element} (scanning)")
                txt_lbl.setStyleSheet("font-size: 14px; margin-left: 5px;")
                h.addWidget(img_lbl); h.addWidget(txt_lbl); h.addStretch()
                v_leg_layout.addLayout(h)
            else:
                for state in ALLOWED_STATES_ASSEMBLY:
                    h = QHBoxLayout()
                    fig = Figure(figsize=(1*SCALING_FACTOR, 0.2*SCALING_FACTOR))
                    ax  = fig.add_axes([0,0,1,1])
                    style = STATE_STYLES_ASSEMBLY[state]
                    line  = Line2D([0,1],[0.5,0.5],
                                   color=color,
                                   linestyle=style["linestyle"],
                                   linewidth=style["linewidth"],
                                   alpha=0.75)
                    ax.add_line(line); ax.axis("off")
                    canvas = FigureCanvasAgg(fig); canvas.draw()
                    buf    = BytesIO(); fig.savefig(buf, format="png"); buf.seek(0)
                    img_lbl = QLabel();
                    pix = QPixmap.fromImage(QImage.fromData(buf.getvalue()))
                    img_lbl.setPixmap(pix.scaled(
                        int(50*SCALING_FACTOR), int(20*SCALING_FACTOR),
                        Qt.KeepAspectRatio
                    ))
                    txt_lbl = QLabel(f"{element} ({state})")
                    txt_lbl.setStyleSheet("font-size: 14px; margin-left: 5px;")
                    h.addWidget(img_lbl); h.addWidget(txt_lbl); h.addStretch()
                    v_leg_layout.addLayout(h)

        v_leg_layout.addStretch()
        body_h.addWidget(self.legend_area, 1)

        # TIMER
        self.timer = QTimer()
        self.timer.timeout.connect(self._on_timer)
        self.timer.start(100)

    def _on_timer(self):
        """
        Periodic update callback.

        1. Drain all raw packets from `zmq_data_queue`, forwarding each to all
           `plot_input_queues` so the plotting workers receive the latest data.
        2. For each view, retrieve the most recent rendered image (PNG or raw RGBA)
           from its `plot_output_queue` and update the corresponding QLabel.
        3. Update the status bar text based on the last displayed element/state.
        """
        # 1) dispatch incoming data to workers…
        last_pkt = None
        while not self.zmq_data_queue.empty():
            pkt = self.zmq_data_queue.get_nowait()
            last_pkt = pkt
            for q in self.plot_input_queues.values():
                q.put(pkt)

        # 2) update each view & capture its metadata
        current_meta = None
        for view, q in self.plot_output_queues.items():
            # keep only the latest packet in this queue
            item = None
            while not q.empty():
                item = q.get_nowait()
            if item is None:
                continue

            element, state = item[0], item[1]

            # old PNG path: (element, state, png_bytes)
            if len(item) == 3:
                img_bytes = item[2]
                pix = QPixmap.fromImage(QImage.fromData(img_bytes))
            # new raw-RGBA path: (element, state, raw_bytes, w, h)
            else:
                raw_bytes, w, h = item[2], item[3], item[4]
                img = QImage(raw_bytes, w, h, QImage.Format_RGBA8888)
                pix = QPixmap.fromImage(img)

            self.labels[view].setPixmap(pix)
            current_meta = (element, state)

        # 3) update status bar from last shown frame
        if current_meta is not None:
            el, st = current_meta
            self.status_label.setText(f"Element: {el}   |   State: {st}")





