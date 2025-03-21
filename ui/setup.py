# ui/setup.py
from ui.comms import (
    connectRobotserver,
    connectPlottingserver,
    connectTrackerserver,
    initCameraHandler,
    initTracker,
    initPlotting
)

from . import MarkerDetectionLocalization as mdl

def system_setup():
    """
    Sets up all necessary connections:
      - Connects to the robot server.
      - Connects to the plotting server and initializes plotting.
      - Connects to the tracker server and initializes tracker.
      - Initializes the camera handler.
    
    Parameters:
      connect_plotting (bool): If True, calls connectPlottingserver() and initPlotting().
      connect_tracker (bool): If True, calls connectTrackerserver() and initTracker().
    """
    connectRobotserver()
    connectPlottingserver()
 
    connectTrackerserver()
    initCameraHandler()
    mdl.start_robot_data_listener()
    initTracker()
    initPlotting()
