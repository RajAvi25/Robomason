# explanations.py
 
explanations = {
    "Start Coordinate": "The start coordinate is the robot's end effector's global (absolute) position when the task begins.",
    "End Coordinate": "The end coordinate is the global position of the robot’s end effector when it completes the task.",
    "Start Time": "The start time is recorded as the moment (in seconds) when a task begins relative to the initial timestamp, which is considered t = 0s.",
    "End Time": "The end time marks when a robotic task is completed, measured from the start of execution.",
    "Elapsed Time": "The elapsed time for a motion segment is simply the difference between the end time and the start time.",
    "Average Velocity": "The average velocity represents the speed of the robot’s movement along its trajectory.",
    "Average Acceleration": "The average acceleration is calculated based on the change in velocity over time.",
    "Traversed Length": "The traversed length is the total Euclidean distance the robot’s end effector moves along its trajectory. It considers every movement from the start to the end of the task.",
    "Path Efficiency": "Path efficiency measures how direct the robot's movement is compared to the shortest path. It is expressed as a percentage.",
    "Path Efficiency Formula": r"$$ Path Efficiency = \left(\frac{\text{Straight-line distance}}{\text{Traversed length}}\right) \times 100 $$",
    "Average Curvature": "The average curvature quantifies the deviation of the robot’s trajectory from a straight line. It is calculated by measuring the change in direction (angle) between consecutive trajectory segments and dividing it by the total traversed length.",
    "Average Curvature Formula": r"$$ Average Curvature = \frac{\sum \Delta angle}{\text{Traversed length}} $$"
}
 