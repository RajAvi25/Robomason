# Robotic Construction and Worker Detection System

This project focuses on autonomous robotic construction with integrated real-time safety and worker detection. It includes multi-angle trajectory visualization, worker detection using ArUco markers, obstacle avoidance, data logging, and construction task automation.

---

## Project Overview

This system has three key capabilities:
- Robotic automation of construction tasks using a UR5e robot.
- Real-time worker detection using computer vision and ArUco markers.
- Multi-angle visualization and in-depth analysis of robot motion and safety behavior.

---

## Features

### 1. Trajectory Visualization
- Orthogonal, plan, front, and side view plots.
- Real-time rendering of the robot.
- Color-coded lines for different construction elements.
- CEP50 plots for evaluating dispersion in trajectory data.
- Forward paths use thick lines; return paths use thin lines.

### 2. Data Collection and Analysis
- Logs construction task metrics such as:
  - Task ID
  - Timestamps (start, stop)
  - Elapsed time
  - Trajectory length
  - Velocity, acceleration
  - Path efficiency and curvature
- Logs worker detection data:
  - Worker ID and instance
  - Timestamp
  - 3D position
- Annotated tables generated in post-processing.
- Path and curvature analysis to assess construction accuracy.

### 3. Construction Automation
- Supports assembly, disassembly, and reassembly of elements.
- Handles positional adjustments (e.g., raising modules by 2mm).
- Disassembly site relocated behind the robot for efficiency.

### 4. Worker Detection
- Uses ArUco markers to identify and track workers.
- Evaluates detection accuracy (target error: 1–2 cm).
- FOV (field of view) analysis of visible/non-visible zones.

### 5. Obstacle Avoidance and Safety
- Robot alters path by 10 cm inward upon worker detection.
- Triggers avoidance when worker appears in the bottom 30% of the FOV.
- Identifies a safe joint configuration for robot shutdown.

---

## Data Format

### Construction Data Table

| Task ID | Task Description | Start Time | End Time | Elapsed Time | Trajectory Length | Avg Velocity | Avg Acceleration | Path Efficiency | Avg Curvature |
|---------|------------------|------------|----------|---------------|-------------------|--------------|------------------|------------------|----------------|

### Worker Detection Table

| Worker ID | Instance | Timestamp | X | Y | Z |
|-----------|----------|-----------|---|---|---|

---

## Experimental Notes

- **Camera FOV Analysis:** Measurements conducted to compare actual FOV with datasheet specs.
- **FOV Calculations:** Measured FOVs averaged ~60.3° (horizontal) and ~45.3° (vertical) versus expected values of 65° ±2 and 40° ±1.
- **Worker Detection Mapping:** Includes plan and isometric plots of detected positions.

---

## Technologies

- Python
- OpenCV (for ArUco marker detection)
- Matplotlib / Plotly (for data visualization)
- UR5e Robot API
- OnRobot Eye Camera

---

## License

MIT License. See the `LICENSE` file for details.
