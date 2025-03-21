# Robotic Construction and Worker Detection System

This project combines robotic construction automation with real-time worker detection and safety mechanisms. The system visualizes robot trajectories, detects workers using camera systems, computes accurate world coordinates from image frames, and logs detailed task and safety data.

---

## Project Overview

This system is composed of:

- A UR5e robot for construction automation.
- A camera-based worker detection system using ArUco markers.
- Data pipelines for visualization, analysis, and safety management.
- Multi-phase construction cycle: Scanning → Assembly → Disassembly → Reassembly.

---

## Features

### 1. Trajectory Visualization
- Plots from multiple angles: plan, front, side, and orthogonal views.
- Displays historical and real-time robot joint positions.
- Thick lines represent forward motion; thin lines represent return.
- Color-coded by construction element.
- CEP50 plots to analyze path dispersion.

### 2. Construction Process
The construction process is structured into four phases:
1. Scanning
2. Assembly
3. Disassembly
4. Reassembly

Each phase utilizes two predefined ArUco marker locations for worker simulation and safety analysis.
---
### 4. Worker Detection
- Uses ArUco markers to identify and track workers.
- Evaluates detection accuracy (target error: 1–2 cm).
- FOV (field of view) analysis of visible/non-visible zones.

Worker locations are calculated using:
- Image center offset
- Pixels-per-meter scaling
- Known camera height and orientation (downward-facing)
- Ground plane assumption (Z=0)

---

## 4. Obstacle Avoidance and Safety

- If a worker appears in the **bottom 30% of the camera’s field of view**, the robot performs a detour (inward movement by 10 cm).
- Safe joint positions are saved to protect the robot after daily operation.
- Avoidance motion is logged along with trajectory and detection data.

---

## 5. Data Collection and Post-Processing

### Construction Data Table

| Task ID | Task Description | Start Time | End Time | Elapsed Time | Trajectory Length | Avg Velocity | Avg Acceleration | Path Efficiency | Avg Curvature |

### Worker Detection Table

| Worker ID | Instance | Timestamp | X | Y | Z |

- Logs include curvature, velocity, path efficiency, etc.
- Used to assess system responsiveness and detect failures or inefficiencies.

---

## 6. Camera Field of View (FOV) Analysis

- **Camera datasheet values:**
  - Horizontal FOV: 65 ±2°
  - Vertical FOV: 40 ±1°
- Measured values across multiple tests:
  - Average horizontal: 60.32°
  - Average vertical: 45.3°
- Discrepancies visualized; intersection of FOV cone and ground plane plotted.

---

## 📁 Repository Structure

| Folder/File | Description |
|------------|-------------|
| `robomason/` | Main project folder |
| ├── `UserWindow.ipynb` | Jupyter notebook for UI calibration |
| ├── `system_config.py` | System configuration file |
| ├── `ifc/` | IFC file handling module |
| │ ├── `IFC_functions.py` | Functions related to IFC processing |
| │ ├── `ifc_loader.py` | IFC file loader script |
| ├── `_workingdata/` | Contains all experiment runs, trajectory data, and visualizations |
| │ ├── `_disassemblyruns/` | Data from disassembly experiments |
| │ │ ├── `Disassembly_live_camera.mp4` | Recorded video from live camera |
| │ │ ├── `trajectory_summary.xlsx` | Summary of recorded trajectories |
| │ ├── `_constructionruns/` | Data from construction experiments |
| │ ├── `_reconstructionruns/` | Data from reconstruction experiments |
| │ ├── `_rawtrajectories/` | Raw trajectory files |
| │ ├── `IFC/` | IFC project files |
| │ ├── `_siteinfo/` | Saved site positions |
| ├── `ui/` | User interface modules |
| ├── `camera/` | Camera-related processing code |
| ├── `construct/` | Construction process logic |
| ├── `robot_controller/` | Robot control logic |
| ├── `plotting/` | Plotting and visualization utilities |
| ├── `detections/` | Object and worker detection code |
| ├── `tracker/` | Tracking algorithms and localization |


