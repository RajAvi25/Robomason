import numpy as np
import pandas as pd
import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Rectangle
from construction_config import *

z_level = -0.155
x_limits = (-0.45, 0.71)
y_limits = (-0.35, 0.75)
z_limits = (z_level, 0.5)

line_styles = {
    "grab": "-",   # Solid line for grab
    "place": "--"  # Dashed line for place
}

line_widths = {
    "grab": 5.0,   # Thicker line for grab
    "place": 1.5   # Thinner line for place
}

line_alpha = {
    "grab": 1.0,   # Fully opaque for grab
    "place": 0.7   # Slightly transparent for place
}

TRANSLATION = np.array([0.02099, 0.34301, -0.30002])

ELEMENT_COLORS = {
    "Scanning site":        "deepskyblue",      
    "Foundation":           "mediumseagreen",   
    "Searching element":    "red",              
    "Wall 1":               "gold",           
    "Wall 2":               "darkorange",       
    "Floor 1":              "blueviolet",      
    "Floor 2":              "mediumorchid",    
    "Bathroom module 1":    "saddlebrown",     
    "Bathroom module 2":    "peru",             
    "Bathroom module 3":    "burlywood",        
}

display_robot = True

DH_params = [
    {'theta': 0, 'a': 0,       'd': 0.1625,  'alpha': np.pi / 2},
    {'theta': 0, 'a': -0.425,  'd': 0,       'alpha': 0},
    {'theta': 0, 'a': -0.3922, 'd': 0,       'alpha': 0},
    {'theta': 0, 'a': 0,       'd': 0.1333,  'alpha': np.pi / 2},
    {'theta': 0, 'a': 0,       'd': 0.0997,  'alpha': -np.pi / 2},
    {'theta': 0, 'a': 0,       'd': 0.0996,  'alpha': 0}
]

def calculate_euclidean_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return np.linalg.norm(np.array(point1) - np.array(point2))

def summarize_trajectory_metrics(pos_data):
    """
    Summarizes trajectory metrics by segment (each unique consecutive element+state).
    Skips any data packets where element or state is None.
    Returns a DataFrame with summary statistics.
    """
    def round_coordinates(coord):
        if isinstance(coord, list):
            return [round(val, 2) for val in coord]
        return coord

    # Define the construction phases with ArUco IDs
    construction_phases = {
        "Scanning site": '[0],[1],[2]',
        "Searching element": '-',
        "Foundation": '[11]',
        "Toilet": '-',
        "Wall": '[10]',
        "Floor": '[12]'
    }

    # Prepare lists for final DataFrame
    elements = []
    states = []
    start_coords = []
    end_coords = []
    start_times = []
    end_times = []
    traversed_lengths = []
    path_efficiencies = []
    curvatures = []
    aruco_ids = []
    elapsed_times = []
    avg_velocities = []
    avg_accelerations = []

    # If no data, return empty DF
    if not pos_data:
        return pd.DataFrame()

    # --- 1) Find the first valid data packet to initialize ---
    idx = 0
    while idx < len(pos_data):
        if (pos_data[idx]['element'] is not None) and (pos_data[idx]['state'] is not None):
            break
        idx += 1

    # If we never found a valid packet, return empty
    if idx == len(pos_data):
        return pd.DataFrame()

    # Initialize references for the first valid packet
    initial_timestamp = pos_data[idx]['timestamp_send']
    prev_element = pos_data[idx]['element'].strip()
    prev_state = pos_data[idx]['state'].strip()
    start_coord = pos_data[idx]['coordinates']
    start_time = pos_data[idx]['timestamp_send']
    prev_coordinates = start_coord
    prev_direction = None
    total_length = 0.0
    total_curvature = 0.0
    segment_count = 0
    start_velocity = 0.0

    # Helper function to finalize the current segment
    def finalize_segment(end_time, end_coord):
        """Append one row of metrics for the current segment."""
        elements.append(prev_element)
        states.append(prev_state)
        start_coords.append(round_coordinates(start_coord))
        end_coords.append(round_coordinates(end_coord))

        # Calculate times relative to the initial timestamp
        seg_start_s = start_time - initial_timestamp
        seg_end_s = end_time - initial_timestamp
        seg_elapsed = end_time - start_time

        start_times.append(round(seg_start_s, 2))
        end_times.append(round(seg_end_s, 2))
        elapsed_times.append(round(seg_elapsed, 2))

        # Store the total length
        total_dist = round(total_length, 2)
        traversed_lengths.append(total_dist)

        # Path efficiency
        straight_line_distance = calculate_euclidean_distance(start_coord, end_coord)
        path_eff = (straight_line_distance / total_length * 100.0) if total_length > 0 else 100.0
        path_efficiencies.append(round(path_eff, 2))

        # Average curvature
        avg_curv = total_curvature / segment_count if segment_count > 0 else 0.0
        curvatures.append(round(avg_curv, 2))

        # ArUco ID
        aruco_ids.append(construction_phases.get(prev_element, '-'))

        # Average velocity
        avg_vel = total_length / seg_elapsed if seg_elapsed > 0 else 0.0
        avg_velocities.append(round(avg_vel, 2))

        # Average acceleration
        avg_acc = (avg_vel - start_velocity) / seg_elapsed if seg_elapsed > 0 else 0.0
        avg_accelerations.append(round(avg_acc, 2))

        return avg_vel  # so we can update start_velocity

    # --- 2) Main loop from the next packet onwards ---
    i = idx + 1
    while i < len(pos_data):
        data = pos_data[i]
        i += 1

        # Skip if element or state is None
        if data['element'] is None or data['state'] is None:
            continue

        element = data['element'].strip()
        state = data['state'].strip()
        coordinates = data['coordinates']
        timestamp = data['timestamp_send']

        # Check if we are continuing the same segment or not
        if element == prev_element and state == prev_state:
            # Accumulate traversed length
            dist = calculate_euclidean_distance(prev_coordinates, coordinates)
            total_length += dist

            # Curvature
            current_direction = np.array(coordinates) - np.array(prev_coordinates)
            norm = np.linalg.norm(current_direction)
            if norm > 1e-9:
                current_direction /= norm
                if prev_direction is not None:
                    cosine_angle = np.clip(np.dot(prev_direction, current_direction), -1.0, 1.0)
                    angle_change = np.arccos(cosine_angle)
                    total_curvature += angle_change
                    segment_count += 1
                prev_direction = current_direction
        else:
            # We have a new (element, state). Finalize the old segment
            start_velocity = finalize_segment(timestamp, prev_coordinates)

            # Now reset everything for the new segment
            prev_element = element
            prev_state = state
            start_coord = coordinates
            start_time = timestamp
            total_length = 0.0
            total_curvature = 0.0
            segment_count = 0
            prev_direction = None

        # Update references
        prev_coordinates = coordinates

    # --- 3) Finalize the last segment (for the last valid packet) ---
    finalize_segment(pos_data[-1]['timestamp_send'], pos_data[-1]['coordinates'])

    # --- 4) Create the summary DataFrame ---
    summary = pd.DataFrame({
        'Element': elements,
        'State': states,
        'ArUco ID': aruco_ids,
        'Start [x,y,z]': start_coords,
        'End [x,y,z]': end_coords,
        'Start Time [s]': start_times,
        'End Time [s]': end_times,
        'Elapsed Time [s]': elapsed_times,
        'Average Velocity [m/s]': avg_velocities,
        'Average Acceleration [m/s²]': avg_accelerations,
        'Traversed Length [m]': traversed_lengths,
        'Path Efficiency [%]': path_efficiencies,
        'Average Curvature [radians]': curvatures,
    })

    # Replace states for printing
    summary['State'] = summary['State'].replace({'grab': 'pick', 'place': 'place'})
    summary['Element'] = summary['Element'].replace('Toilet', 'Bathroom module')

    # For scanning and searching, override state
    mask = summary['Element'].isin(['Scanning site', 'Searching element'])
    summary.loc[mask, 'State'] = '-'

    return summary

def workerdetection_analysis(data):
    # Filter out packets where worker is spotted and the 'element' (Construction Element) is not None
    filtered_data = [d for d in data if d.get('worker spotted')]
    filtered_data = [d for d in filtered_data if d.get('element') is not None]

    # 1. Compute the global start time (t0) as the minimum timestamp among packets with worker spotted.
    timestamps = [d['timestamp_send'] for d in filtered_data if d.get('worker spotted')]
    t0 = min(timestamps)

    # 2. Create a dictionary to store the first detection for each worker id.
    first_detections = {}
    for d in filtered_data:
        if d.get('worker spotted'):
            worker_id = d.get('worker id')
            ts = d.get('timestamp_send')
            if worker_id not in first_detections or ts < first_detections[worker_id]['timestamp_send']:
                first_detections[worker_id] = d

    # 3. Build a list of rows to create the DataFrame.
    rows = []
    for worker_id, detection in first_detections.items():
        # Compute relative time offset from the start time (t0)
        relative_time = detection['timestamp_send'] - t0

        # Get worker coordinates from the packet (assumed to be the worker location)
        worker_location = detection.get('worker coordinates')

        # Create a string showing the first detection location and the relative time (formatted to 2 decimal places)
        first_detected = f"{worker_location} ({relative_time:.2f})"

        # Get and modify Activity from the 'state' key.
        activity = detection.get('state')
        if activity:
            if activity == 'grab':
                activity = 'picked'
            elif activity == 'place':
                activity = 'placed'
        
        # Get and modify Construction Element from the 'element' key.
        construction = detection.get('element')
        if construction:
            if construction == 'Toilet':
                construction = 'Bathroom module'
            # If Construction Element is either of these, override activity with '-'
            if construction in ['Scanning site', 'Searching element']:
                activity = '-'

        rows.append({
            "worker id": worker_id,
            "worker location": worker_location,
            "first detected at location (time)": first_detected,
            "Activity": activity,
            "Construction Element": construction
        })

    # 4. Create the DataFrame.
    df = pd.DataFrame(rows)
    return df

class ElementResolver:
    def __init__(self):
        self.wall_count   = 0
        self.floor_count  = 0
        self.toilet_count = 0

    def resolve_element_name(self, raw_element):
        raw_element = raw_element.lower().strip()
        elements = {
            "foundation":       "Foundation",
            "scanning site":    "Scanning site",
            "searching element":"Searching element",
            "wall":             "Wall",
            "floor":            "Floor",
            "toilet":           "Bathroom module"
        }
        for key, label in elements.items():
            if key in raw_element:
                if key in ["scanning site", "searching element", "foundation"]:
                    return label
                if key == "wall":
                    self.wall_count = min(self.wall_count + 1, 2)
                    return f"{label} {self.wall_count}"
                if key == "floor":
                    self.floor_count = min(self.floor_count + 1, 2)
                    return f"{label} {self.floor_count}"
                if key == "toilet":
                    self.toilet_count = min(self.toilet_count + 1, 3)
                    return f"{label} {self.toilet_count}"
        return "default"
    
def transformation_matrix(a, alpha, d, theta):
    """Return the individual transformation matrix for each joint."""
    return np.array([
        [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
        [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
        [0,              np.sin(alpha),               np.cos(alpha),               d],
        [0,              0,                           0,                           1]
    ])

def forward_kinematics(joint_angles):
    """Compute all UR5 joint positions in 3D (end of each link)."""
    translation = np.array([0.02099, 0.34301, -0.30002])  # Same offset as your real-time code
    positions = []

    # Start: the base of joint0 is ~ (0,0,0), but for plotting we add the same translation
    # The user sometimes anchors the first link at z=DH_params[0]['d'], but let's
    # keep it consistent with your real-time code's approach.
    T = np.eye(4)
    for i, params in enumerate(DH_params):
        # For each joint, add the local transform
        theta = params['theta'] + joint_angles[i]
        T_joint = transformation_matrix(params['a'], params['alpha'], params['d'], theta)
        T = T @ T_joint
        # The x,y,z of this joint:
        x, y, z = T[0, 3], T[1, 3], T[2, 3]
        positions.append(np.array([x, y, z]) + translation)

    return positions

def add_environment_3d(ax):
    """Draw ground plane & environment boxes in 3D."""
    xx, yy = np.meshgrid(
        np.linspace(x_limits[0], x_limits[1], 10),
        np.linspace(y_limits[0], y_limits[1], 10)
    )
    zz = np.full_like(xx, z_level)
    ax.plot_surface(xx, yy, zz, color='peru', alpha=0.3)

    box_positions = [
        (-0.22, -0.13),  # pickup site
        (0.5802, 0.2528),    # bathroom module site
        (0.5802, 0.5669),    # construction site
        (0.242, -0.14)     # deconstruction site
    ]
    box_dims = [
        (29.5/100, 21/100),
        (21/100, 29.5/100),
        (21/100, 29.5/100),
        (45/100, 29.5/100)
    ]
    box_color = 'darkblue'
    box_alpha = 0.5

    for (x_center, y_center), dims in zip(box_positions, box_dims):
        width, height = dims
        rect_xs = [
            x_center - width/2,
            x_center + width/2,
            x_center + width/2,
            x_center - width/2,
            x_center - width/2
        ]
        rect_ys = [
            y_center - height/2,
            y_center - height/2,
            y_center + height/2,
            y_center + height/2,
            y_center - height/2
        ]
        rect_zs = [z_level]*5
        verts = [list(zip(rect_xs, rect_ys, rect_zs))]
        poly = Poly3DCollection(verts, facecolors=box_color, alpha=box_alpha, edgecolors='k')
        ax.add_collection3d(poly)

def add_environment_2d_top(ax):
    """Draw ground rectangle & boxes in top view (X vs. Y)."""
    ground_width  = x_limits[1] - x_limits[0]
    ground_height = y_limits[1] - y_limits[0]
    ax.add_patch(
        plt.Rectangle((x_limits[0], y_limits[0]), ground_width, ground_height,
                      facecolor='peru', alpha=0.3)
    )

    box_positions = [
        (-0.22, -0.13),  # pickup site
        (0.5802, 0.2528),    # bathroom module site  #Scotty
        (0.5802, 0.5669),    # construction site
        (0.242, -0.14)     # deconstruction site
    ]

    box_dims = [
        (29.5/100, 21/100),
        (21/100, 29.5/100),
        (21/100, 29.5/100),
        (45/100, 29.5/100)
    ]
    box_color = 'darkblue'
    box_alpha = 0.5

    for (x_center, y_center), dims in zip(box_positions, box_dims):
        w, h = dims
        x_left   = x_center - w/2
        y_bottom = y_center - h/2
        ax.add_patch(
            plt.Rectangle((x_left, y_bottom), w, h,
                          facecolor=box_color, alpha=box_alpha, edgecolor='k')
        )

def plot_robot_3d(ax, joint_angles, color='k'):
    """Draw the robot in 3D on the given Axes3D object."""
    joint_positions = forward_kinematics(joint_angles)

    # Optionally anchor the very bottom for aesthetic reasons
    ground_anchor = np.array([0.0, 0.34301, -0.2])
    joint_positions = [ground_anchor] + joint_positions

    xs = [p[0] for p in joint_positions]
    ys = [p[1] for p in joint_positions]
    zs = [p[2] for p in joint_positions]
    ax.plot(xs, ys, zs, marker='o', color=color, linestyle='-', linewidth=2)

def plot_robot_2d(ax, joint_angles, view='top', color='k'):
    """Draw the robot in 2D (top/front/side) on the given Matplotlib Axes."""
    joint_positions = forward_kinematics(joint_angles)

    # Optionally anchor the base for aesthetic reasons:
    ground_anchor = np.array([0.0, 0.34301, -0.105])
    joint_positions = [ground_anchor] + joint_positions

    if view == 'top':
        xs = [p[0] for p in joint_positions]
        ys = [p[1] for p in joint_positions]
        ax.plot(xs, ys, marker='o', color=color, linestyle='-', linewidth=2)
    elif view == 'front':
        xs = [p[0] for p in joint_positions]
        zs = [p[2] for p in joint_positions]
        ax.plot(xs, zs, marker='o', color=color, linestyle='-', linewidth=2)
    elif view == 'side':
        ys = [p[1] for p in joint_positions]
        zs = [p[2] for p in joint_positions]
        ax.plot(ys, zs, marker='o', color=color, linestyle='-', linewidth=2)

def plot_complete_data(data, display_robot=True):
    """
    Creates five figures that represent the robot's complete trajectory in different views.
    Returns the generated figure objects instead of displaying them.
    """
    resolver = ElementResolver()

    segments = []
    segment_xs, segment_ys, segment_zs = [], [], []
    previous_element = None
    previous_state = None
    last_joints = None

    total_entries = len(data)
    progress_step = max(1, total_entries // 20)

    print(f"Processing {total_entries} data points...")
    start_time = time.time()

    for i, entry in enumerate(data):
        if entry['element'] is not None and entry['state'] is not None:
            raw_element = entry.get('element', 'default').strip()
            current_element = resolver.resolve_element_name(raw_element)
            current_state = entry.get('state', 'default').strip()

            coords = np.array(entry.get('coordinates', [0, 0, 0]), dtype=float) + TRANSLATION
            x, y, z = coords

            if 'joints' in entry:
                last_joints = np.array(entry['joints'], dtype=float)

            if i % progress_step == 0 or i == total_entries - 1:
                elapsed_time = time.time() - start_time
                progress_percent = (i / total_entries) * 100
                print(f"Progress: {progress_percent:.1f}% ({i}/{total_entries}) | Time elapsed: {elapsed_time:.2f}s")

            if previous_element is not None and (current_element != previous_element or current_state != previous_state):
                if segment_xs:
                    seg_color = ELEMENT_COLORS.get(previous_element, "black")
                    seg_style = line_styles.get(previous_state, '-')
                    seg_width = line_widths.get(previous_state, 2.5)
                    seg_alpha = line_alpha.get(previous_state, 1.0)
                    segments.append(
                        (segment_xs[:], segment_ys[:], segment_zs[:], seg_style, seg_color, seg_width, seg_alpha)
                    )
                    segment_xs.clear()
                    segment_ys.clear()
                    segment_zs.clear()

            segment_xs.append(x)
            segment_ys.append(y)
            segment_zs.append(z)
            previous_element = current_element
            previous_state = current_state

    if segment_xs:
        seg_color = ELEMENT_COLORS.get(previous_element, "black")
        seg_style = line_styles.get(previous_state, '-')
        seg_width = line_widths.get(previous_state, 2.5)
        seg_alpha = line_alpha.get(previous_state, 1.0)
        segments.append(
            (segment_xs[:], segment_ys[:], segment_zs[:], seg_style, seg_color, seg_width, seg_alpha)
        )

    elapsed_time = time.time() - start_time
    print(f"Data processing complete! Total time: {elapsed_time:.2f}s")

    figures = []

    # 3D Main View
    fig_main = plt.figure(figsize=(14, 20))
    ax_main = fig_main.add_subplot(111, projection='3d')
    add_environment_3d(ax_main)
    
    for xs, ys, zs, style, color, w, a in segments:
        ax_main.plot(xs, ys, zs, linestyle=style, color=color, linewidth=w, alpha=a)

    ax_main.set_xlabel('X [m]')
    ax_main.set_ylabel('Y [m]')
    ax_main.set_zlabel('Z [m]')
    ax_main.set_xlim(x_limits)
    ax_main.set_ylim(y_limits)
    ax_main.set_zlim(z_limits)
    ax_main.set_title('Complete Robot Trajectory (Isometric view)')

    if display_robot and last_joints is not None:
        plot_robot_3d(ax_main, last_joints, color='k')
    
    figures.append(fig_main)

    # 2D Top View
    fig_top = plt.figure(figsize=(14, 20))
    ax_top = fig_top.add_subplot(111)
    add_environment_2d_top(ax_top)

    for xs, ys, zs, style, color, w, a in segments:
        ax_top.plot(xs, ys, linestyle=style, color=color, linewidth=w, alpha=a)

    ax_top.set_xlabel('X [m]')
    ax_top.set_ylabel('Y [m]')
    ax_top.set_xlim(x_limits)
    ax_top.set_ylim(y_limits)
    ax_top.set_title('Complete Robot Trajectory (Plan view)')

    if display_robot and last_joints is not None:
        plot_robot_2d(ax_top, last_joints, view='top', color='k')
    
    figures.append(fig_top)

    # 2D Side View (Y vs Z)
    fig_side = plt.figure(figsize=(14, 20))
    ax_side = fig_side.add_subplot(111)

    for xs, ys, zs, style, color, w, a in segments:
        ax_side.plot(ys, zs, linestyle=style, color=color, linewidth=w, alpha=a)

    ax_side.set_xlabel('Y [m]')
    ax_side.set_ylabel('Z [m]')
    ax_side.set_xlim(y_limits)
    ax_side.set_ylim(z_limits)
    ax_side.set_title('Complete Robot Trajectory (Side view)')

    figures.append(fig_side)

    return figures

def calculate_instantaneous_velocities(pos_data, initial_timestamp=None):
    velocities = []
    times = []

    if initial_timestamp is None:
        initial_timestamp = pos_data[0]['timestamp_send']

    for i in range(1, len(pos_data)):
        prev_data = pos_data[i - 1]
        curr_data = pos_data[i]

        # Calculate the distance and time difference
        distance = calculate_euclidean_distance(prev_data['coordinates'], curr_data['coordinates'])
        time_diff = curr_data['timestamp_send'] - prev_data['timestamp_send']

        # Calculate velocity and append it to the list
        if time_diff > 0:
            velocity = distance / time_diff
            velocities.append(velocity)
            # Calculate elapsed time from the initial timestamp
            times.append(curr_data['timestamp_send'] - initial_timestamp)

    return times, velocities

def plot_velocity_segments(pos_data, start_from_zero=True):
    # Get the global initial timestamp
    global_initial_timestamp = pos_data[0]['timestamp_send']

    # Group pos_data by segments (same element and state)
    segments = []
    current_segment = [pos_data[0]]

    for i in range(1, len(pos_data)):
        if (pos_data[i]['element'] == pos_data[i - 1]['element'] and 
            pos_data[i]['state'] == pos_data[i - 1]['state']):
            current_segment.append(pos_data[i])
        else:
            segments.append(current_segment)
            current_segment = [pos_data[i]]

    # **Filter out segments 
    segments = [seg for seg in segments if seg[0]['element'] != 'searching element']
    segments = [seg for seg in segments if seg[0]['element'] != None]

    segments.append(current_segment)

    # Set up the plotting space with one subplot per segment
    num_segments = len(segments)
    fig, axes = plt.subplots(num_segments, 1, figsize=(8, num_segments * 2))
    if num_segments == 1:
        axes = [axes]  # Ensure axes is iterable for a single subplot

    # Plot each segment's instantaneous velocity
    for i, (segment, ax) in enumerate(zip(segments, axes)):
        if start_from_zero:
            initial_timestamp = segment[0]['timestamp_send']
        else:
            initial_timestamp = global_initial_timestamp

        elapsed_times, velocities = calculate_instantaneous_velocities(segment, initial_timestamp)
        element = segment[0]['element']
        state = segment[0]['state']

        ax.plot(elapsed_times, velocities, linestyle='-', linewidth=1.5)
        ax.set_title(f"Element: {element}, State: {state}")
        ax.set_ylabel("Velocity (m/s)")

        # Set x-axis ticks and labels explicitly
        if elapsed_times:
            min_time = min(elapsed_times)
            max_time = max(elapsed_times)
            ticks = np.linspace(min_time, max_time, num=5)
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{tick:.2f}" for tick in ticks])
        else:
            ax.set_xticks([])
            ax.set_xticklabels([])

        # Display major ticks on the x-axis and y-axis
        ax.tick_params(axis='x', which='major', labelsize=8)
        ax.tick_params(axis='y', which='major', labelsize=8)

    plt.xlabel("Elapsed Time (s)")
    plt.tight_layout()
    # plt.show()
    return fig


def calculate_instantaneous_accelerations(pos_data, initial_timestamp=None):
    velocities = []
    velocity_times = []

    if initial_timestamp is None:
        initial_timestamp = pos_data[0]['timestamp_send']

    # First, calculate velocities
    for i in range(1, len(pos_data)):
        prev_data = pos_data[i - 1]
        curr_data = pos_data[i]

        # Calculate the distance and time difference
        distance = calculate_euclidean_distance(prev_data['coordinates'], curr_data['coordinates'])
        time_diff = curr_data['timestamp_send'] - prev_data['timestamp_send']

        # Calculate velocity and append it to the list
        if time_diff > 0:
            velocity = distance / time_diff
            velocities.append(velocity)
            # Time at which this velocity occurs
            velocity_times.append(curr_data['timestamp_send'] - initial_timestamp)

    accelerations = []
    acceleration_times = []

    # Then, calculate accelerations from velocities
    for i in range(1, len(velocities)):
        delta_v = velocities[i] - velocities[i - 1]
        delta_t = velocity_times[i] - velocity_times[i - 1]

        # Calculate acceleration and append it to the list
        if delta_t > 0:
            acceleration = delta_v / delta_t
            accelerations.append(acceleration)
            # Time at which this acceleration occurs
            acceleration_times.append(velocity_times[i])

    return acceleration_times, accelerations

def plot_acceleration_segments(pos_data, start_from_zero=True):
    # Get the global initial timestamp
    global_initial_timestamp = pos_data[0]['timestamp_send']

    # Group pos_data by segments (same element and state)
    segments = []
    current_segment = [pos_data[0]]

    for i in range(1, len(pos_data)):
        if (pos_data[i]['element'] == pos_data[i - 1]['element'] and 
            pos_data[i]['state'] == pos_data[i - 1]['state']):
            current_segment.append(pos_data[i])
        else:
            segments.append(current_segment)
            current_segment = [pos_data[i]]

    segments.append(current_segment)

    # **Filter out segments 
    segments = [seg for seg in segments if seg[0]['element'] != 'searching element']
    segments = [seg for seg in segments if seg[0]['element'] != None]

    # Set up the plotting space with one subplot per segment
    num_segments = len(segments)
    fig, axes = plt.subplots(num_segments, 1, figsize=(8, num_segments * 2))
    if num_segments == 1:
        axes = [axes]  # Ensure axes is iterable for a single subplot

    # Plot each segment's instantaneous acceleration
    for i, (segment, ax) in enumerate(zip(segments, axes)):
        # Determine the initial timestamp based on start_from_zero
        if start_from_zero:
            initial_timestamp = segment[0]['timestamp_send']
        else:
            initial_timestamp = global_initial_timestamp

        elapsed_times, accelerations = calculate_instantaneous_accelerations(segment, initial_timestamp)
        element = segment[0]['element']
        state = segment[0]['state']

        ax.plot(elapsed_times, accelerations, linestyle='-', linewidth=1.5)
        ax.set_title(f"Element: {element}, State: {state}")
        ax.set_ylabel("Acceleration (m/s²)")

        # Set x-axis ticks and labels explicitly
        if elapsed_times:
            min_time = min(elapsed_times)
            max_time = max(elapsed_times)
            ticks = np.linspace(min_time, max_time, num=5)
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{tick:.2f}" for tick in ticks])
        else:
            ax.set_xticks([])
            ax.set_xticklabels([])

        # Display major ticks on the x-axis and y-axis
        ax.tick_params(axis='x', which='major', labelsize=8)
        ax.tick_params(axis='y', which='major', labelsize=8)

    plt.xlabel("Elapsed Time (s)")
    plt.tight_layout()
    return fig

def plot_workers(worker_data, show=True):
    """
    Plot worker locations in both a 3D view and a 2D top view.
    
    Parameters:
        worker_data (list): A list of dictionaries. Each dict should contain at least:
            - 'worker spotted' (bool): True if a worker was detected.
            - 'worker id': an integer id.
            - 'worker coordinates': a list or tuple with at least three numbers [x, y, z].
        show (bool): If True, calls plt.show() to display the figures.
    
    Returns:
        tuple: (fig3d, fig2d) matplotlib Figure objects for the 3D and 2D views.
    """
      # --- PROCESS WORKER DATA ---
    # Filter and extract valid worker entries.
    # Each valid entry is a tuple: (worker id, x, y, z)
    workers = []
    for entry in worker_data:
        if entry.get('worker spotted', False) and entry.get('worker id') is not None and entry.get('worker coordinates'):
            worker_id = entry['worker id']
            try:
                # Convert coordinates to float and override the z value with z_level for consistency
                x, y, _ = map(float, entry['worker coordinates'])
            except Exception:
                continue
            workers.append((worker_id, x, y, z_level))
    
    # --- CREATE 3D PLOT ---
    fig3d = plt.figure()
    ax3d = fig3d.add_subplot(111, projection='3d')
    ax3d.set_xlabel("X [m]")
    ax3d.set_ylabel("Y [m]")
    ax3d.set_zlabel("Z [m]")
    ax3d.set_xlim(x_limits)
    ax3d.set_ylim(y_limits)
    ax3d.set_zlim(z_limits)
    ax3d.view_init(elev=45, azim=-45)
    
    # Draw the ground plane
    xx, yy = np.meshgrid(np.linspace(x_limits[0], x_limits[1], 10),
                         np.linspace(y_limits[0], y_limits[1], 10))
    zz = np.full_like(xx, z_level)
    ax3d.plot_surface(xx, yy, zz, color='peru', alpha=0.3, rstride=100, cstride=100)
    
    # Draw environment boxes in 3D
    for (x_center, y_center), dims in zip(box_positions, box_dims):
        width, height = dims
        rect_xs = [x_center - width/2, x_center + width/2, x_center + width/2, x_center - width/2, x_center - width/2]
        rect_ys = [y_center - height/2, y_center - height/2, y_center + height/2, y_center + height/2, y_center - height/2]
        rect_zs = [z_level] * 5
        verts = [list(zip(rect_xs, rect_ys, rect_zs))]
        poly = Poly3DCollection(verts, facecolors=box_color, alpha=box_alpha, edgecolors='k')
        ax3d.add_collection3d(poly)
    
    # Plot each worker on the 3D axes
    for (wid, wx, wy, wz) in workers:
        color = WORKER_COLORS.get(wid, 'black')
        marker = worker_marker_styles.get(wid, 'x')
        if marker == 'zone':
            # Draw a custom square marker
            size = WORKER_SQUARE_SIZE
            sq_xs = [wx, wx + size, wx + size, wx, wx]
            sq_ys = [wy, wy, wy - size, wy - size, wy]
            sq_zs = [wz] * 5
            verts = [list(zip(sq_xs, sq_ys, sq_zs))]
            poly = Poly3DCollection(verts, facecolors=color, alpha=0.7, edgecolors=color)
            ax3d.add_collection3d(poly)
        else:
            ax3d.scatter(wx, wy, wz, color=color, marker=marker, s=50)
    ax3d.set_title("Worker Positions (3D View)")
    
    # --- CREATE 2D TOP VIEW PLOT ---
    fig2d = plt.figure()
    ax2d = fig2d.add_subplot(111)
    ax2d.set_xlabel("X [m]")
    ax2d.set_ylabel("Y [m]")
    ax2d.set_xlim(x_limits)
    ax2d.set_ylim(y_limits)
    
    # Draw ground rectangle
    ground_width = x_limits[1] - x_limits[0]
    ground_height = y_limits[1] - y_limits[0]
    ground_rect = Rectangle((x_limits[0], y_limits[0]), ground_width, ground_height,
                            facecolor='peru', alpha=0.3)
    ax2d.add_patch(ground_rect)
    
    # Draw environment boxes in 2D
    for (x_center, y_center), dims in zip(box_positions, box_dims):
        w, h = dims
        x_left = x_center - w / 2
        y_bottom = y_center - h / 2
        box_rect = Rectangle((x_left, y_bottom), w, h,
                             facecolor=box_color, alpha=box_alpha, edgecolor='k')
        ax2d.add_patch(box_rect)
    
    # Plot each worker on the 2D axes
    for (wid, wx, wy, _) in workers:
        color = WORKER_COLORS.get(wid, 'black')
        marker = worker_marker_styles.get(wid, 'x')
        if marker == 'zone':
            rect = Rectangle((wx, wy), WORKER_SQUARE_SIZE, WORKER_SQUARE_SIZE,
                             facecolor=color, alpha=0.7, edgecolor=color)
            ax2d.add_patch(rect)
        else:
            ax2d.scatter(wx, wy, color=color, marker=marker, s=50)
    ax2d.set_title("Worker Positions (2D Top View)")
    
    if show:
        plt.show()
        
    return fig3d, fig2d
