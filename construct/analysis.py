# construct/analysis.py

import numpy as np
import pandas as pd
import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch

from shapely.geometry import Polygon

from configs.system_config import *
from configs.construction_config import *

display_robot = True

def process_hazard_events(data):
    """
    Processes the list of dictionaries and extracts  hazard events.
    A hazard event is defined as a period where 'obstacle_maneuver'
    toggles from False to True (start) and later back to False (end).

    During the event:
    - If any packet has 'worker spotted' = True, we record 'worker id'.
    - We capture 'element' and 'state' from the first packet that starts the event.
    - We capture 'start_coordinate' from the first packet of the event.
    - We capture 'end_coordinate' from the last packet that ends the event.
    - We calculate the travel distance using the end effector coordinates
      by summing the Euclidean distances between consecutive packets
      during the event.

    Parameters:
        data (list): List of dictionaries containing event data.
    
    Returns:
        events (list): List of dictionaries. Each dictionary contains:
            'start': float                    # relative start time of the event
            'end': float                      # relative end time of the event
            'worker_id': int/None             # worker ID if spotted, else None
            'element': str/None               # 'element' from the first packet of the event
            'state': str/None                 # 'state' from the first packet of the event
            'start_coordinate': list or None  # coordinates from the first packet
            'end_coordinate': list or None    # coordinates from the last packet
            'travel_distance': float          # total travel distance during the event
    """
    # Sort the data by timestamp_send (ascending order).
    sorted_data = sorted(data, key=lambda x: x['timestamp_send'])
    base_time = sorted_data[0]['timestamp_send']
    events = []
    in_event = False
    
    # Variables to track event details
    event_start = None
    event_worker_id = None
    event_element = None
    event_state = None
    event_start_coord = None
    prev_coord = None
    event_distance = 0.0

    for packet in sorted_data:
        obstacle = packet.get('obstacle_maneuver', False)
        ts = packet.get('timestamp_send')
        worker_spotted = packet.get('worker spotted', False)
        worker_id = packet.get('worker id') if worker_spotted else None
        
        if not in_event and obstacle:
            # Start of a new event
            in_event = True
            event_start = ts
            event_worker_id = worker_id
            event_element = packet.get('element')
            event_state = packet.get('state')
            event_start_coord = packet.get('coordinates')

            prev_coord = event_start_coord
            event_distance = 0.0

        elif in_event:
            if obstacle:
                # Still within the same event
                # Update worker ID if we haven't captured one yet
                if event_worker_id is None and worker_spotted:
                    event_worker_id = worker_id

                current_coord = packet.get('coordinates')
                if prev_coord is not None and current_coord is not None:
                    # Calculate Euclidean distance between consecutive coordinates.
                    dist = calculate_euclidean_distance(prev_coord, current_coord)
                    event_distance += dist
                prev_coord = current_coord
            else:
                # End of event: obstacle_maneuver became False.
                event_end = ts
                event_end_coord = packet.get('coordinates')
                
                # Save event info
                events.append({
                    'start_time': round(event_start - base_time, 2),
                    'end_time': round(event_end - base_time, 2),
                    'worker_id': event_worker_id,
                    'element': event_element,
                    'state': event_state,
                    'start_coordinate': event_start_coord,
                    'end_coordinate': event_end_coord,
                    'travel_distance': round(event_distance, 2)
                })
                
                # Reset for the next potential event
                in_event = False
                event_start = None
                event_worker_id = None
                event_element = None
                event_state = None
                event_start_coord = None

    # If we exit the loop while still in an event, close it with the last packet's info
    if in_event:
        last_packet = sorted_data[-1]
        event_end = last_packet['timestamp_send']
        event_end_coord = last_packet.get('coordinates')
        events.append({
            'start_time': event_start,
            'end_time': event_end,
            'worker_id': event_worker_id,
            'element': event_element,
            'state': event_state,
            'start_coordinate': event_start_coord,
            'end_coordinate': event_end_coord,
            'travel_distance': round(event_distance, 2)
        })

    # Assign hazard IDs 
    for i, e in enumerate(events):
        e["id"] = f"H_{i:02d}"
        
    return events

def plot_hazard_events(events):
    """
    Creates a figure with two columns:
      - Left: Hazard events plotted as horizontal lines with text boxes above each line.
      - Right: A details panel listing all events.

    Text boxes are offset above each event line and remain inside the plot horizontally.
    We allow a larger top margin so that text isn't forced down onto the line.
    """

    # Make the figure tall enough to accommodate high offsets.
    fig, (ax_plot, ax_details) = plt.subplots(
        ncols=2, figsize=(14, 9),
        gridspec_kw={'width_ratios': [3, 1]}
    )

    legend_handles = {}

    # Vertical offset in display pixels above the event line.
    # Increase if you want more space above the line.
    V_OFFSET = 40

    for i, event in enumerate(events, start=1):
        start = event['start_time']
        end = event['end_time']
        hazard_id = event.get('id', 'NoID')
        w_id = event.get('worker_id', None)
        
        # Pick the color from the worker dict if present, else black.
        color = WORKER_COLORS.get(w_id, 'black')
        
        # Draw the horizontal line at y = i
        ax_plot.hlines(y=i, xmin=start, xmax=end, color=color, linewidth=2)
        
        # For legend
        if w_id not in legend_handles:
            label = f"Worker {w_id}" if w_id is not None else "Unconfirmed worker ID"
            legend_handles[w_id] = (color, label)
        
        # The event midpoint
        mid_x = (start + end) / 2.0
        
        # Place annotation 40 px above the line, horizontally centered.
        ann = ax_plot.annotate(
            str(hazard_id),
            xy=(mid_x, i),
            xytext=(0, V_OFFSET),
            textcoords='offset points',
            ha='center',
            va='bottom',   # 'bottom' so the text's bottom aligns with xytext
            fontsize=8,
            bbox=dict(facecolor='white', edgecolor='black', alpha=0.7),
            arrowprops=dict(
                arrowstyle='->',
                color='gray',
                lw=1,
                shrinkA=5,
                shrinkB=5
            )
        )
        
        # Keep annotation horizontally inside the plot, plus a smaller vertical check
        # so we don't push it back on top of the line if there's enough space.
        _keep_annotation_in_bounds(ax_plot, fig, ann, boundary_margin_x=30, boundary_margin_y=20)

    # Label & scale the hazard plot
    ax_plot.set_xlabel("Time [s]")
    ax_plot.set_ylabel("Event Number")
    ax_plot.set_title("Hazard Detection Events")
    ax_plot.set_yticks(range(1, len(events) + 1))
    
    # Add padding to x-axis
    min_time = min(e['start_time'] for e in events)
    max_time = max(e['end_time'] for e in events)
    x_range = max_time - min_time
    ax_plot.set_xlim(min_time - 0.05 * x_range, max_time + 0.05 * x_range)
    
    # Legend outside
    legend_elems = []
    for w_id, (c, label) in legend_handles.items():
        legend_elems.append(Line2D([0], [0], color=c, lw=3, label=label))
    if legend_elems:
        ax_plot.legend(handles=legend_elems, loc='upper left', bbox_to_anchor=(1.05, 1))

    # -------- Right Panel: Detailed text
    ax_details.axis('off')
    ax_details.set_title("Hazard Details", loc='center', fontweight='bold')
    
    details_lines = []
    for event in events:
        h_id = event.get('id', 'NoID')
        w_id = event.get('worker_id', None)
        start = event['start_time']
        end = event['end_time']
        duration = end - start
        details_lines.append(
            f"• {h_id}: Worker {w_id}, Start: {start:.2f}s, End: {end:.2f}s, Duration: {duration:.2f}s"
        )
    details_text = "\n\n".join(details_lines)
    ax_details.text(
        0, 1, details_text,
        transform=ax_details.transAxes,
        ha='left', va='top', fontsize=10, wrap=True
    )
    
    plt.tight_layout()
    plt.close(fig)
    return fig

def _keep_annotation_in_bounds(ax, fig, ann, boundary_margin_x=30, boundary_margin_y=20):
    """
    Ensures that the annotation's bounding box remains inside the axes, applying
    separate horizontal and vertical margins in display pixels.

    If the text box is too close to the left/right edges, we shift it horizontally.
    If it's above or below the plot edges by boundary_margin_y, we shift it vertically.
    """

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    
    bbox = ann.get_window_extent(renderer=renderer)
    ax_bbox = ax.get_window_extent(renderer=renderer)
    
    # Current offset
    dx, dy = ann.get_position()

    # Left boundary
    if bbox.x0 < ax_bbox.x0 + boundary_margin_x:
        shift = (ax_bbox.x0 + boundary_margin_x) - bbox.x0
        dx += shift
    # Right boundary
    if bbox.x1 > ax_bbox.x1 - boundary_margin_x:
        shift = (ax_bbox.x1 - boundary_margin_x) - bbox.x1
        dx += shift
    # Bottom boundary
    if bbox.y0 < ax_bbox.y0 + boundary_margin_y:
        shift = (ax_bbox.y0 + boundary_margin_y) - bbox.y0
        dy += shift
    # Top boundary
    if bbox.y1 > ax_bbox.y1 - boundary_margin_y:
        shift = (ax_bbox.y1 - boundary_margin_y) - bbox.y1
        dy += shift

    ann.set_position((dx, dy))
    fig.canvas.draw()


def tabulate_hazard_events(events):
    """
    Takes the list of event dictionaries from process_hazard_events
    and returns a pandas DataFrame.

    Parameters:
        events (list): List of dictionaries with keys:
                       'id', 'start_time', 'end_time', 'worker_id',
                       'element', 'state', 'start_coordinate',
                       'end_coordinate', 'travel_distance'

    Returns:
        pd.DataFrame: DataFrame where each row corresponds to one hazard event,
                      including event ID and travel distance.
    """

    # Create a DataFrame directly from the list of dictionaries
    df = pd.DataFrame(events)
    
    # Reorder or rename columns:
    columns_order = [
        'id', 'worker_id', 'start_time', 'end_time', 'travel_distance',
        'start_coordinate', 'end_coordinate', 'element', 'state'
    ]
    df = df[columns_order]
    rename_map = {
        "id": " Hazard ID",
        "worker_id": "Worker ID",
        "start_time": "Start Time",
        "end_time": "End Time",
        "travel_distance": "Travel Distance",
        "start_coordinate": "Start Coordinate",
        "end_coordinate": "End Coordinate",
        "element": "Element",
        "state": "Activity"
    }
    df = df.rename(columns=rename_map)
    df["Activity"] = df["Activity"].replace({'grab': 'pick'})

    return df

def calculate_euclidean_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return np.linalg.norm(np.array(point1) - np.array(point2))

def summarize_trajectory_metrics(assembly_data,
                                           hazard_events=None,
                                           construction_type="assembly"):
    """
    Summarizes trajectory metrics by grouping consecutive raw packets that have the same
    (element, state). For each group (which we call a 'segment'), we compute:
        - Start/End coordinates
        - Start/End times (relative to the earliest timestamp)
        - Elapsed Time
        - Traversed length
        - Path efficiency
        - Average curvature
        - Average velocity
        - Average acceleration
        - Hazard Event overlaps
      and return a pandas DataFrame with one row per segment.

    Parameters
    ----------
    assembly_data : list of dict
        Raw trajectory data. Each dict must at least have:
            'element' (str),
            'state' (str),
            'coordinates' ([x,y,z]),
            'timestamp_send' (float) or a similar timestamp key.
        The list should be ordered in ascending time if you want correct segmentation.

    hazard_events : list of dict or None
        Hazard event data, each with:
            'start_time', 'end_time', 'element', 'state', 'id', and optionally 'worker_id'.
        If None, we skip hazard checks.

    construction_type : str
        "assembly" or "reassembly" => states are {'search', 'pick', 'swing', 'place', 'swing_back'}
        "disassembly" => states are {'pick', 'swing', 'place', 'swing_back'}

    Returns
    -------
    pd.DataFrame
        One row per (element, state) segment, with columns:
          - Element
          - State
          - ArUco ID
          - Start [x,y,z]
          - End [x,y,z]
          - Start Time [s]
          - End Time [s]
          - Elapsed Time [s]
          - Average Velocity [m/s]
          - Average Acceleration [m/s²]
          - Traversed Length [m]
          - Path Efficiency [%]
          - Average Curvature [radians]
          - Hazard Event
    """

    construction_type = construction_type.lower().strip()
    if construction_type == "disassembly":
        allowed_states = {'pick', 'swing', 'place', 'swing_back'}
    else:  # "assembly" or "reassembly"
        allowed_states = {'search', 'pick', 'swing', 'place', 'swing_back'}

    # ArUco ID mapping 
    aruco_mapping = {
        "scanning_site": '[0],[1],[2]',
        "Foundation": '[11]',
        "Wall": '[10]',
        "Floor": '[12]',
        "Toilet": '-' 
    }

    # 4) Hazard-event lookup
    def get_hazard_description(element, state, seg_start, seg_end):
        """
        Return a neatly formatted string describing hazard events that:
          - Overlap in time with [seg_start, seg_end]
          - Match the same element and state
        """
        if not hazard_events:
            return "No hazard"

        overlapping = []
        for h in hazard_events:
            # Must match both element and state
            h_element = h.get('element')
            h_state   = h.get('state')
            if h_element != element or h_state != state:
                continue

            h_start = h.get('start_time', None)
            h_end   = h.get('end_time', None)
            if h_start is None or h_end is None:
                continue

            # If hazard times are suspiciously small, treat them as relative
            # and shift by the earliest data timestamp if needed:
            if h_start < 1000:
                h_start += initial_timestamp
                h_end   += initial_timestamp

            # Check for overlap
            if (h_end >= seg_start) and (h_start <= seg_end):
                rel_start = h_start - initial_timestamp
                rel_end   = h_end - initial_timestamp
                worker_id = h.get('worker_id')
                hazard_id = h.get('id', "Haz?")
                if worker_id is not None:
                    overlapping.append(f"{hazard_id}: Worker {worker_id} from {rel_start:.2f}s to {rel_end:.2f}s")
                else:
                    overlapping.append(f"{hazard_id}: Hazard from {rel_start:.2f}s to {rel_end:.2f}s")

        return " | ".join(overlapping) if overlapping else "No hazard"

    assembly_data_sorted = sorted(assembly_data, key=lambda d: d.get('timestamp_send', 0.0))

    if assembly_data_sorted:
        initial_timestamp = assembly_data_sorted[0].get('timestamp_send', 0.0)
    else:
        return pd.DataFrame()

    elements, states_list = [], []
    start_coords, end_coords = [], []
    start_times, end_times, elapsed_times = [], [], []
    traversed_lengths, path_efficiencies, avg_curvatures = [], [], []
    avg_velocities, avg_accelerations = [], []
    hazard_descriptions, aruco_ids = [], []


    prev_element = None
    prev_state   = None
    seg_start_coord = None
    seg_start_time  = None
    prev_coordinates = None
    prev_direction   = None
    total_length     = 0.0
    total_curvature  = 0.0
    curvature_count  = 0
    start_velocity   = 0.0  # for acceleration calculation in each finalized segment

   
    def finalize_segment(end_time, end_coord):
        nonlocal start_velocity
        if not seg_start_coord:
            return  # means we didn't start a segment properly

        seg_elapsed = end_time - seg_start_time
        # Relative times
        rel_start_s = seg_start_time - initial_timestamp
        rel_end_s   = end_time - initial_timestamp

        # If it's scanning_site,force state = '-'
        effective_state = "-" if (prev_element == "scanning_site") else prev_state
        
        # Convert "Toilet" to "Bathroom module" if needed
        effective_element = "Bathroom module" if (prev_element == "Toilet") else prev_element

        # Path efficiency
        straight_line_distance = calculate_euclidean_distance(seg_start_coord, end_coord)
        path_eff = (straight_line_distance / total_length * 100.0) if total_length > 0 else 100.0

        # Average curvature
        avg_curv = total_curvature / curvature_count if curvature_count > 0 else 0.0

        # Average velocity
        avg_vel = total_length / seg_elapsed if seg_elapsed > 1e-9 else 0.0

        # Average acceleration: difference from the velocity we had at the *start* of this segment
        avg_acc = (avg_vel - start_velocity) / seg_elapsed if seg_elapsed > 1e-9 else 0.0
        start_velocity = avg_vel  # next segment's baseline

        # Hazard info
        hazard_info = get_hazard_description(effective_element, effective_state,
                                             seg_start_time, end_time)

        # ArUco ID
        aruco_id = aruco_mapping.get(prev_element, '-')

        elements.append(effective_element)
        states_list.append(effective_state)
        aruco_ids.append(aruco_id)
        start_coords.append([round(v, 2) for v in seg_start_coord])
        end_coords.append([round(v, 2) for v in end_coord])
        start_times.append(round(rel_start_s, 2))
        end_times.append(round(rel_end_s, 2))
        elapsed_times.append(round(seg_elapsed, 2))
        traversed_lengths.append(round(total_length, 2))
        path_efficiencies.append(round(path_eff, 2))
        avg_curvatures.append(round(avg_curv, 2))
        avg_velocities.append(round(avg_vel, 2))
        avg_accelerations.append(round(avg_acc, 2))
        hazard_descriptions.append(hazard_info)

    i = 0
    while i < len(assembly_data_sorted):
        packet = assembly_data_sorted[i]
        i += 1

        element = packet.get('element')
        state   = packet.get('state')
        coords  = packet.get('coordinates')
        t_send  = packet.get('timestamp_send')


        if (element is None) or (state is None) or (coords is None) or (t_send is None):
            continue

        if element == "scanning_site":
            state = "-"

        if element != "scanning_site" and state not in allowed_states:
            # You could skip or forcibly set, but for clarity we skip
            continue

        if prev_element is None and prev_state is None:
            # Initialize the segment
            prev_element = element
            prev_state   = state
            seg_start_coord  = coords
            seg_start_time   = t_send
            prev_coordinates = coords
            prev_direction   = None
            total_length     = 0.0
            total_curvature  = 0.0
            curvature_count  = 0
            # We do NOT reset start_velocity here because it’s the velocity at the start of the very first segment
            continue

        if element == prev_element and state == prev_state:
            # Accumulate distance, curvature
            dist = calculate_euclidean_distance(prev_coordinates, coords)
            total_length += dist

            # Curvature
            delta_vec = np.array(coords) - np.array(prev_coordinates)
            norm = np.linalg.norm(delta_vec)
            if norm > 1e-9:
                direction = delta_vec / norm
                if prev_direction is not None:
                    cos_angle = np.clip(np.dot(prev_direction, direction), -1.0, 1.0)
                    angle_change = np.arccos(cos_angle)
                    total_curvature += angle_change
                    curvature_count += 1
                prev_direction = direction

            prev_coordinates = coords

        else:
            # The element or state changed => finalize the old segment
            finalize_segment(t_send, prev_coordinates)
            # Start a new segment
            prev_element = element
            prev_state   = state
            seg_start_coord  = coords
            seg_start_time   = t_send
            prev_coordinates = coords
            prev_direction   = None
            total_length     = 0.0
            total_curvature  = 0.0
            curvature_count  = 0


    if prev_element is not None or prev_state is not None:
        finalize_segment(assembly_data_sorted[-1]['timestamp_send'], prev_coordinates)

    summary = pd.DataFrame({
        'Element': elements,
        'State': states_list,
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
        'Average Curvature [radians]': avg_curvatures,
        'Hazard Event': hazard_descriptions
    })

    element_mapping = {
        "scanning_site": "Scanning site",
        "Wall_1": "Wall 1",
        'Floor_1':"Floor 1",
        'Floor_2':"Floor 2",  
        'Wall_1': "Wall 1",
        'Wall_2': "Wall 2", 
        'bathroom_module_1': "Bathroom module 1",
        'bathroom_module_2': "Bathroom module 2",
        'bathroom_module_3': "Bathroom module 3"
    }
    summary["Element"] = summary["Element"].replace(element_mapping)

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
        first_detected = f"{relative_time:.2f}"

        activity = detection.get('state')        
        construction = detection.get('element')
 
        rows.append({
            "worker id": worker_id,
            "worker location": worker_location,
            "timestamp": first_detected,
            "Activity": activity,
            "Construction Element": construction
        })

    # 4. Create the DataFrame.
    df = pd.DataFrame(rows)
    return df

# class ElementResolver:
#     def __init__(self):
#         self.wall_count   = 0
#         self.floor_count  = 0
#         self.toilet_count = 0

#     def resolve_element_name(self, raw_element):
#         raw_element = raw_element.lower().strip()
#         elements = {
#             "foundation":       "Foundation",
#             "scanning site":    "Scanning site",
#             "searching element":"Searching element",
#             "wall":             "Wall",
#             "floor":            "Floor",
#             "toilet":           "Bathroom module"
#         }
#         for key, label in elements.items():
#             if key in raw_element:
#                 if key in ["scanning site", "searching element", "foundation"]:
#                     return label
#                 if key == "wall":
#                     self.wall_count = min(self.wall_count + 1, 2)
#                     return f"{label} {self.wall_count}"
#                 if key == "floor":
#                     self.floor_count = min(self.floor_count + 1, 2)
#                     return f"{label} {self.floor_count}"
#                 if key == "toilet":
#                     self.toilet_count = min(self.toilet_count + 1, 3)
#                     return f"{label} {self.toilet_count}"
#         return "default"
    
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

# def create_custom_legend_handles():
#     """
#     Build a list of legend handles (Line2D objects) for each
#     element + state combination we want to display.
#     """
#     legend_handles = []

#     # List out the elements you want to show in the legend:
#     elements = [
#         "Scanning site",
#         "Foundation",
#         "Searching element",
#         "Wall 1",
#         "Wall 2",
#         "Floor 1",
#         "Floor 2",
#         "Bathroom module 1",
#         "Bathroom module 2",
#         "Bathroom module 3"
#     ]

#     # The states we'll differentiate by line thickness/alpha
#     states = ["grab", "place"]

#     for element in elements:
#         color_for_element = ELEMENT_COLORS.get(element, "black")

#         for st in states:
#             lw    = line_widths.get(st, 2.0)
#             alpha = line_alpha.get(st, 1.0)
#             # We'll just keep the same line style '-'
#             linestyle = line_styles.get(st, '-')

#             # Build a label, e.g. "Wall 1 (grab)" or "Floor 2 (place)"
#             label_text = f"{element} ({st})"

#             line = Line2D(
#                 [0], [0],               # dummy data
#                 color=color_for_element,
#                 lw=lw,
#                 linestyle=linestyle,
#                 alpha=alpha,
#                 label=label_text
#             )
#             legend_handles.append(line)

#     return legend_handles

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

def normalize_element_name(element):
    """
    Convert an element string (which might use underscores) into the format expected
    by ELEMENT_COLORS. For example, "wall_2" becomes "Wall 2".
    """
    if element is None:
        return ""
    candidate = element.replace("_", " ").strip()
    # Look for an exact match (ignoring case) in ELEMENT_COLORS keys.
    for key in ELEMENT_COLORS:
        if candidate.lower() == key.lower():
            return key
    return candidate.title()


def plot_complete_data(data, title, construction_type="assembly", display_robot=True, debug=False):
    """
    Creates four figures representing the complete robot trajectory.
    
    The input data is a list of dictionaries that already have a numbered element (e.g. "Wall 1",
    "Floor 2", etc.) and a state. For assembly/reassembly, allowed states are:
        'search', 'pick', 'swing', 'place', 'swing_back'
    while for disassembly the allowed states are:
        'pick', 'swing', 'place', 'swing_back'
    If an entry’s normalized element is "Scanning site", its state is forced to "-".
    
    State style mappings are read from the configuration file:
       STATE_STYLES_ASSEMBLY, ALLOWED_STATES_ASSEMBLY for assembly/reassembly,
       STATE_STYLES_DISASSEMBLY, ALLOWED_STATES_DISASSEMBLY for disassembly,
    and SCANNING_STYLE for scanning site entries.
    
    Element colors come from ELEMENT_COLORS. The integrated legend lists, for each element and state,
    entries of the form "Element (state)" with the element color and state linestyle & linewidth.
    A translucent horizontal separator is added after each element group.
    
    The function produces four views:
       - 3D Isometric
       - 2D Top (Plan) view (X vs. Y)
       - 2D Side view (Y vs. Z)
       - 2D Front view (X vs. Z)
       
    Axis labels are "X [m]", "Y [m]", "Z [m]".
    
    Returns a list of Matplotlib Figure objects.
    """
    # --- Select state styles from configuration ---
    if construction_type.lower() == "disassembly":
        state_styles = STATE_STYLES_DISASSEMBLY
        allowed_states = ALLOWED_STATES_DISASSEMBLY
    else:
        state_styles = STATE_STYLES_ASSEMBLY
        allowed_states = ALLOWED_STATES_ASSEMBLY

    scanning_style = SCANNING_STYLE

    # --- Group consecutive data points into segments ---
    # Each segment: (list_xs, list_ys, list_zs, linestyle, color, linewidth, alpha)
    segments = []
    seg_xs, seg_ys, seg_zs = [], [], []
    prev_element = None
    prev_state = None
    last_joints = None

    for entry in data:
        # Skip entries missing element or state
        if entry.get('element') is None or entry.get('state') is None:
            continue
        # Normalize element name so that "wall_2" and "Wall 2" are treated equally.
        current_element = normalize_element_name(entry.get('element'))
        current_state = entry.get('state').strip()
        # For scanning site, force state to "-"
        if current_element.lower() == "scanning site":
            current_state = "-"

        # Get coordinates (assume TRANSLATION is defined)
        coords = np.array(entry.get('coordinates', [0, 0, 0]), dtype=float) + TRANSLATION
        x, y, z = coords

        if 'joints' in entry:
            last_joints = np.array(entry['joints'], dtype=float)

        # If element or state changes, finish previous segment.
        if prev_element is not None and (current_element != prev_element or current_state != prev_state):
            col = ELEMENT_COLORS.get(prev_element, "black")
            if prev_state == "-":
                ls = scanning_style["linestyle"]
                lw = scanning_style["linewidth"]
            else:
                style_dict = state_styles.get(prev_state, {"linestyle": "-", "linewidth": 2.5})
                ls = style_dict["linestyle"]
                lw = style_dict["linewidth"]
            segments.append((seg_xs[:], seg_ys[:], seg_zs[:], ls, col, lw, 1.0))
            seg_xs.clear(); seg_ys.clear(); seg_zs.clear()

        seg_xs.append(x)
        seg_ys.append(y)
        seg_zs.append(z)
        prev_element = current_element
        prev_state = current_state

    if seg_xs:
        col = ELEMENT_COLORS.get(prev_element, "black")
        if prev_state == "-":
            ls = scanning_style["linestyle"]
            lw = scanning_style["linewidth"]
        else:
            style_dict = state_styles.get(prev_state, {"linestyle": "-", "linewidth": 2.5})
            ls = style_dict["linestyle"]
            lw = style_dict["linewidth"]
        segments.append((seg_xs[:], seg_ys[:], seg_zs[:], ls, col, lw, 1.0))

    # --- Build integrated legend ---

    integrated_handles = []
    integrated_labels = []
    # Build a set of normalized unique elements from the data.
    unique_elements = { normalize_element_name(entry.get('element')) for entry in data if entry.get('element') }
    for element in sorted(unique_elements):
        # For "Scanning site", we assume a single state "-"
        if element.lower() == "scanning site":
            states_to_include = ["-"]
        else:
            states_to_include = allowed_states
        for st in states_to_include:
            if st == "-":
                style_dict = scanning_style
                label = f"{element} (scanning_site)"
            else:
                style_dict = state_styles.get(st, {"linestyle": "-", "linewidth": 2})
                label = f"{element} ({st})"
            h = Line2D([0], [0],
                       color=ELEMENT_COLORS.get(element, "black"),
                       lw=style_dict["linewidth"],
                       linestyle=style_dict["linestyle"],
                       alpha=0.75)
            integrated_handles.append(h)
            integrated_labels.append(label)
        # Add a separator entry after each element group (a translucent horizontal line)
        sep = Line2D([0], [0], color="gray", lw=2, linestyle="-", alpha=0.5)
        integrated_handles.append(sep)
        integrated_labels.append("________________________________")

    # --- Plotting ---
    figures = []

    # 1) 3D Isometric View
    fig_3d = plt.figure(figsize=(14, 20))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    add_environment_3d(ax_3d)
    for xs, ys, zs, ls, col, lw, alpha in segments:
        ax_3d.plot(xs, ys, zs, linestyle=ls, color=col, linewidth=lw, alpha=alpha)
    ax_3d.set_xlabel("X [m]")
    ax_3d.set_ylabel("Y [m]")
    ax_3d.set_zlabel("Z [m]")
    ax_3d.set_xlim(x_limits)
    ax_3d.set_ylim(y_limits)
    ax_3d.set_zlim(z_limits)
    ax_3d.set_title(f"{title} trajectory (Isometric view)")
    if display_robot and last_joints is not None:
        plot_robot_3d(ax_3d, last_joints, color='k')
    ax_3d.legend(integrated_handles, integrated_labels, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig_3d.subplots_adjust(right=0.8)
    figures.append(fig_3d)

    # 2) 2D Top (Plan) View (X vs Y)
    fig_top = plt.figure(figsize=(14, 20))
    ax_top = fig_top.add_subplot(111)
    add_environment_2d_top(ax_top)
    for xs, ys, zs, ls, col, lw, alpha in segments:
        ax_top.plot(xs, ys, linestyle=ls, color=col, linewidth=lw, alpha=alpha)
    ax_top.set_xlabel("X [m]")
    ax_top.set_ylabel("Y [m]")
    ax_top.set_xlim(x_limits)
    ax_top.set_ylim(y_limits)
    ax_top.set_title(f"{title} trajectory (Plan view)")
    if display_robot and last_joints is not None:
        plot_robot_2d(ax_top, last_joints, view='top', color='k')
    ax_top.legend(integrated_handles, integrated_labels, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig_top.subplots_adjust(right=0.8)
    figures.append(fig_top)

    # 3) 2D Side View (Y vs Z)
    fig_side = plt.figure(figsize=(14, 20))
    ax_side = fig_side.add_subplot(111)
    for xs, ys, zs, ls, col, lw, alpha in segments:
        ax_side.plot(ys, zs, linestyle=ls, color=col, linewidth=lw, alpha=alpha)
    ax_side.set_xlabel("Y [m]")
    ax_side.set_ylabel("Z [m]")
    ax_side.set_xlim(y_limits)
    ax_side.set_ylim(z_limits)
    ax_side.set_title(f"{title} trajectory (Side view)")
    ax_side.legend(integrated_handles, integrated_labels, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig_side.subplots_adjust(right=0.8)
    figures.append(fig_side)

    # 4) 2D Front View (X vs Z)
    fig_front = plt.figure(figsize=(14, 20))
    ax_front = fig_front.add_subplot(111)
    for xs, ys, zs, ls, col, lw, alpha in segments:
        ax_front.plot(xs, zs, linestyle=ls, color=col, linewidth=lw, alpha=alpha)
    ax_front.set_xlabel("X [m]")
    ax_front.set_ylabel("Z [m]")
    ax_front.set_xlim(x_limits)
    ax_front.set_ylim(z_limits)
    ax_front.set_title(f"{title} trajectory (Front view)")
    ax_front.legend(integrated_handles, integrated_labels, loc="upper left", bbox_to_anchor=(1.02, 1))
    fig_front.subplots_adjust(right=0.8)
    figures.append(fig_front)

    if not debug:
        for fig in figures:
            plt.close(fig)
    return figures

def plot_workers(worker_data, plot_3d=True):
    """
    Plot worker locations in both a 3D view and a 2D top view.
    
    Parameters:
        worker_data (list): A list of dictionaries. Each dict should contain at least:
            - 'worker spotted' (bool): True if a worker was detected.
            - 'worker id': an integer id.
            - 'worker coordinates': a list or tuple with at least three numbers [x, y, z].
        show (bool): If True, calls plt.show() to display the figures.
        plot_3d (bool): If True, creates a 3D view in addition to the 2D view.
    
    Returns:
        tuple: (fig3d, fig2d) matplotlib Figure objects for the 3D and 2D views.
               If plot_3d is False, fig3d will be None.
    """   
    # --- PROCESS WORKER DATA ---
    workers = []
    for entry in worker_data:
        if (entry.get('worker spotted', False) and
            entry.get('worker id') is not None and
            entry.get('worker coordinates')):
            
            worker_id = entry['worker id']
            try:
                # Convert coordinates to float and override z with z_level for consistency
                x, y, _ = map(float, entry['worker coordinates'])
            except Exception:
                continue
            workers.append((worker_id, x, y, z_level))
    
    # Build a dictionary of unique worker styles for the legend
    unique_worker_styles = {}
    for (wid, _, _, _) in workers:
        if wid not in unique_worker_styles:
            marker = worker_marker_styles.get(wid, 'x')
            color = WORKER_COLORS.get(wid, 'black')
            unique_worker_styles[wid] = (marker, color)
    
    # Create legend handles and labels
    legend_handles = []
    legend_labels = []
    for wid, (marker, color) in unique_worker_styles.items():
        label = f"Worker {wid}"
        
        if marker == 'zone':
            # Use a rectangular patch in the legend
            handle = Patch(facecolor=color, edgecolor=color, alpha=0.7)
        else:
            # For unfilled markers like 'x', use markeredgecolor
            handle = Line2D(
                [0], [0],
                marker=marker,
                color='none',            # no connecting line in legend
                markeredgecolor=color,   # color for 'x' lines
                markerfacecolor=color,   # if you had a filled marker, it would fill with this color
                markersize=8,
                linewidth=0
            )
        
        legend_handles.append(handle)
        legend_labels.append(label)
    
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
    ground_rect = Rectangle(
        (x_limits[0], y_limits[0]),
        ground_width,
        ground_height,
        facecolor='peru',
        alpha=0.3
    )
    ax2d.add_patch(ground_rect)
    
    # Draw environment boxes in 2D
    for (x_center, y_center), dims in zip(box_positions, box_dims):
        w, h = dims
        x_left = x_center - w / 2
        y_bottom = y_center - h / 2
        box_rect = Rectangle(
            (x_left, y_bottom),
            w,
            h,
            facecolor=box_color,
            alpha=box_alpha,
            edgecolor='k'
        )
        ax2d.add_patch(box_rect)
    
    # Plot each worker on the 2D axes
    for (wid, wx, wy, _) in workers:
        color = WORKER_COLORS.get(wid, 'black')
        marker = worker_marker_styles.get(wid, 'x')
        
        if marker == 'zone':
            rect = Rectangle(
                (wx, wy),
                WORKER_SQUARE_SIZE,
                WORKER_SQUARE_SIZE,
                facecolor=color,
                alpha=0.7,
                edgecolor=color
            )
            ax2d.add_patch(rect)
        else:
            ax2d.scatter(wx, wy, color=color, marker=marker, s=50)
    
    # ax2d.set_title("Worker Positions (2D Top View)")
    ax2d.legend(legend_handles, legend_labels)
    
    # --- CREATE 3D PLOT (if requested) ---
    fig3d = None
    if plot_3d:
        fig3d = plt.figure(figsize=(12, 12))
        ax3d = fig3d.add_subplot(111, projection='3d')
        ax3d.set_xlabel("X [m]")
        ax3d.set_ylabel("Y [m]")
        ax3d.set_zlabel("Z [m]", labelpad=-1)
        ax3d.zaxis.label.set_position((0.1, 0.5))
        ax3d.set_xlim(x_limits)
        ax3d.set_ylim(y_limits)
        ax3d.set_zlim(z_limits)
        ax3d.view_init(elev=45, azim=-45)
        # ax3d.set_title("Worker Positions (3D View)")
        fig3d.subplots_adjust(left=0.1, right=0.65, top=0.9, bottom=0.15)
        
        # Draw the ground plane
        xx, yy = np.meshgrid(
            np.linspace(x_limits[0], x_limits[1], 10),
            np.linspace(y_limits[0], y_limits[1], 10)
        )
        zz = np.full_like(xx, z_level)
        ax3d.plot_surface(xx, yy, zz, color='peru', alpha=0.3, rstride=100, cstride=100)
        
        # Draw environment boxes in 3D
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
            rect_zs = [z_level] * 5
            verts = [list(zip(rect_xs, rect_ys, rect_zs))]
            poly = Poly3DCollection(verts, facecolors=box_color, alpha=box_alpha, edgecolors='k')
            ax3d.add_collection3d(poly)
        
        # Plot each worker on the 3D axes
        for (wid, wx, wy, wz) in workers:
            color = WORKER_COLORS.get(wid, 'black')
            marker = worker_marker_styles.get(wid, 'x')
            
            if marker == 'zone':
                size = WORKER_SQUARE_SIZE
                sq_xs = [wx, wx + size, wx + size, wx, wx]
                sq_ys = [wy, wy, wy - size, wy - size, wy]
                sq_zs = [wz] * 5
                verts = [list(zip(sq_xs, sq_ys, sq_zs))]
                poly = Poly3DCollection(verts, facecolors=color, alpha=0.7, edgecolors=color)
                ax3d.add_collection3d(poly)
            else:
                ax3d.scatter(wx, wy, wz, color=color, marker=marker, s=50)
        
        ax3d.legend(legend_handles, legend_labels)

    plt.close(fig2d)
    if fig3d is not None:
        plt.close(fig3d)
    return fig3d, fig2d

def segment_data(pos_data, debug=False):
    """
    Segments pos_data (a list of dictionaries) into instance dictionaries representing robot operations.
    
    Each instance dictionary has:
      - "instance": a unique instance name, e.g., "Wall_1", "Foundation", etc.
      - "element": the element (e.g., "Wall").
      - For special elements (here, "scanning_site"), only one segment (with state '-') is expected.
      - For non-special elements, consecutive raw segments (each with a distinct state) are grouped together.
    
    New rules:
      - For assembly and reassembly, valid states are:
            'search', 'pick', 'swing', 'place', 'swing_back'
      - For disassembly, valid states are:
            'pick', 'swing', 'place', 'swing_back'
      - If the element is 'scanning_site', its state is forced to '-'
    
    Returns:
        A list of instance dictionaries in the order they appear.
    """
    # --- Phase 1: Build raw segments ---
    raw_segments = []
    i = 0
    while i < len(pos_data):
        # Skip packets with None for element or state.
        if pos_data[i].get('element') is None or pos_data[i].get('state') is None:
            i += 1
            continue

        # Get and clean current element and state.
        current_element = pos_data[i]['element'].strip()
        current_state = pos_data[i]['state'].strip()
        # Force state to '-' for "scanning_site" (case-insensitive).
        if current_element.lower() == "scanning_site":
            current_state = "-"

        # Start new segment.
        seg = [pos_data[i]]
        i += 1

        # Add consecutive packets with matching element and state.
        while i < len(pos_data):
            if pos_data[i].get('element') is None or pos_data[i].get('state') is None:
                i += 1
                continue
            elem = pos_data[i]['element'].strip()
            st = pos_data[i]['state'].strip()
            # Force state for scanning_site.
            if elem.lower() == "scanning_site":
                st = "-"
            if elem == current_element and st == current_state:
                seg.append(pos_data[i])
                i += 1
            else:
                break
        raw_segments.append({
            "element": current_element,
            "state": current_state,
            "data": seg
        })

    # --- Phase 2: Group raw segments into instance dictionaries ---
    instances = []
    # Define special elements that are handled individually.
    special = {"scanning_site"}
    counters = {}  # to count instances per element, e.g., "Wall": count

    j = 0
    while j < len(raw_segments):
        seg = raw_segments[j]
        element = seg["element"]
        state = seg["state"]

        # If it's a special element (e.g., scanning_site), record the segment as its own instance.
        if element.lower() in special:
            instance_dict = {
                "instance": element,  # Use the element name directly (or add numbering if desired)
                "element": element,
                state: seg["data"]
            }
            instances.append(instance_dict)
            j += 1
        else:
            # For non-special elements, group consecutive segments with the same element.
            counters[element] = counters.get(element, 0) + 1
            instance_name = f"{element}"
            instance_dict = {
                "instance": instance_name,
                "element": element,
            }
            # While the next raw segment belongs to the same element, add it.
            while j < len(raw_segments) and raw_segments[j]["element"] == element:
                current_seg = raw_segments[j]
                st = current_seg["state"]
                # In this grouping, each key (state) is assigned the list of packets.
                instance_dict[st] = current_seg["data"]
                j += 1
            instances.append(instance_dict)

    if debug:
        for instance in instances:
            print("Instance:", instance["instance"])
            for key, value in instance.items():
                if key not in ["instance", "element"]:
                    print(f"  State '{key}' packet count: {len(value)}")
            print()
                
    return instances


def compute_jacobian(q, DH_params):
    """
    Computes the 6x6 Jacobian for a 6-DOF manipulator given joint angles q
    and a list of DH parameter dictionaries. For each joint i, the total joint angle is
    DH_params[i]['theta'] + q[i].
    """
    T = np.eye(4)
    origins = [np.array([0, 0, 0])]
    z_axes = [np.array([0, 0, 1])]  # z0

    for i in range(len(q)):
        theta = DH_params[i]['theta'] + q[i]
        a = DH_params[i]['a']
        d = DH_params[i]['d']
        alpha = DH_params[i]['alpha']
        
        # Standard DH transformation matrix
        T_i = np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0,             np.sin(alpha),                np.cos(alpha),               d],
            [0,             0,                           0,                           1]
        ])
        T = T @ T_i
        origins.append(T[:3, 3])
        z_axes.append(T[:3, 2])
    
    p_end = origins[-1]
    J = np.zeros((6, len(q)))
    for i in range(len(q)):
        z = z_axes[i]
        p = origins[i]
        Jv = np.cross(z, (p_end - p))
        Jw = z
        J[:3, i] = Jv
        J[3:, i] = Jw
    return J


def segment_velocity_acceleration_plot(datasegment, first_packet, debug=False):
    """
    Computes joint and Cartesian velocities and accelerations from a datasegment,
    and returns two separate figures: one for velocity and one for acceleration.

    It uses the state key from the first_packet (e.g. 'search', 'pick', etc.) to
    select the list of dictionaries within the segment. If that key is not found,
    it will try to use any available state key (ignoring 'instance' and 'element').
    
    Parameters:
        datasegment (dict): A dictionary representing a segment with keys for each state.
        first_packet (dict): A reference packet from one of the states in the segment.
        debug (bool): If True, figures will remain open; otherwise they are closed before return.
    
    Returns:
        fig_vel, fig_acc, times, cartesian_velocities, cartesian_accelerations
            where fig_vel and fig_acc are the Matplotlib figures for velocity and acceleration.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        # Determine state key: use the one from first_packet.
        state_key = first_packet.get('state', "").strip()
        if state_key not in datasegment:
            # If not found, try any key except 'instance' and 'element'
            possible_keys = [key for key in datasegment if key not in ['instance', 'element']]
            if possible_keys:
                state_key = possible_keys[0]
            else:
                raise ValueError("No valid state keys found in segment.")
        data = datasegment.get(state_key, [])
        if not data:
            raise ValueError("No data found for state key: " + state_key)

        n = len(data)
        t0 = first_packet['timestamp_send']
        times = np.array([pkt['timestamp_send'] - t0 for pkt in data])
        joint_angles = np.array([pkt['joints'] for pkt in data])
        joint_velocities = np.gradient(joint_angles, times, axis=0)
        joint_accelerations = np.gradient(joint_velocities, times, axis=0)

        # Compute Jacobians for each set of joint angles (assumes DH_params is defined)
        jacobians = [compute_jacobian(q, DH_params) for q in joint_angles]
        jacobians = np.array(jacobians)

        # Compute Cartesian velocities
        cartesian_velocities = np.array([jacobians[i] @ joint_velocities[i] for i in range(n)])
        
        # Compute Cartesian accelerations via finite differences.
        cartesian_accelerations = []
        for i in range(n - 1):
            dt = times[i+1] - times[i]
            J_dot = (jacobians[i+1] - jacobians[i]) / dt if dt != 0 else np.zeros_like(jacobians[i])
            a_cart = jacobians[i] @ joint_accelerations[i] + J_dot @ joint_velocities[i]
            cartesian_accelerations.append(a_cart)
        # Append the last acceleration assuming J_dot is zero.
        cartesian_accelerations.append(jacobians[-1] @ joint_accelerations[-1])
        cartesian_accelerations = np.array(cartesian_accelerations)

        linear_vel = np.linalg.norm(cartesian_velocities[:, :3], axis=1)
        angular_vel = np.linalg.norm(cartesian_velocities[:, 3:], axis=1)
        linear_acc = np.linalg.norm(cartesian_accelerations[:, :3], axis=1)
        angular_acc = np.linalg.norm(cartesian_accelerations[:, 3:], axis=1)
        
    # --- Plot Velocity ---
    fig_vel, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(times, linear_vel, label='Linear Velocity')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Linear Velocity [m/s]')
    ax1.grid(True)
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))

    ax2.plot(times, angular_vel, label='Angular Velocity')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Angular Velocity [rad/s]')
    ax2.grid(True)
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    fig_vel.tight_layout(rect=[0, 0, 0.9, 1])
    
    # --- Plot Acceleration ---
    fig_acc, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 8))
    ax3.plot(times, linear_acc, label='Linear Acceleration')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Linear Acceleration [m/s²]')
    ax3.grid(True)
    ax3.legend(loc='upper left', bbox_to_anchor=(1, 1))

    ax4.plot(times, angular_acc, label='Angular Acceleration')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Angular Acceleration [rad/s²]')
    ax4.grid(True)
    ax4.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    fig_acc.tight_layout(rect=[0, 0, 0.9, 1])
    
    if debug:
        plt.show()
    
    plt.close(fig_vel)
    plt.close(fig_acc)
    
    return fig_vel, fig_acc, times, cartesian_velocities, cartesian_accelerations


def complete_velocity_acceleration_plot(data, debug=False):
    """
    Computes joint and Cartesian velocities and accelerations from a list of dicts,
    and returns two separate figures: one for velocity, one for acceleration.
    
    Each dict in 'data' is expected to have:
      - 'timestamp_send': a timestamp value
      - 'joints': a list/array of joint angles
      
    Note: This function assumes that the function 'compute_jacobian'
    and the variable 'DH_params' are available in the scope.
    """

    # Calculate relative times from the first packet's timestamp.
    t0 = data[0]['timestamp_send']
    times = np.array([pkt['timestamp_send'] - t0 for pkt in data])
    
    # Extract joint angles from each dictionary.
    joint_angles = np.array([pkt['joints'] for pkt in data])
    
    # Compute joint velocities and accelerations using np.gradient.
    joint_velocities = np.gradient(joint_angles, times, axis=0)
    joint_accelerations = np.gradient(joint_velocities, times, axis=0)
    
    # Compute Jacobians for each set of joint angles.
    jacobians = [compute_jacobian(q, DH_params) for q in joint_angles]
    jacobians = np.array(jacobians)
    
    # Compute Cartesian velocities.
    n = len(data)
    cartesian_velocities = np.array([jacobians[i] @ joint_velocities[i] for i in range(n)])
    
    # Compute Cartesian accelerations.
    cartesian_accelerations = []
    for i in range(n - 1):
        dt = times[i+1] - times[i]
        # Compute the time derivative of the Jacobian.
        J_dot = (jacobians[i+1] - jacobians[i]) / dt if dt != 0 else np.zeros_like(jacobians[i])
        a_cart = jacobians[i] @ joint_accelerations[i] + J_dot @ joint_velocities[i]
        cartesian_accelerations.append(a_cart)
    # Append the last acceleration assuming J_dot is zero.
    cartesian_accelerations.append(jacobians[-1] @ joint_accelerations[-1])
    cartesian_accelerations = np.array(cartesian_accelerations)
    
    # Compute norms for plotting (first 3: linear, last 3: angular).
    linear_vel = np.linalg.norm(cartesian_velocities[:, :3], axis=1)
    angular_vel = np.linalg.norm(cartesian_velocities[:, 3:], axis=1)
    linear_acc = np.linalg.norm(cartesian_accelerations[:, :3], axis=1)
    angular_acc = np.linalg.norm(cartesian_accelerations[:, 3:], axis=1)
    
    # --- Velocity Figure ---
    fig_vel, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Linear velocity subplot.
    ax1.plot(times, linear_vel, label='Linear Velocity')
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Linear Velocity [m/s]')
    # ax1.set_title("Linear Velocity")
    ax1.grid(True)
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # Angular velocity subplot.
    ax2.plot(times, angular_vel, label='Angular Velocity')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Angular Velocity [rad/s]')
    # ax2.set_title("Angular Velocity")
    ax2.grid(True)
    ax2.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    fig_vel.tight_layout(rect=[0, 0, 0.9, 1])
    
    # --- Acceleration Figure ---
    fig_acc, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Linear acceleration subplot.
    ax3.plot(times, linear_acc, label='Linear Acceleration')
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Linear Acceleration [m/s²]')
    # ax3.set_title("Linear Acceleration")
    ax3.grid(True)
    ax3.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # Angular acceleration subplot.
    ax4.plot(times, angular_acc, label='Angular Acceleration')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Angular Acceleration [rad/s²]')
    # ax4.set_title("Angular Acceleration")
    ax4.grid(True)
    ax4.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    fig_acc.tight_layout(rect=[0, 0, 0.9, 1])
    
    if debug:
        plt.show()
    
    plt.close(fig_vel)
    plt.close(fig_acc)
    
    return fig_vel, fig_acc, times, cartesian_velocities, cartesian_accelerations

def analyse_segment(_pos_data, construction_type="assembly"):
    """
    Processes segmented positional data (_pos_data) that has been segmented based on
    the new state definitions.
    
    Depending on the construction type, the expected state keys in each segment are:
      - Assembly / Reassembly:
          ['search', 'pick', 'swing', 'place', 'swing_back']
      - Disassembly:
          ['pick', 'swing', 'place', 'swing_back']
      
    If the element is 'scanning_site', then its only key is '-' (one state).
    
    For each segment, the function:
      1. Determines a reference packet based on the first available state (in a defined order).
      2. Creates kinematic plots (velocity and acceleration) using that reference.
      3. Generates trajectory plots for every available state in the segment.
      4. Optionally (for the first segment) creates a FOV plot using the first available state's data.
      
    Parameters
    ----------
    _pos_data : list of dict
        The raw (or pre-segmented) positional data.
    construction_type : str, optional
        "assembly" or "reassembly" (default) or "disassembly" that determines which
        state order to use.
      
    Returns
    -------
    list of dict
        A list of packets, where each packet is a dictionary with keys:
          - 'segment'
          - 'kinematic plots'
          - 'trajectory_plots'
          - optionally 'FOV'
    """
    # Generate segments using the segmentation routine.
    segments = segment_data(_pos_data)
    plots = []
    
    # Define the default state ordering based on construction type.
    if construction_type.lower() == "disassembly":
        default_state_order = ['pick', 'swing', 'place', 'swing_back']
    else:  # For "assembly" or "reassembly"
        default_state_order = ['search', 'pick', 'swing', 'place', 'swing_back']
    
    # Process each segment.
    for i, segment in enumerate(segments):
        print(segments[i]['instance'])
        
        # Determine the state order to use:
        # If the element is 'scanning_site', force state order to ['-']
        element = segment.get('element')
        if element == 'scanning_site':
            state_order = ['-']
        else:
            state_order = default_state_order
        
        # Choose the reference packet: the first available packet under the first state (in our order)
        ref_packet = None
        for state in state_order:
            if state in segment and len(segment[state]) > 0:
                ref_packet = segment[state][0]
                break
        if ref_packet is None:
            print(f"Skipping segment {segment['instance']} due to missing expected states: {state_order}")
            continue
        
        # Kinematic plots: generate velocity and acceleration plots using the ref_packet.
        fig_vel, fig_acc, _, _, _ = segment_velocity_acceleration_plot(segment, ref_packet)
        
        # Trajectory plots: For every state (in our ordering) present in the segment,
        # create a trajectory plot.
        figures = {}
        for state in state_order:
            if state in segment:
                figures[state] = plot_complete_data(segment[state], segment['instance'])
    
        # Build the output packet for this segment.
        packet = {
            'segment': segment['instance'],
            'kinematic plots': [fig_vel, fig_acc],
            'trajectory_plots': figures,
        }
        
        # For the very first segment, add an FOV plot if available.
        if i == 0 and state_order[0] in segment:
            fov_plotter = FOVPlotter()
            FOV_figures = fov_plotter.plot_FOV(segment[state_order[0]], view="2d", representation="area")
            packet['FOV'] = FOV_figures
        
        plots.append(packet)
    
    return plots

class FOVPlotter:
    def __init__(self):
        self.fov_color_rgb = (0.258, 0.961, 0.596)
        self.fov_alpha = 0.0125

    def calculate_offsets(self, ang):
        """Calculate positional offsets based on a given angle (in degrees)."""
        h, k = 0.01560, 0.00041
        A = np.array([-0.021 - h, 0.107 - k])
        angle_radians = np.radians(ang)
        R = np.array([[np.cos(angle_radians), -np.sin(angle_radians)],
                      [np.sin(angle_radians),  np.cos(angle_radians)]])
        rotated_vector = np.dot(R, A)
        B = rotated_vector + np.array([h, k])
        return B[0], B[1]

    def calculate_rotation(self, R_rotated):
        """
        Compute the rotation angle (in degrees) between a given rotation matrix and a reference matrix.
        """
        R_rotated = np.array(R_rotated)
        R_relative = np.dot(R_rotated, R_ZERO.T)
        angle = np.arccos((np.trace(R_relative) - 1) / 2)
        angle_degrees = np.degrees(angle)
        # Determine rotation direction based on the relative axis
        axis = np.array([R_relative[2, 1] - R_relative[1, 2],
                         R_relative[0, 2] - R_relative[2, 0],
                         R_relative[1, 0] - R_relative[0, 1]])
        if axis[2] < 0:
            angle_degrees = -angle_degrees
        return angle_degrees

    def _compute_fov_corners(self, x0, y0, z0, ground_z, alpha_deg):
        """
        Compute the 2D corners of the camera FOV footprint (at a specified ground level).
        Returns a (4,2) array. For 3D plotting, a constant z can be appended.
        """
        alpha = np.radians(alpha_deg)
        d = z0 - ground_z
        th_h_2 = np.radians(FOV_horizontal / 2.0)
        th_v_2 = np.radians(FOV_vertical / 2.0)
        w_half = d * np.tan(th_h_2)
        h_half = d * np.tan(th_v_2)
        corners_local = np.array([
            [ w_half,  h_half],
            [-w_half,  h_half],
            [-w_half, -h_half],
            [ w_half, -h_half]
        ])
        # Rotate the local corners
        R = np.array([[np.cos(alpha), -np.sin(alpha)],
                      [np.sin(alpha),  np.cos(alpha)]])
        corners_rot = corners_local @ R.T
        # Translate to world coordinates
        return corners_rot + np.array([x0, y0])

    def draw_environment_2d(self, ax):
        """Draw the 2D environment: ground and boxes."""
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits[0] - 0.2, y_limits[1] + 0.2)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        # Draw ground
        ground = Rectangle((x_limits[0], y_limits[0]- 0.2),
                           x_limits[1] - x_limits[0],
                           (y_limits[1] + 0.2) - (y_limits[0] - 0.2),
                           facecolor='peru', alpha=0.3)
        ax.add_patch(ground)
        # Draw boxes
        for (cx, cy), (w, h) in zip(box_positions, box_dims):
            x1, y1 = cx - w/2, cy - h/2
            rect = Rectangle((x1, y1), w, h, facecolor=box_color,
                             alpha=box_alpha, edgecolor='k', linewidth=2)
            ax.add_patch(rect)

    def draw_environment_3d(self, ax):
        """Draw the 3D environment: ground surface and boxes."""
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits[0] - 0.2, y_limits[1])
        ax.set_zlim(z_limits)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.view_init(elev=35, azim=-45)
        # Draw ground surface
        xx, yy = np.meshgrid(np.linspace(x_limits[0], x_limits[1], 10),
                             np.linspace(y_limits[0], y_limits[1], 10))
        ax.plot_surface(xx, yy, np.full_like(xx, z_level), color='peru',
                        alpha=0.3, rstride=1, cstride=1)
        # Draw boxes
        for (cx, cy), (w, h) in zip(box_positions, box_dims):
            x1, x2 = cx - w/2, cx + w/2
            y1, y2 = cy - h/2, cy + h/2
            verts = [list(zip([x1, x2, x2, x1, x1],
                              [y1, y1, y2, y2, y1],
                              [z_level + 0.06] * 5))]
            ax.add_collection3d(
                Poly3DCollection(verts, facecolors=box_color,
                                 alpha=box_alpha, edgecolors='k')
            )

    def _apply_packet_transform(self, packet):
        """
        Extract the camera's coordinates and orientation from a packet, apply the
        translation and offset correction, and return (x0, y0, z0, alpha_deg).
        """
        coords = np.array(packet.get("coordinates", [0, 0, 0]), dtype=float) + TRANSLATION
        x0, y0, z0 = coords
        alpha_deg = self.calculate_rotation(packet["orientation"])
        x_shift, y_shift = self.calculate_offsets(alpha_deg)
        return x0 - x_shift, y0 - y_shift, z0, alpha_deg

    def plot_area_2d(self, packets):
        """Plot the FOV footprints as filled areas in 2D (plan view)."""
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        self.draw_environment_2d(ax)
        ax.set_title("Camera FOV (Plan View)")
        ax.set_ylim(y_limits[0] - 0.2, y_limits[1])
        for packet in packets:
            x0, y0, z0, alpha_deg = self._apply_packet_transform(packet)
            # Use a slightly lower ground level for area plotting (as in your code)
            corners = self._compute_fov_corners(x0, y0, z0, z_level - 0.15, alpha_deg)
            ax.fill(corners[:, 0], corners[:, 1],
                    facecolor=self.fov_color_rgb, alpha=self.fov_alpha,
                    edgecolor=self.fov_color_rgb, linewidth=2)
        plt.close(fig)
        return fig

    def plot_area_3d(self, packets):
        """Plot the FOV footprints as filled areas in 3D (isometric view)."""
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection='3d')
        self.draw_environment_3d(ax)
        ax.set_title("Camera FOV (Orthogonal View)")
        for packet in packets:
            x0, y0, z0, alpha_deg = self._apply_packet_transform(packet)
            corners = self._compute_fov_corners(x0, y0, z0, z_level - 0.15, alpha_deg)
            # Append constant z for 3D display
            corners_3d = np.hstack([corners, np.full((corners.shape[0], 1), z_level - 0.15)])
            ax.add_collection3d(
                Poly3DCollection([list(corners_3d)], facecolors=self.fov_color_rgb,
                                 alpha=self.fov_alpha - 0.004,
                                 edgecolors=self.fov_color_rgb, linewidth=2)
            )
        plt.close(fig)
        return fig

    def plot_boundaries(self, packets, view="both"):
        """
        Compute the union of all FOV footprints (as 2D polygons) and plot its boundary.
        The view can be "2d", "3d", or "both" (a single figure with two subplots).
        """
        union_poly = None
        for packet in packets:
            x0, y0, z0, alpha_deg = self._apply_packet_transform(packet)
            # For boundary, use the base ground level (z_level)
            corners = self._compute_fov_corners(x0, y0, z0, z_level, alpha_deg)
            poly = Polygon(corners)
            union_poly = poly if union_poly is None else union_poly.union(poly)
        if union_poly is None:
            print("No polygons found.")
            return

        boundary = union_poly.boundary

        # Helper functions for boundary plotting
        def plot_2d_boundary(ax, boundary_obj, color='green'):
            if boundary_obj.geom_type == 'MultiLineString':
                for line in boundary_obj.geoms:
                    x, y = line.xy
                    ax.plot(x, y, color=color, linewidth=2)
            elif boundary_obj.geom_type == 'LineString':
                x, y = boundary_obj.xy
                ax.plot(x, y, color=color, linewidth=2)
            else:
                try:
                    for geom in boundary_obj.geoms:
                        xx, yy = geom.xy
                        ax.plot(xx, yy, color=color, linewidth=2)
                except Exception as e:
                    print("Error plotting 2D boundary:", e)

        def plot_3d_boundary(ax, boundary_obj, color='green', z_const=None):
            if z_const is None:
                z_const = z_level
            if boundary_obj.geom_type == 'MultiLineString':
                for line in boundary_obj.geoms:
                    x, y = line.xy
                    coords_3d = np.column_stack([x, y, [z_const]*len(x)])
                    ax.plot(coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2],
                            color=color, linewidth=2)
            elif boundary_obj.geom_type == 'LineString':
                x, y = boundary_obj.xy
                ax.plot(x, y, [z_const]*len(x), color=color, linewidth=2)
            else:
                try:
                    for geom in boundary_obj.geoms:
                        xx, yy = geom.xy
                        ax.plot(xx, yy, [z_const]*len(xx), color=color, linewidth=2)
                except Exception as e:
                    print("Error plotting 3D boundary:", e)

        view = view.lower()
        if view == "both":
            fig = plt.figure(figsize=(12, 6))
            ax3d = fig.add_subplot(1, 2, 1, projection='3d')
            self.draw_environment_3d(ax3d)
            ax3d.set_title("Camera FOV (Orthogonal View)")
            ax2d = fig.add_subplot(1, 2, 2)
            self.draw_environment_2d(ax2d)
            ax2d.set_title("Camera FOV (Plan View)")
            plot_2d_boundary(ax2d, boundary, color='green')
            plot_3d_boundary(ax3d, boundary, color='green', z_const=z_level + 0.001)

        elif view == "2d":
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111)
            self.draw_environment_2d(ax)
            ax.set_title("Camera FOV (Plan View)")
            plot_2d_boundary(ax, boundary, color='green')

        elif view == "3d":
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection='3d')
            self.draw_environment_3d(ax)
            ax.set_title("Camera FOV (Orthogonal View)")
            plot_3d_boundary(ax, boundary, color='green', z_const=z_level + 0.001)
        else:
            raise ValueError("Invalid view option for boundaries. Choose '2d', '3d', or 'both'.")
        
        plt.close(fig)
        return fig

    def plot_FOV(self, packets, view="both", representation="area"):
        """
        Plot FOV footprints using a single function.
        
        Parameters:
            packets: list of packet dictionaries (each should include "coordinates" and "orientation")
            view: "2d", "3d", or "both"
            representation: "area" for filled footprints or "boundary" for the union perimeter
            
        Returns:
            A list of matplotlib Figure objects (or a single figure) depending on the options.
        """
        representation = representation.lower()
        view = view.lower()
        if representation == "area":
            if view == "2d":
                return [self.plot_area_2d(packets)]
            elif view == "3d":
                return [self.plot_area_3d(packets)]
            elif view == "both":
                return [self.plot_area_2d(packets), self.plot_area_3d(packets)]
            else:
                raise ValueError("Invalid view option. Choose '2d', '3d', or 'both'.")
        elif representation == "boundary":
            # For boundary, the method returns a single figure (with one or two subplots)
            return [self.plot_boundaries(packets, view=view)]
        else:
            raise ValueError("Invalid representation type. Choose 'area' or 'boundary'.")
        
    def camera_footprint_on_plane(self, x0, y0, z0, ground_z, theta_h, theta_v, alpha_deg):
        """
        Compute the 4 corner points (X, Y, Z=ground_z) of the camera's rectangular
        FOV footprint on a horizontal plane at Z = ground_z.
        """
        alpha = np.radians(alpha_deg)
        d = z0 - ground_z
        th_h_2 = np.radians(theta_h / 2.0)
        th_v_2 = np.radians(theta_v / 2.0)
        w_half = d * np.tan(th_h_2)
        h_half = d * np.tan(th_v_2)
        # 4 corners in camera's local XY (without rotation)
        corners_local = np.array([
            [ +w_half, +h_half],
            [ -w_half, +h_half],
            [ -w_half, -h_half],
            [ +w_half, -h_half]
        ])
        # Rotate the local corners by the roll angle alpha
        cosA, sinA = np.cos(alpha), np.sin(alpha)
        R = np.array([[cosA, -sinA],
                      [sinA,  cosA]])
        corners_rot = corners_local @ R.T  # shape (4, 2)
        # Translate to world coordinates (append ground_z)
        corners_world = np.array([[x0 + cx, y0 + cy, ground_z] for cx, cy in corners_rot])
        
        print("FOV corner points (Global coordinates):")
        for i, (x, y, z) in enumerate(corners_world):
            print(f"Corner {i+1}: X={x:.4f}, Y={y:.4f}, Z={z:.4f}")
        return corners_world
        
    def plot_camera_fov_at_location(self,x0, y0, z0, theta_h, theta_v, alpha_deg = 0.0, ground_z=z_level):
        """
        Plot the camera's field-of-view (FOV) footprint for a specific location.
        
        Parameters:
            x0, y0, z0 : float
                Camera position coordinates.
            theta_h, theta_v : float
                Horizontal and vertical FOV angles (in degrees).
            alpha_deg : float
                Camera rotation (roll) in degrees.
            ground_z : float, optional
                Z coordinate of the ground plane (default is z_level).
        
        Returns:
            fig : matplotlib Figure object containing the 3D and 2D plots.
        """
        # Compute FOV footprint on the ground using the provided helper function
        corners = self.camera_footprint_on_plane(x0, y0, z0, ground_z, theta_h, theta_v, alpha_deg)

        # Create a figure with 2 subplots side by side
        fig = plt.figure(figsize=(12, 6))
        plt.subplots_adjust(wspace=0.3)

        # Left subplot: 3D environment
        ax3d = fig.add_subplot(1, 2, 1, projection='3d')
        self.draw_environment_3d(ax3d)
        ax3d.scatter(x0, y0, z0, color='red', s=25, label='Camera Center')

        # Draw the FOV projection (perimeter and lines from camera center) in 3D
        for i in range(len(corners)):
            cx, cy, cz = corners[i]
            nx, ny, nz = corners[(i + 1) % len(corners)]
            ax3d.plot([cx, nx], [cy, ny], [cz, nz], color='green')  # Perimeter edge
            ax3d.plot([x0, cx], [y0, cy], [z0, cz], color='gray', linestyle='--')

        ax3d.legend()
        ax3d.set_title("3D Environment + Camera FOV")

        # Right subplot: 2D top-down environment
        ax2d = fig.add_subplot(1, 2, 2)
        self.draw_environment_2d(ax2d)
        for i in range(len(corners)):
            cx, cy, _ = corners[i]
            nx, ny, _ = corners[(i + 1) % len(corners)]
            ax2d.plot([cx, nx], [cy, ny], color='green')

        ax2d.scatter([x0], [y0], color='red', s=25, label='Camera Center')
        ax2d.legend()
        ax2d.set_title("2D Top-Down View + Camera FOV")

        # Optionally, add a filled polygon for the FOV in both views
        fov_corners = [tuple(c) for c in corners]  # Convert corners to tuples for 3D
        ax3d.add_collection3d(
            Poly3DCollection([fov_corners], facecolors='green', alpha=0.7,
                            edgecolors='green', linewidth=2)
        )

        fov_corners_array = np.array(corners)
        ax2d.fill(fov_corners_array[:, 0], fov_corners_array[:, 1],
                facecolor='green', alpha=0.7, edgecolor='green', linewidth=2)

        plt.show()
        return fig


