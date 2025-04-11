## ui/utils.py

import os
import re
from pathlib import Path
import pickle

import base64

import threading
import time

import pandas as pd

from matplotlib import pyplot as plt
from IPython.display import display
from matplotlib.figure import Figure

import ui
from construct import analysis
import camera

from ui.explanations import explanations 

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill, Border, Side
from openpyxl.utils import get_column_letter

import webbrowser

def analyze_trajectory(options,construction_type):
    """
    Performs various types of analyses and plotting on trajectory and detection data 
    based on the specified options.

    Args:
        options (dict): A dictionary containing flags to enable different types of analyses,
                        and the input data required for processing.
            Expected keys:
                - 'data' (DataFrame): The input dataset containing trajectory and detection information.
                - 'trajectory_analysis' (bool): If True, performs trajectory metric analysis and plots kinematic data.
                - 'hazard_analysis' (bool): If True, processes and visualizes hazard events (used with trajectory_analysis).
                - 'detection_analysis' (bool): If True, performs worker detection analysis.
                - 'plot_full_run' (bool): If True, plots the full trajectory run.
                - 'segment_analysis' (bool): If True, generates segment-wise analysis plots.
                - 'plot_workers' (bool): If True, plots worker positions on the site.

    Returns:
        dict: A dictionary containing results based on the selected analyses, including:
            - 'hazard_summary': Summary table of detected hazard events (if applicable).
            - 'hazard_plot': Plot of hazard event timeline (if applicable).
            - 'trajectory_summary': Metrics and statistics for the full trajectory.
            - 'Complete_kinematic_plots': List containing velocity and acceleration plots.
            - 'worker_detection': Results of worker detection analysis.
            - 'Trajectory plots': Visualization of the complete trajectory.
            - 'segment_plots': Plots from segment-wise analysis.
            - 'worker plot': Visualization of detected worker positions.
    """

    data = options.get("data", None)
    results = {}

    if options.get("trajectory_analysis", False):
        print("[INFO] Starting trajectory analysis...")
        evnt = None
        if options.get("hazard_analysis", False):
            print("[INFO] Performing hazard event processing...")
            _hazard_events = analysis.process_hazard_events(data)
            print("[INFO] Tabulating hazard events...")
            results["hazard_summary"] = analysis.tabulate_hazard_events(_hazard_events)
            print("[INFO] Plotting hazard events...")
            results["hazard_plot"] = analysis.plot_hazard_events(_hazard_events)
            evnt = _hazard_events
        print("[INFO] Summarizing trajectory metrics...")
        results["trajectory_summary"] = analysis.summarize_trajectory_metrics(data, construction_type="assembly", hazard_events=evnt)
        print("[INFO] Plotting velocity and acceleration...")
        fig_vel, fig_acc, _, _, _ = analysis.complete_velocity_acceleration_plot(data)
        results["Complete_kinematic_plots"] = [fig_vel, fig_acc]
        print("[INFO] Trajectory analysis completed.")

    if options.get("detection_analysis", False):
        print("[INFO] Starting worker detection analysis...")
        results["worker_detection"] = analysis.workerdetection_analysis(data)
        print("[INFO] Worker detection analysis completed.")

    if options.get("segment_analysis", False):
        print("[INFO] Performing segment-wise analysis...")
        results["segment_plots"] = analysis.analyse_segment(data)
        print("[INFO] Segment analysis completed.")

    if options.get("plot_workers", False):
        print("[INFO] Plotting worker positions...")
        results["worker plot"] = analysis.plot_workers(data,False) # This bool enables a 3D worker detection plot
        print("[INFO] Worker plotting completed.")

    if options.get("plot_full_run", False):
        print("[INFO] Plotting full trajectory run...")
        results["Trajectory plots"] = analysis.plot_complete_data(data, 'Complete')
        print("[INFO] Full run plot completed.")

    print("[INFO] All requested analyses completed.")
    return results


def load_data(n):
    # Retrieve current tracking data.
    data = ui.comms.get_tracking_packets()
    
    # Directory to store logs.
    directory = '/home/avi/Desktop/robomason/_workingdata/_rawtrajectories'
    
    # List files that are pickle files (assuming names like "0.pkl", "1.pkl", etc.)
    files = [f for f in os.listdir(directory) if f.endswith('.pkl')]
    
    # Extract numeric parts from file names.
    log_numbers = []
    for filename in files:
        try:
            # Remove the file extension and convert to an integer.
            log_num = int(os.path.splitext(filename)[0])
            log_numbers.append(log_num)
        except ValueError:
            # If filename does not match our numeric pattern, skip it.
            continue
    
    # Decide on the file name based on the count of existing log files.
    if len(log_numbers) < n:
        # Determine next log number.
        next_log_num = max(log_numbers) + 1 if log_numbers else 0
        file_path = os.path.join(directory, f"{next_log_num}.pkl")
    else:
        # If there are already n logs, delete all existing log files.
        for f in files:
            os.remove(os.path.join(directory, f))
        # Start fresh with file name 0.
        file_path = os.path.join(directory, "0.pkl")
    
    # Save the data to a pickle file.
    with open(file_path, "wb") as file:
        pickle.dump(data, file)
    
    return data

def save_trajectory_summary_excel(df, file_path):
    """
    Save the trajectory summary DataFrame to an Excel file with improved styling using openpyxl.
    
    Args:
        df (pd.DataFrame): The trajectory summary DataFrame.
        file_path (str): Full path (including .xlsx) where the file will be saved.
    """
    # Create a new workbook and select the active worksheet.
    wb = Workbook()
    ws = wb.active
    ws.title = "Trajectory Summary"

    # Define some styles.
    header_font = Font(name='Calibri', bold=True, color="000000")  # black text
    header_fill = PatternFill(start_color="C9C9C9", end_color="C9C9C9", fill_type="solid")
    thin_border = Border(
        left=Side(style='thin'),
        right=Side(style='thin'),
        top=Side(style='thin'),
        bottom=Side(style='thin')
    )
    center_alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    # 1) Write column headers with styling.
    for col_idx, col_name in enumerate(df.columns, start=1):
        cell = ws.cell(row=1, column=col_idx, value=col_name)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = center_alignment
        cell.border = thin_border

    # 2) Write DataFrame rows, applying borders/alignment.
    for row_idx, row in df.iterrows():
        for col_idx, value in enumerate(row, start=1):
            # Convert lists/tuples to string to avoid ValueError
            if isinstance(value, (list, tuple)):
                value = ", ".join(str(x) for x in value)

            cell = ws.cell(row=row_idx + 2, column=col_idx, value=value)
            cell.alignment = center_alignment
            cell.border = thin_border

    # 3) Auto-fit column widths based on max cell length in each column.
    #    Note: openpyxl does not have a built-in auto-fit; we approximate by measuring text length.
    for col in ws.columns:
        max_length = 0
        column_letter = get_column_letter(col[0].column)
        for cell in col:
            cell_value = str(cell.value) if cell.value is not None else ""
            max_length = max(max_length, len(cell_value))
        # A little extra spacing
        ws.column_dimensions[column_letter].width = max_length + 2

    # 4) Freeze the top row so headers remain visible when scrolling.
    ws.freeze_panes = "A2"

    # 5) Save the workbook.
    wb.save(file_path)

def save_worker_detection_excel(df, file_path):
    """
    Save the worker detection DataFrame to an Excel file with improved styling using openpyxl.
    - The 'first detected at location (time)' column is modified to show only the time (no coordinates).
    - The header row height is increased to accommodate wrapped text.
    
    Args:
        df (pd.DataFrame): The worker detection DataFrame.
        file_path (str): Full path (including .xlsx) where the file will be saved.
    """
    # Create a new workbook and select the active worksheet.
    wb = Workbook()
    ws = wb.active
    ws.title = "Worker Detection"

    # Define some styles.
    header_font = Font(name='Calibri', bold=True, color="000000")  # black text
    header_fill = PatternFill(start_color="C9C9C9", end_color="C9C9C9", fill_type="solid")
    thin_border = Border(
        left=Side(style='thin'),
        right=Side(style='thin'),
        top=Side(style='thin'),
        bottom=Side(style='thin')
    )
    center_alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    # 1) Write column headers with styling.
    for col_idx, col_name in enumerate(df.columns, start=1):
        cell = ws.cell(row=1, column=col_idx, value=col_name)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = center_alignment
        cell.border = thin_border

    # Increase the header row height so wrapped text is fully visible.
    ws.row_dimensions[1].height = 30

    # 2) Write DataFrame rows, applying borders/alignment.
    for row_idx, row in df.iterrows():
        for col_idx, value in enumerate(row, start=1):
            # Convert lists/tuples to string to avoid ValueError
            if isinstance(value, (list, tuple)):
                value = ", ".join(str(x) for x in value)

            # If this is the "first detected at location (time)" column, strip out coordinates, keep only time
            col_header = df.columns[col_idx - 1]
            if col_header == "first detected at location (time)" and isinstance(value, str):
                # Look for text inside parentheses (time) and replace the cell value with just that
                match = re.search(r"\((.*?)\)", value)
                if match:
                    value = match.group(1)  # only the time portion

            cell = ws.cell(row=row_idx + 2, column=col_idx, value=value)
            cell.alignment = center_alignment
            cell.border = thin_border

    # 3) Auto-fit column widths based on max cell length in each column.
    for col in ws.columns:
        max_length = 0
        column_letter = get_column_letter(col[0].column)
        for cell in col:
            cell_value = str(cell.value) if cell.value is not None else ""
            max_length = max(max_length, len(cell_value))
        ws.column_dimensions[column_letter].width = max_length + 2

    # 4) Freeze the top row so headers remain visible when scrolling.
    ws.freeze_panes = "A2"

    # 5) Save the workbook.
    wb.save(file_path)


def save_results(results, data, mode='ct'):
    """
    Saves analysis results and raw data to a new subfolder.
    
    The function creates a new subfolder inside
    /home/avi/Desktop/robomason/_workingdata/{subfolder}
    with a name of the form '{run_type}_run_XX', where XX is the next available run number.
    
    DataFrames in results are saved as Excel files.
    Figures (from Matplotlib) are saved as high-quality SVG images.
    The raw data is saved as a pickle file.
    
    Args:
        results (dict): A dictionary with keys:
            - "trajectory_summary": DataFrame to be saved as Excel.
            - "worker_detection": DataFrame to be saved as Excel.
            - "Trajectory plots": List of figures to be saved as SVGs.
            - "segment_plots": List of dictionaries (one per segment). Each dictionary is expected to have:
                   * "segment": a string representing the segment name.
                   * "kinematic plots": a list of figures.
                   * "trajectory_plots": either a dictionary (keys = state names, values = list of figures)
                                          or a list of figures.
                   * Optionally, "FOV": a list of figures.
        data: Raw data to be saved as a pickle file.
        mode (str): Determines the subfolder type. Options:
            - 'ct' (default) -> "_constructionruns"
            - 'dis' -> "_disassemblyruns"
            - 'rect' -> "_reconstructionruns"
            - 'full' -> "_fullruns"
    """
    # Determine the base directory and run type based on mode
    mode_mapping = {
        'ct': ('_constructionruns', 'construction'),
        'dis': ('_disassemblyruns', 'disassembly'),
        'rect': ('_reconstructionruns', 'reconstruction'),
        'full': ('_fullruns', 'complete_construction')
    }
    
    if mode not in mode_mapping:
        raise ValueError("Invalid mode. Choose from 'ct', 'dis', 'rect', or 'full'.")
    
    subfolder, run_type = mode_mapping[mode]
    base_dir = f'/home/avi/Desktop/robomason/_workingdata/{subfolder}'
    os.makedirs(base_dir, exist_ok=True)

    # List all subdirectories matching our naming scheme.
    subfolders = [name for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))]
    run_numbers = []
    pattern = re.compile(fr'{run_type}_run_(\d+)')
    for folder in subfolders:
        match = pattern.match(folder)
        if match:
            run_numbers.append(int(match.group(1)))
    next_run = max(run_numbers) + 1 if run_numbers else 0
    new_folder_name = f"{run_type}_run_{next_run:02d}"
    new_folder_path = os.path.join(base_dir, new_folder_name)
    os.makedirs(new_folder_path, exist_ok=True)
    
    # Save DataFrames
    if "trajectory_summary" in results:
        traj_summary_path = os.path.join(new_folder_path, "trajectory_summary.xlsx")
        save_trajectory_summary_excel(results["trajectory_summary"], traj_summary_path)
    
    if "Complete_kinematic_plots" in results:
        fig_vel, fig_acc = results["Complete_kinematic_plots"]
        vel_path = os.path.join(new_folder_path, "complete_velocity_plot.svg")
        acc_path = os.path.join(new_folder_path, "complete_acceleration_plot.svg")
        if isinstance(fig_vel, Figure):
            fig_vel.savefig(vel_path, format="svg")
        if isinstance(fig_acc, Figure):
            fig_acc.savefig(acc_path, format="svg")
    
    if "hazard_summary" in results:
        results["hazard_summary"].to_excel(os.path.join(new_folder_path, "hazard_summary.xlsx"), index=False)
    
    if "hazard_plot" in results:
        results["hazard_plot"].savefig(os.path.join(new_folder_path, "marker detection events.svg"), format="svg")
    
    if "worker_detection" in results:
        worker_detect_path = os.path.join(new_folder_path, "worker_detection.xlsx")
        save_worker_detection_excel(results["worker_detection"], worker_detect_path)
    
    # Save global trajectory plots
    if "Trajectory plots" in results:
        figures = results["Trajectory plots"]
        view_names = ["Isometric view", "Plan view", "Side view", "Front view"]
        for idx, fig in enumerate(figures):
            if idx < len(view_names):
                fig_path = os.path.join(new_folder_path, f"trajectory_{view_names[idx]}.svg")
                fig.savefig(fig_path, format="svg")
    
    # Save segment-specific plots
    if "segment_plots" in results:
        for segment_dict in results["segment_plots"]:
            # Convert segment identifier to string, then clean it.
            raw_segment = str(segment_dict.get("segment", "Unnamed"))
            clean_segment = raw_segment.strip().replace("/", "-")
            segment_folder = os.path.join(new_folder_path, clean_segment)
            os.makedirs(segment_folder, exist_ok=True)
    
            # Save kinematic plots
            for i, fig in enumerate(segment_dict.get('kinematic plots', [])):
                if isinstance(fig, Figure):
                    fig.savefig(os.path.join(segment_folder, f"plot_{clean_segment}_kinematic_{i}.svg"), format="svg")
    
            # Save trajectory plots
            traj_plots = segment_dict.get("trajectory_plots")
            if traj_plots:
                if isinstance(traj_plots, dict):
                    # For each state key (like "search", "pick", etc.), save its list of figures.
                    for state, figs in traj_plots.items():
                        state_folder = os.path.join(segment_folder, state)
                        os.makedirs(state_folder, exist_ok=True)
                        # If figs is not a list, wrap it.
                        if not isinstance(figs, list):
                            figs = [figs]
                        for i, fig in enumerate(figs):
                            if isinstance(fig, Figure):
                                fig.savefig(os.path.join(state_folder, f"trajectory_{clean_segment}_{state}_{i}.svg"), format="svg")
                elif isinstance(traj_plots, list):
                    # Otherwise, if trajectory_plots is just a list, save them all.
                    for i, fig in enumerate(traj_plots):
                        if isinstance(fig, Figure):
                            fig.savefig(os.path.join(segment_folder, f"trajectory_{clean_segment}_{i}.svg"), format="svg")
    
            # Save FOV plots, if any.
            for i, fig in enumerate(segment_dict.get("FOV", [])):
                if isinstance(fig, Figure):
                    fig.savefig(os.path.join(segment_folder, f"FOV_plot_{i}.svg"), format="svg")
    
    if "worker plot" in results:
        figures = results["worker plot"]
        view_names = ["Isometric view", "Plan view"]
        for idx, fig in enumerate(figures):
            if fig is None:
                continue
            if idx < len(view_names):
                fig_path = os.path.join(new_folder_path, f"worker_{view_names[idx]}.svg")
                fig.savefig(fig_path, format="svg")
    
    # Save the raw data as a pickle file.
    raw_data_path = os.path.join(new_folder_path, "raw_data.pkl")
    with open(raw_data_path, "wb") as file:
        pickle.dump(data, file)
    
    print(f"Results saved in {new_folder_path}")
    return new_folder_path


def show_project_structure():
    directory = "/home/avi/Desktop/robomason"
    for root, _, files in os.walk(directory):
        level = root.replace(directory, "").count(os.sep)
        indent_space = " " * 4 * level
        print(f"{indent_space}- {os.path.basename(root)}/")  # Print folder name
        for file in files:
            print(f"{indent_space}    - {file}") 

def initializeUI():
    ui.system_setup()
    framehandler = camera.FrameHandler(ws_url="ws://localhost:9090", camera_index=4, frame_rate=15, is_sender=False)
    threading.Thread(target=framehandler.start_streaming, daemon=True).start()
    time.sleep(1.5)
    frame = framehandler.get_latest_frame()
    plt.imshow(frame)
    return framehandler

def open_html_in_browser(html_file_path):
    """
    Opens the specified HTML file in the default web browser.
    
    Parameters:
        html_file_path (str): The full path to the HTML file.
        save_path (str): The destination path where the HTML should be saved.
    """
    webbrowser.open(f"file:///{html_file_path}")


def generate_segment(base_path):
    """
    Generates HTML for the segment analysis by iterating over each segment folder
    in the base_path. For each segment, images are embedded in a defined order for 
    the trajectory plots: search, pick, swing, place, swing_back. If a given state 
    folder does not exist (as in disassembly runs which lack "search"), it is skipped.
    
    Parameters:
        base_path (str): The base folder path containing the segment subfolders.
    
    Returns:
        str: HTML snippet with segment analysis.
    """
    # List all segment subfolders in the base_path
    segment_folders = [folder for folder in os.listdir(base_path)
                       if os.path.isdir(os.path.join(base_path, folder))]
    
    # Optional mapping: convert "Toilet_1" etc. into "Bathroom Module 1"
    toilet_to_bathroom = {
        "Toilet_1": "Bathroom Module 1",
        "Toilet_2": "Bathroom Module 2",
        "Toilet_3": "Bathroom Module 3"
    }
    
    # Define the desired state order. If a state folder is absent, it will be skipped.
    ordered_state_names = ["search", "pick", "swing", "place", "swing_back"]
    
    html = ""
    for folder in sorted(segment_folders):
        folder_path = os.path.join(base_path, folder)
        # Use mapping if available
        display_name = toilet_to_bathroom.get(folder, folder)
        html += f"<div class='section'><h3>{display_name}</h3>"
    
        # --- Kinematic Plots ---
        html += "<div class='image-grid'>"
        # Here we assume kinematic plots are named as 'plot_{folder}_kinematic_{i}.svg'.
        # Change the range if you expect more plots.
        for i in range(2):
            kin_plot = os.path.join(folder_path, f"plot_{folder}_kinematic_{i}.svg")
            if os.path.isfile(kin_plot):
                data_uri = encode_image_to_data_uri(kin_plot, mime_type="image/svg+xml")
                if data_uri:
                    html += f'<img src="{data_uri}" alt="Kinematic Plot {i}" style="width:90%; max-width:900px; border:1px solid #ccc;">'
        html += "</div>"
    
        # --- Trajectory Plots (ordered) ---
        # For each state in the ordered list, check if a folder exists and then process it.
        for state in ordered_state_names:
            state_folder = os.path.join(folder_path, state)
            if not os.path.isdir(state_folder):
                continue  # Skip missing state folder (e.g. "search" in disassembly)
            
            # Add a header for this state.
            html += f"<h4 style='margin-top: 20px;'>{state.capitalize()}</h4><div class='image-grid'>"
            files = sorted(os.listdir(state_folder))
            for f in files:
                if f.endswith(".svg"):
                    file_path = os.path.join(state_folder, f)
                    data_uri = encode_image_to_data_uri(file_path, mime_type="image/svg+xml")
                    if data_uri:
                        html += (f'<img src="{data_uri}" alt="{f}" style="width:30%; min-width:280px; max-width:360px; border:1px solid #ccc;">')
            html += "</div>"
        html += "</div>"
    return html

def encode_image_to_data_uri(filepath, mime_type="image/svg+xml"):
    """
    Reads the image file from filepath, encodes it in base64, and returns a data URI.
    
    Parameters:
        filepath (str): Path to the image file.
        mime_type (str): MIME type, default is "image/svg+xml".
    
    Returns:
        str: Data URI string for embedding.
    """
    try:
        with open(filepath, "rb") as f:
            data = f.read()
        encoded = base64.b64encode(data).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"
    except Exception as e:
        print(f"Error encoding {filepath}: {e}")
        return ""

def create_construction_summary_html(segment_folder_path):
    """
    Generates an HTML file with a construction summary based on files in the input folder.
    Embeds images directly as data URIs so that the resulting HTML is a standalone file.
    
    Parameters:
        segment_folder_path (str): The path to the folder that contains the Excel and segment files.
    
    Returns:
        str: The absolute path to the generated HTML file.
    """
    # Get folder name and define the output HTML filename
    folder_name = os.path.basename(os.path.normpath(segment_folder_path))
    output_file = os.path.join(segment_folder_path, "Accumulated data.html")

    parent_folder_1 = os.path.dirname(segment_folder_path)
    parent_folder_2 = os.path.dirname(parent_folder_1)
    diff = os.path.relpath(parent_folder_1, parent_folder_2)
    diff_cleaned = diff.replace("_", " ").replace("runs", " Run").strip().title()
    html_title = f"{diff_cleaned} Summary"

    # Define paths to Excel files and images
    construction_summary_path = os.path.join(segment_folder_path, "trajectory_summary.xlsx")
    worker_detection_path = os.path.join(segment_folder_path, "worker_detection.xlsx")
    hazard_summary_path = os.path.join(segment_folder_path, "hazard_summary.xlsx")
    marker_detection_path = os.path.join(segment_folder_path, "marker detection events.svg")
    
    # Define paths for images in a sibling folder
    base_path_parent = segment_folder_path
    worker_images = [
        os.path.join(base_path_parent, "worker_Isometric view_.svg"),
        os.path.join(base_path_parent, "worker_Plan view.svg")
    ]
    
    trajectory_images = [
        os.path.join(base_path_parent, "trajectory_Isometric view.svg"),
        os.path.join(base_path_parent, "trajectory_Plan view.svg"),
        os.path.join(base_path_parent, "trajectory_Side view.svg"),
        os.path.join(base_path_parent, "trajectory_Front view.svg")
    ]
    
    # List to hold HTML for each section (tile)
    html_sections = []
    
    # --- 1. Kinematics & Trajectory ---
    combined_images = trajectory_images + [
        os.path.join(base_path_parent, "complete_velocity_plot.svg"),
        os.path.join(base_path_parent, "complete_acceleration_plot.svg")
    ]
    available_images = [img for img in combined_images if os.path.exists(img)]
    if available_images:
        traj_section = """
    <div class="section">
        <h2>Kinematics & Trajectory</h2>
        <div class="image-grid">
        """
        for path in available_images:
            data_uri = encode_image_to_data_uri(path, mime_type="image/svg+xml")
            if data_uri:
                alt_text = os.path.splitext(os.path.basename(path))[0].replace("_", " ").capitalize()
                traj_section += f'<img src="{data_uri}" alt="{alt_text}">\n'
        traj_section += """
        </div>
    </div>
        """
        html_sections.append(traj_section)
    
    # --- 2. Construction Summary ---
    if os.path.exists(construction_summary_path):
        try:
            df_construction = pd.read_excel(construction_summary_path)
            html_table_construction = df_construction.to_html(index=False, classes="dataframe", border=0)
            
            # Determine summary type based on parent folder
            parent_folder_1 = os.path.dirname(segment_folder_path)
            parent_folder_2 = os.path.dirname(parent_folder_1)
            diff = os.path.relpath(parent_folder_1, parent_folder_2)
            summary_title_map = {
                "_constructionruns": "Assembly Summary",
                "_disassemblyruns": "Disassembly Summary",
                "_reconstructionruns": "Reassembly Summary"
            }
            section_title = summary_title_map.get(diff, "Construction Summary")

            cons_section = f"""
    <div class="section">
        <h2>{section_title}</h2>
        {html_table_construction}
        <div class="explanation">
            """
            for key, text in explanations.items():
                if "Formula" in key:
                    cons_section += f'<p class="formula">{text}</p>\n'
                else:
                    cons_section += f'<p><strong>{key}</strong>: {text}</p>\n'
            cons_section += """
        </div>
    </div>
            """
            html_sections.append(cons_section)
        except Exception as e:
            print(f"Error processing construction summary: {e}")
    
    # --- 3. Hazard Analysis ---
    if os.path.exists(hazard_summary_path) and os.path.exists(marker_detection_path):
        try:
            df_hazard = pd.read_excel(hazard_summary_path)
            html_table_hazard = df_hazard.to_html(index=False, classes="dataframe", border=0)
            marker_detection_data_uri = encode_image_to_data_uri(marker_detection_path, mime_type="image/svg+xml")
            haz_section = f"""
    <div class="section">
        <h2>Hazard Analysis</h2>
        {html_table_hazard}
        <div class="image-grid" style="margin-top: 20px;">
            <img src="{marker_detection_data_uri}" alt="Marker Detection Events" style="max-width:90%; border:1px solid #ccc;">
        </div>
    </div>
            """
            html_sections.append(haz_section)
        except Exception as e:
            print(f"Error processing hazard analysis: {e}")
    
    # --- 4. Worker Views ---
    available_worker_images = [img for img in worker_images if os.path.exists(img)]
    if available_worker_images:
        worker_views_section = """
    <div class="section">
        <h2>Workers and Work zones</h2>
        <div class="image-grid">
        """
        for path in available_worker_images:
            data_uri = encode_image_to_data_uri(path, mime_type="image/svg+xml")
            if data_uri:
                worker_views_section += f'<img src="{data_uri}" alt="Worker View">\n'
        worker_views_section += """
        </div>
    </div>
        """
        html_sections.append(worker_views_section)
    
    # --- 5. Worker Detection ---
    if os.path.exists(worker_detection_path):
        try:
            df_worker_detection = pd.read_excel(worker_detection_path)
            if df_worker_detection.empty:
                worker_detection_table = "<p>No worker detection data available.</p>"
            else:
                worker_detection_table = df_worker_detection.to_html(index=False, classes="dataframe", border=0)
            worker_det_section = f"""
    <div class="section">
        <h2>Worker Detection</h2>
        {worker_detection_table}
    </div>
            """
            html_sections.append(worker_det_section)
        except Exception as e:
            print(f"Error processing worker detection: {e}")
    
    # --- 6. Segment Analysis ---
    segment_file_list_html = generate_segment(segment_folder_path)
    if segment_file_list_html.strip():
        seg_section = f"""
    <div class="section">
        <h2>Segment Analysis</h2>
        {segment_file_list_html}
    </div>
        """
        html_sections.append(seg_section)
    
    # --- Build the complete HTML content ---
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{html_title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background-color: #f4f4f4;
            margin: 0;
            padding: 20px;
        }}
        h1 {{
            text-align: center;
            margin-bottom: 40px;
        }}
        .section {{
            background: white;
            padding: 20px;
            margin-bottom: 40px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            text-align: center;
            margin-bottom: 20px;
        }}
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 20px;
            justify-items: center;
        }}
        .image-grid img {{
            max-width: 100%;
            height: auto;
            max-height: 70vh;
            border: 1px solid #ccc;
            display:block;
            margin: 0 auto;
        }}
        .image-grid img:only-child {{
            grid-column: 1 / -1;
            justify-self: center;
        }}
        @media (min-width: 768px) {{
            .image-grid img {{
                max-width: 45vw;
            }}
        }}
        .dataframe {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        .dataframe th, .dataframe td {{
            border: 1px solid #ccc;
            padding: 6px 10px;
            text-align: left;
        }}
        .dataframe thead {{
            background-color: #f0f0f0;
        }}
        .dataframe tr:nth-child(even) {{
            background-color: #fafafa;
        }}
        .explanation {{
            margin-top: 20px;
        }}
        .formula {{
            font-style: italic;
            text-align: center;
            margin-top: 10px;
        }}
    </style>
    <script type="text/javascript" async
        src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
    </script>
</head>
<body>
    <h1>{html_title}</h1>
"""
    for section in html_sections:
        html_content += section
    html_content += """
</body>
</html>
    """
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    return os.path.abspath(output_file)


def save_results(results, data, mode):
    """
    Saves analysis results and raw data to a new subfolder.
    
    The function creates a new subfolder inside
    /home/avi/Desktop/robomason/_workingdata/{subfolder}
    with a name of the form '{run_type}_run_XX', where XX is the next available run number.
    
    DataFrames in results are saved as Excel files.
    Figures (from Matplotlib) are saved as high-quality SVG images.
    The raw data is saved as a pickle file.
    
    Args:
        results (dict): A dictionary with keys:
            - "trajectory_summary": DataFrame to be saved as Excel.
            - "worker_detection": DataFrame to be saved as Excel.
            - "Trajectory plots": List of figures to be saved as SVGs.
            - "segment_plots": List of dictionaries representing segment results.
        data: Raw data to be saved as a pickle file.
        mode (str): Determines the subfolder type. Options:
            - 'ct' (default) -> "_constructionruns"
            - 'dis' -> "_disassemblyruns"
            - 'rect' -> "_reconstructionruns"
            - 'full' -> "_fullruns"
    """
    # Determine the base directory and run type based on mode
    mode_mapping = {
        'ct': ('_constructionruns', 'construction'),
        'dis': ('_disassemblyruns', 'disassembly'),
        'rect': ('_reconstructionruns', 'reconstruction'),
        'full': ('_fullruns', 'complete_construction')
    }
    
    if mode not in mode_mapping:
        raise ValueError("Invalid mode. Choose from 'ct', 'dis', 'rect', or 'full'.")
    
    subfolder, run_type = mode_mapping[mode]
    base_dir = f'/home/avi/Desktop/robomason/_workingdata/{subfolder}'
    os.makedirs(base_dir, exist_ok=True)

    subfolders = [name for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))]
    run_numbers = []
    pattern = re.compile(fr'{run_type}_run_(\d+)')
    for folder in subfolders:
        match = pattern.match(folder)
        if match:
            run_numbers.append(int(match.group(1)))
    next_run = max(run_numbers) + 1 if run_numbers else 0
    new_folder_name = f"{run_type}_run_{next_run:02d}"
    new_folder_path = os.path.join(base_dir, new_folder_name)
    os.makedirs(new_folder_path, exist_ok=True)
    
    # Save DataFrames
    if "trajectory_summary" in results:
        traj_summary_path = os.path.join(new_folder_path, "trajectory_summary.xlsx")
        save_trajectory_summary_excel(results["trajectory_summary"], traj_summary_path)
    
    if "Complete_kinematic_plots" in results:
        fig_vel, fig_acc = results["Complete_kinematic_plots"]
        vel_path = os.path.join(new_folder_path, "complete_velocity_plot.svg")
        acc_path = os.path.join(new_folder_path, "complete_acceleration_plot.svg")
        if isinstance(fig_vel, Figure):
            fig_vel.savefig(vel_path, format="svg")
        if isinstance(fig_acc, Figure):
            fig_acc.savefig(acc_path, format="svg")
    
    if "hazard_summary" in results:
        results["hazard_summary"].to_excel(os.path.join(new_folder_path, "hazard_summary.xlsx"), index=False)
    
    if "hazard_plot" in results:
        results["hazard_plot"].savefig(os.path.join(new_folder_path, "marker detection events.svg"), format="svg")
    
    if "worker_detection" in results:
        worker_detect_path = os.path.join(new_folder_path, "worker_detection.xlsx")
        save_worker_detection_excel(results["worker_detection"], worker_detect_path)
    
    if "Trajectory plots" in results:
        figures = results["Trajectory plots"]
        view_names = ["Isometric view", "Plan view", "Side view", "Front view"]
        for idx, fig in enumerate(figures):
            if idx < len(view_names):
                fig_path = os.path.join(new_folder_path, f"trajectory_{view_names[idx]}.svg")
                fig.savefig(fig_path, format="svg")
    
    # Updated segment-specific plots saving logic
    if "segment_plots" in results:
        for segment_dict in results["segment_plots"]:
            raw_segment = str(segment_dict.get("segment", "Unnamed"))
            clean_segment = raw_segment.strip().replace("/", "-")
            segment_folder = os.path.join(new_folder_path, clean_segment)
            os.makedirs(segment_folder, exist_ok=True)
    
            # Save kinematic plots
            for i, fig in enumerate(segment_dict.get('kinematic plots', [])):
                if isinstance(fig, Figure):
                    fig.savefig(os.path.join(segment_folder, f"plot_{clean_segment}_kinematic_{i}.svg"), format="svg")
    
            # Save trajectory plots:
            traj_plots = segment_dict.get("trajectory_plots")
            if traj_plots:
                if isinstance(traj_plots, dict):
                    # Iterate through each state key, which could be "search", "pick", etc.
                    for state_key, figs in traj_plots.items():
                        state_folder = os.path.join(segment_folder, state_key)
                        os.makedirs(state_folder, exist_ok=True)
                        if not isinstance(figs, list):
                            figs = [figs]
                        for i, fig in enumerate(figs):
                            if isinstance(fig, Figure):
                                fig.savefig(os.path.join(state_folder, f"trajectory_{clean_segment}_{state_key}_{i}.svg"), format="svg")
                elif isinstance(traj_plots, list):
                    # Otherwise, if it's a simple list, save all in the segment folder.
                    for i, fig in enumerate(traj_plots):
                        if isinstance(fig, Figure):
                            fig.savefig(os.path.join(segment_folder, f"trajectory_{clean_segment}_{i}.svg"), format="svg")
    
            # Save FOV plots, if any.
            for i, fig in enumerate(segment_dict.get("FOV", [])):
                if isinstance(fig, Figure):
                    fig.savefig(os.path.join(segment_folder, f"FOV_plot_{i}.svg"), format="svg")
    
    if "worker plot" in results:
        figures = results["worker plot"]
        view_names = ["Isometric view", "Plan view"]
        for idx, fig in enumerate(figures):
            if fig is None:
                continue
            if idx < len(view_names):
                fig_path = os.path.join(new_folder_path, f"worker_{view_names[idx]}.svg")
                fig.savefig(fig_path, format="svg")
    
    # Save raw data as pickle.
    raw_data_path = os.path.join(new_folder_path, "raw_data.pkl")
    with open(raw_data_path, "wb") as file:
        pickle.dump(data, file)
    
    print(f"Results saved in {new_folder_path}")
    return new_folder_path

