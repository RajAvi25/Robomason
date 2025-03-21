import os
import re
import pickle

import threading
import time

from matplotlib import pyplot as plt

import ui
from construct import analysis
import camera

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill, Border, Side
from openpyxl.utils import get_column_letter

def analyze_trajectory(options):
    """
    Function to perform trajectory and detection analysis based on provided options.

    Args:
        options (dict): A dictionary containing boolean flags for different analyses and the data.
            Expected keys:
            - 'trajectory_analysis' (bool)
            - 'detection_analysis' (bool)
            - 'plot_full_run' (bool)
            - 'plot_velocity_graphs' (bool)
            - 'plot_acceleration_graphs' (bool)
            - 'data' (DataFrame or relevant data structure)

    Returns:
        dict: A dictionary containing results of the selected analyses.
    """
    data = options.get("data", None)
    results = {}

    if options.get("trajectory_analysis", False):
        results["trajectory_summary"] = analysis.summarize_trajectory_metrics(data)

    if options.get("detection_analysis", False):
        results["worker_detection"] = analysis.workerdetection_analysis(data)

    if options.get("plot_full_run", False):
        results["Trajectory plots"] = analysis.plot_complete_data(data)

    if options.get("plot_velocity_graphs", False):
        results["velocity graph"] = analysis.plot_velocity_segments(data, start_from_zero=False)

    if options.get("plot_acceleration_graphs", False):
        results["accelaration graph"] = analysis.plot_acceleration_segments(data, start_from_zero=False)

    if options.get("plot_workers",False):
        results["worker plot"] = analysis.plot_workers(data)
        # pass

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
            - "velocity graph": A figure saved as SVG.
            - "acceleration graph": A figure saved as SVG.
        data: Raw data to be saved as a pickle file.
        mode (str): Determines the subfolder type. Options:
            - 'ct' (default) -> "_constructionruns"
            - 'dis' -> "_disassemblyruns"
            - 'rect' -> "_reconstructionruns"
    """
    
    # Determine the base directory and run type based on mode
    mode_mapping = {
        'ct': ('_constructionruns', 'construction'),
        'dis': ('_disassemblyruns', 'disassembly'),
        'rect': ('_reconstructionruns', 'reconstruction')
    }
    
    if mode not in mode_mapping:
        raise ValueError("Invalid mode. Choose from 'ct', 'dis', or 'rect'.")
    
    subfolder, run_type = mode_mapping[mode]
    base_dir = f'/home/avi/Desktop/robomason/_workingdata/{subfolder}'
    
    # Ensure base directory exists
    os.makedirs(base_dir, exist_ok=True)

    # List all subdirectories and find those that match our naming scheme.
    subfolders = [name for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))]
    run_numbers = []
    pattern = re.compile(fr'{run_type}_run_(\d+)')
    
    for folder in subfolders:
        match = pattern.match(folder)
        if match:
            run_numbers.append(int(match.group(1)))
    
    # Determine the new run number.
    next_run = max(run_numbers) + 1 if run_numbers else 0
    new_folder_name = f"{run_type}_run_{next_run:02d}"
    new_folder_path = os.path.join(base_dir, new_folder_name)
    os.makedirs(new_folder_path, exist_ok=True)
    
    # Save DataFrames to Excel sheets.
    if "trajectory_summary" in results:
        traj_summary_path = os.path.join(new_folder_path, "trajectory_summary.xlsx")
        save_trajectory_summary_excel(results["trajectory_summary"], traj_summary_path)
    
    if "worker_detection" in results:
        worker_detect_path = os.path.join(new_folder_path, "worker_detection.xlsx")
        save_worker_detection_excel(results["worker_detection"], worker_detect_path)
    
    # Save figures as SVGs.
    if "Trajectory plots" in results:
        figures = results["Trajectory plots"]
        for idx, fig in enumerate(figures):
            view_names = ["Isometric view", "Plan view", "Side view"]
            if idx < len(view_names):
                fig_path = os.path.join(new_folder_path, f"trajectory_{view_names[idx]}.svg")
                fig.savefig(fig_path, format="svg")
    
    if "velocity graph" in results:
        vel_graph_path = os.path.join(new_folder_path, "velocity_graph.svg")
        results["velocity graph"].savefig(vel_graph_path, format="svg")
    
    if "acceleration graph" in results:
        acc_graph_path = os.path.join(new_folder_path, "acceleration_graph.svg")
        results["acceleration graph"].savefig(acc_graph_path, format="svg")

    if "worker plot" in results:
        figures = results["worker plot"]
        for idx, fig in enumerate(figures):
            view_names = ["Isometric view", "Plan view"]
            if idx < len(view_names):
                fig_path = os.path.join(new_folder_path, f"worker_{view_names[idx]}.svg")
                fig.savefig(fig_path, format="svg")
    
    # Save the raw data as a pickle file.
    raw_data_path = os.path.join(new_folder_path, "raw_data.pkl")
    with open(raw_data_path, "wb") as file:
        pickle.dump(data, file)
    
    print(f"Results saved in {new_folder_path}")

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