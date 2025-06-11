# detections/accuracy.py

"""
Module: accuracy
This module provides functions for plotting data, generating summary tables,
transforming datasets, and performing Excel-based data analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from configs.system_config import *
from configs.construction_config import *


def CEP50_plot(all_data, actual_position, aroundTrue=False, issmall=False, test_number=None, isShow=True):
    """
    Plots detected points and their 50th percentile circle around either the centroid
    or the actual position for each valid ID from the provided data.

    Parameters:
        all_data (list): List of dictionaries containing an 'ID' and a 'coordinates' key.
        actual_position (dict): Dictionary where key is worker ID and value is (x, y) position.
        aroundTrue (bool): Flag to determine if the 50th percentile circle is drawn around the 
                           actual position (if True) or around the centroid (if False).
        issmall (bool): If True, the plot limits will be adjusted to show a smaller region.
        test_number (optional): For labeling the plot title when provided.
        isShow (bool): If True, the plot is shown; otherwise, it is closed after creation.

    Returns:
        matplotlib.figure.Figure: The generated plot figure.
    """
   
    # Separate coordinates by ID
    id_data = {}
    for item in all_data:
        if item['worker spotted']:
            worker_id = item['worker id']
            if worker_id in WORKER_COLORS.keys():
                if worker_id not in id_data:
                    id_data[worker_id] = {'x': [], 'y': [], 'z': []}
                id_data[worker_id]['x'].append(item['coordinates'][0] * 100)
                id_data[worker_id]['y'].append(item['coordinates'][1] * 100)
                id_data[worker_id]['z'].append(item['coordinates'][2] * 100)

    # Calculate overall min and max for x and y
    overall_min_x, overall_max_x = float('inf'), float('-inf')
    overall_min_y, overall_max_y = float('inf'), float('-inf')
    for data in id_data.values():
        overall_min_x = min(overall_min_x, min(data['x']))
        overall_max_x = max(overall_max_x, max(data['x']))
        overall_min_y = min(overall_min_y, min(data['y']))
        overall_max_y = max(overall_max_y, max(data['y']))

    # Plot setup
    plt.figure(figsize=(20, 20))
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    # Process each ID group
    for worker_id, data in id_data.items():
        color = WORKER_COLORS[worker_id]
        x_values = data['x']
        y_values = data['y']

        # Calculate average
        average_x = np.mean(x_values)
        average_y = np.mean(y_values)

        plt.scatter(x_values, y_values, color=color, label=f'ID {worker_id} Points', s=5)

        # Plot average point
        plt.scatter(average_x, average_y, color=color, marker='D', s=50, 
                    label=f'ID {worker_id} Avg ({round(average_x, 2)}, {round(average_y, 2)})')

        # Plot actual position
        actual_x, actual_y = actual_position[worker_id]
        plt.scatter(actual_x, actual_y, color=color, marker='s', s=100, 
                    label=f'ID {worker_id} Actual Pos ({actual_x}, {actual_y})')

        # Calculate 50th percentile circle
        if not aroundTrue:
            distances = [np.sqrt((x - average_x)**2 + (y - average_y)**2) for x, y in zip(x_values, y_values)]
            percentile_50_distance = np.percentile(distances, 50)
            circle = plt.Circle((average_x, average_y), percentile_50_distance, color=color,
                                fill=False, linestyle='--', linewidth=1, 
                                label=f'ID {worker_id} 50th Percentile')
        else:
            distances = [np.sqrt((x - actual_x)**2 + (y - actual_y)**2) for x, y in zip(x_values, y_values)]
            percentile_50_distance = np.percentile(distances, 50)
            circle = plt.Circle((actual_x, actual_y), percentile_50_distance, color=color,
                                fill=False, linestyle='--', linewidth=1, 
                                label=f'ID {worker_id} 50th Percentile')
        ax.add_patch(circle)

    # Adjust plot limits for issmall
    if issmall:
        margin = (overall_max_x - overall_min_x) * 0.35  # Adjust margin as needed
        ax.set_xlim(overall_min_x - margin, overall_max_x + margin)
        ax.set_ylim(overall_min_y - margin, overall_max_y + margin)

    # Set labels, reference lines, legend, and grid
    plt.xlabel("x [cm]", fontsize=12)
    plt.ylabel("y [cm]", fontsize=12)

    plt.axhline(0, color='gray', linewidth=0.8, linestyle='--')
    plt.axvline(0, color='gray', linewidth=0.8, linestyle='--')
    plt.legend(fontsize=10, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)

    # Display or close figure
    fig = plt.gcf()
    if isShow:
        plt.show()
    else:
        plt.close()

    return fig


def generate_summary_table(all_data, actual_position):
    """
    Generates a summary table (as a pandas DataFrame) containing statistics for each worker ID.

    Columns include:
        - Aruco ID: Unique worker IDs
        - True Position X, True Position Y: Actual positions from provided dictionary
        - Centroid X, Centroid Y: Average X, Y of each worker's points
        - Radius True: Median error circle with respect to true position
        - Radius Centroid: Median error circle with respect to centroid
        - Min Distance X, Min Distance Y: Closest measured point to the centroid
        - Max Distance X, Max Distance Y: Furthest measured point to the centroid

    Parameters:
        all_data (list): List of dictionaries containing an 'ID' and 'coordinates'.
        actual_position (dict): Dictionary where key is worker ID and value is a tuple (x, y).

    Returns:
        pandas.DataFrame: DataFrame summarizing the computed statistics.
    """

    id_data = {}
    for item in all_data:
        if item['worker spotted']:
            worker_id = item['worker id']
            if worker_id not in id_data:
                id_data[worker_id] = {'x': [], 'y': []}
            id_data[worker_id]['x'].append(item['coordinates'][0] * 100)
            id_data[worker_id]['y'].append(item['coordinates'][1] * 100)

    result_rows = []
    for worker_id, data in id_data.items():
        x_coords = np.array(data['x'])
        y_coords = np.array(data['y'])

        # Centroid (average position)
        centroid_x = np.mean(x_coords)
        centroid_y = np.mean(y_coords)

        # Distances to the centroid
        distances_centroid = np.sqrt((x_coords - centroid_x)**2 + (y_coords - centroid_y)**2)
        radius_centroid = round(np.median(distances_centroid), 2)

        # True Position and distances from true position
        true_x, true_y = actual_position[worker_id]
        distances_true = np.sqrt((x_coords - true_x)**2 + (y_coords - true_y)**2)
        radius_true = round(np.median(distances_true), 2)

        # Identify minimum and maximum distance points
        min_index = np.argmin(distances_true)
        max_index = np.argmax(distances_true)

        result_rows.append({
            'Aruco ID': worker_id,
            'True Position X': true_x,
            'True Position Y': true_y,
            'Radius True': radius_true,
            'Centroid X': round(centroid_x, 2),
            'Centroid Y': round(centroid_y, 2),
            'Radius Centroid': radius_centroid,
            'Min Distance X': round(x_coords[min_index], 2),
            'Min Distance Y': round(y_coords[min_index], 2),
            'Max Distance X': round(x_coords[max_index], 2),
            'Max Distance Y': round(y_coords[max_index], 2)
        })

    result_df = pd.DataFrame(result_rows)
    result_df = result_df.sort_values(by='Aruco ID').reset_index(drop=True)
    return result_df


# Optionally, you can add an __all__ list to specify the public API of the module
__all__ = [
    "CEP50_plot", "generate_summary_table", 
 
]

