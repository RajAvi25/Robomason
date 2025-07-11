import json
import subprocess
import numpy as np
from configs.construction_config import python_executable_IFC ,IFC_function_path

def IFC_loaded_sorted(*script_args):
    """
    Load and sort IFC building components by height using an external IFC parser.

    This function invokes an external IFC parsing script (via a separate Python interpreter)
    to read a Building Information Model (IFC file) and extract construction elements.
    It returns a sorted list of elements (as a NumPy array) in ascending order of their 
    average Z-coordinate (height), effectively producing a global block sequence:contentReference[oaicite:0]{index=0}.

    **Parameters:**
    - *script_args: Any* – Command-line arguments to pass to the IFC parsing script. 
      Typically this includes the path to an IFC file and optional flags (e.g., debug).

    **Returns:**
    - *np.ndarray* – A NumPy array of shape (N, 4), where each row corresponds to an element.
      The row format is `[Name, X_mean, Y_mean, Z_mean]`. The array is sorted by Z_mean 
      (lowest first, e.g., foundation at index 0).

    **Behavior:**
    1. Constructs a shell command using the configured IFC parser path and Python interpreter 
       (see `configs.construction_config.py` for `python_executable_IFC` and `IFC_function_path`).
    2. Executes the command as a subprocess. The external script (see `IFC_functions.py`) will 
       parse the IFC file into element coordinates and print a JSON array of sorted elements.
    3. Captures the subprocess output (JSON string on the last line), parses it, and converts 
       to a NumPy array for return.

    **Assumptions & Dependencies:**
    - Relies on ifcopenshell and PythonOCC (OpenCASCADE) in the external environment for IFC parsing.
    - The external script prints the final sorted list as JSON on a single line.
    - Sorting by height assumes the assembly order is bottom-up (foundation -> walls -> floors, etc.), 
      aligning with the IFC-driven planning approach:contentReference[oaicite:1]{index=1}.

    **Role in System:**
    This function provides the *BIM/IFC-driven parsing* step of the pipeline:contentReference[oaicite:2]{index=2}. It translates 
    high-level building design data (IFC) into an ordered list of parts and their approximate positions, 
    which is the basis for subsequent robot pick-and-place operations.
    """
    command = f"{python_executable_IFC} {IFC_function_path}"

    if script_args:
        command += " " + " ".join(map(str, script_args))

    try:
        # Run the command and capture the output
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        
        # Get the JSON string from the output (assuming it's the last thing printed)
        array_json = result.stdout.strip().splitlines()[-1]  # Take the last line
        
        # Convert the JSON string back to a NumPy array
        return_value = np.array(json.loads(array_json))
        
        return return_value
        
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the script: {e}")
        print(f"Error output: {e.stderr}")
        return None
    except json.JSONDecodeError as je:
        print(f"An error occurred while parsing the JSON string: {je}")
        return None