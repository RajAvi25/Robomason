import json
import subprocess
import numpy as np
from construction_config import python_executable_IFC ,IFC_function_path

def IFC_loaded_sorted(*script_args):
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