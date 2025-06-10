#!/bin/bash

# Make sure the script exits if any command fails
set -e

# Run all scripts from the shell_scripts folder in the background
echo "Starting camera..."
shell_scripts/run_camera.sh &

shell_scripts/run_camera.sh

echo "Starting tracker..."
shell_scripts/run_tracker.sh &

echo "Starting robot..."
shell_scripts/run_robot.sh &

echo "Starting plotting..."
shell_scripts/run_plotting.sh &

# Wait for all background processes to complete
wait
echo "All processes finished."
