#!/usr/bin/env bash
# kill_all.sh
# ------------------------------------------------------------
# Kills any running instances of:
#   • ros2 run my_robot_controller init_topic
#   • conda run -n learningFactory python -m camera.live_camera
#   • conda run -n learningFactory python -m tracker.tracker
#   • conda run -n learningFactory python -m robot_controller.robo_controller
#   • conda run -n learningFactory python -m plotting.main
# and frees ports 7000 & 9090.
# ------------------------------------------------------------

set -euo pipefail

# List of process patterns to kill
patterns=(
  "ros2 run my_robot_controller init_topic"
  "python -m camera.live_camera"
  "python -m tracker.tracker"
  "python -m robot_controller.robo_controller"
  "python -m plotting.main"
)

echo "Killing matching processes…"
for pat in "${patterns[@]}"; do
  echo "  • pattern: ${pat}"
  pkill -f "${pat}" 2>/dev/null || true
done

# Free network ports (adjust if yours differ)
for port in 7000 9090; do
  echo "Freeing port ${port}/tcp…"
  fuser -k "${port}/tcp" 2>/dev/null || true
done

echo "Done."
