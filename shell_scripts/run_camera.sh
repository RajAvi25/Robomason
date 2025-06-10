#!/usr/bin/env bash
# ------------------------------------------------------------
# Spawns two GNOME-Terminal windows, minimises them, then runs:
#   1) ros2 run my_robot_controller init_topic
#   2) conda run -n learningFactory python -m camera.live_camera
# Each window closes automatically when its command finishes.
# ------------------------------------------------------------

set -euo pipefail

TERMINAL=gnome-terminal
XDO_TOOL=xdotool
ENV_NAME="learningFactory"

CLASS1="gterm-init_topic"   # unique WM_CLASS values
CLASS2="gterm-live_camera"
DELAY=0.5          # seconds each terminal sleeps before exec’ing
LAUNCH_PAUSE=0.3   # time for windows to map before minimising

echo "Spawning init_topic window…"
${TERMINAL} --class "${CLASS1}" -- bash -c \
  "sleep ${DELAY}; exec ros2 run my_robot_controller init_topic" &

echo "Spawning live_camera window…"
${TERMINAL} --class "${CLASS2}" -- bash -c \
  "sleep ${DELAY}; exec conda run -n ${ENV_NAME} python -m camera.live_camera" &

sleep "${LAUNCH_PAUSE}"

echo "Minimising the spawned windows…"
for CLASS in "${CLASS1}" "${CLASS2}"; do
  for WID in $(${XDO_TOOL} search --onlyvisible --class "${CLASS}"); do
    ${XDO_TOOL} windowminimize "$WID"
  done
done

echo "Done. Commands begin after ${DELAY}s in minimised windows."
