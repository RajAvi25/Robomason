#!/usr/bin/env bash
# run_robot.sh
# ------------------------------------------------------------
# Spawns one GNOME-Terminal window, minimises it, then runs:
#   • conda run -n learningFactory python -m robot_controller.robo_controller
# The window closes automatically when the command exits.
# ------------------------------------------------------------

set -euo pipefail

TERMINAL=gnome-terminal
XDO_TOOL=xdotool
ENV_NAME="learningFactory"

CLASS="gterm-robo_controller"   # unique WM_CLASS seen instantly by the WM
DELAY=0.5                       # seconds the child sleeps before exec’ing
LAUNCH_PAUSE=0.3                # time for the window to map before minimising

echo "Spawning robo_controller window…"
${TERMINAL} \
  --class "${CLASS}" \
  -- bash -c "sleep ${DELAY}; exec conda run -n ${ENV_NAME} python -m robot_controller.robo_controller" &

# Give the window a moment to appear
sleep "${LAUNCH_PAUSE}"

echo "Minimising the robo_controller window…"
for WID in $(${XDO_TOOL} search --onlyvisible --class "${CLASS}"); do
  ${XDO_TOOL} windowminimize "${WID}"
done

echo "Done. robo_controller starts after ${DELAY}s in a minimised window."
