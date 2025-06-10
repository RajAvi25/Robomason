#!/usr/bin/env bash
# ------------------------------------------------------------
# Spawns one GNOME-Terminal window, minimises it, then runs:
#   • conda run -n learningFactory python -m tracker.tracker
# The window closes automatically when the command finishes.
# ------------------------------------------------------------

set -euo pipefail

TERMINAL=gnome-terminal
XDO_TOOL=xdotool
ENV_NAME="learningFactory"

CLASS="gterm-tracker"   # unique WM_CLASS for this window
DELAY=0.5               # sleep before exec’ing
LAUNCH_PAUSE=0.3        # time for the window to map

command -v "${XDO_TOOL}" >/dev/null || {
  echo "Install “${XDO_TOOL}” first: sudo apt-get install -y xdotool" >&2
  exit 1
}

echo "🚀  Spawning tracker window…"
${TERMINAL} --class "${CLASS}" -- bash -c \
  "sleep ${DELAY}; exec conda run -n ${ENV_NAME} python -m tracker.tracker" &

sleep "${LAUNCH_PAUSE}"

echo "🖥️  Minimising tracker window…"
for WID in $(${XDO_TOOL} search --onlyvisible --class "${CLASS}"); do
  ${XDO_TOOL} windowminimize "$WID"
done

echo "✅  Done. tracker.tracker starts after ${DELAY}s in a minimised window."
