#!/usr/bin/env bash
# run_plotting.sh
# ------------------------------------------------------------
# Kills any existing plotting.main, then spawns a new terminal
# window (titled “plotting_main”) that:
#   • waits a beat so it can be minimized,
#   • exec’s plotting.main (so when it exits the window closes),
#   • runs inside your “learningFactory” env.
# ------------------------------------------------------------

set -euo pipefail

TERMINAL=gnome-terminal
XDO_TOOL=xdotool
ENV_NAME="learningFactory"
PY_MODULE="plotting.main"
PROCESS_PATTERN="python -m ${PY_MODULE}"
PORT=7000
TITLE="plotting_main"
DELAY=0.1        # seconds the child will sleep before exec’ing your module
LAUNCH_PAUSE=0.1 # seconds to wait before minimizing

echo "Stopping previous copies of ${PY_MODULE}…"
pkill -f "${PROCESS_PATTERN}" 2>/dev/null || true

echo "Freeing port ${PORT}/tcp…"
if command -v fuser >/dev/null; then
  fuser -k "${PORT}/tcp" 2>/dev/null || true
fi

# Spawn the terminal window (it sleeps, then exec’s plotting.main)
echo "Spawning plotting window (titled '${TITLE}')…"
${TERMINAL} --title "${TITLE}" -- bash -c \
  "sleep ${DELAY}; exec conda run -n ${ENV_NAME} python -m ${PY_MODULE}" &

# Give it a moment to appear
sleep "${LAUNCH_PAUSE}"

# Minimize that window by title
echo "Minimizing the plotting window…"
for WID in $(${XDO_TOOL} search --onlyvisible --name "${TITLE}"); do
  ${XDO_TOOL} windowminimize "${WID}"
done

echo "Done. plotting.main will start after ${DELAY}s in a minimized window."
