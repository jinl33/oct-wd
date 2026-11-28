#!/bin/bash
if [ $# -lt 3 ]; then
    echo "usage: bash jobs/lab_after.sh JOB NAME jobs/SCRIPT.sh|scripts/SCRIPT.py [arguments]"
    echo "waits until the running job JOB (for example lab_jepa_sigreg.sh) ends, then runs bash jobs/lab_run.sh NAME ..."
    exit 1
fi
WAIT=${1#jobs/}
WAIT=${WAIT#scripts/}
shift
cd "$(dirname "$0")/.."
PIDS=$(pgrep -f "^(bash|python[0-9.]*) (jobs|scripts)/${WAIT//./\\.}( |$)" | tr '\n' ' ')
if [ -z "$PIDS" ]; then
    echo "no running job matches $WAIT; nothing started"
    exit 1
fi
mkdir -p ~/oct-logs
LOG=~/oct-logs/after-$1-$(date +%Y%m%d-%H%M%S).log
nohup bash -c 'for p in $0; do while kill -0 "$p" 2>/dev/null; do sleep "${EVERY:-60}"; done; done; echo "previous job ended $(date)"; bash jobs/lab_run.sh "$@"' "$PIDS" "$@" > "$LOG" 2>&1 < /dev/null &
echo "waiting for $WAIT (process $PIDS) to end, then starting $1; follow it with: tail -f $LOG"
