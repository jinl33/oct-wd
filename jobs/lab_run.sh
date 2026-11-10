#!/bin/bash
set -e
if [ $# -lt 2 ]; then
    echo "usage: bash jobs/lab_run.sh NAME scripts/SCRIPT.py|jobs/SCRIPT.sh [arguments]"
    exit 1
fi
NAME=$1
shift
case "$1" in
    *.sh) RUNNER=bash ;;
    *) RUNNER=python ;;
esac
cd "$(dirname "$0")/.."
source ~/oct-env/bin/activate
source ~/.oct_paths
mkdir -p ~/oct-logs
LOG=~/oct-logs/$NAME-$(date +%Y%m%d-%H%M%S).log
nvidia-smi --query-gpu=name,memory.used,memory.total,driver_version --format=csv > "$LOG"
nohup $RUNNER "$@" >> "$LOG" 2>&1 < /dev/null &
echo "started $NAME as process $!; follow it with: tail -f $LOG"
