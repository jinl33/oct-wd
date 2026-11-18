#!/bin/bash
if [ $# -ne 1 ]; then
    echo "usage: bash jobs/lab_run.sh NAME jobs/lab_next_runs.sh RUNS_FOLDER"
    exit 1
fi
cd "$(dirname "$0")/.."
RUNS=$1
FAILED=0
echo "== memory, SLO adapter"
python scripts/probe_memory.py --designs w2z1 --slo-size 256 --slo-adapter --max-batch 8 || FAILED=$((FAILED + 1))
echo "== w2z1_adapter: train"
python scripts/train.py --out "$RUNS/w2z1_adapter" --design w2z1 --slo-size 256 --slo-adapter --band-scale 0.5 --batch-size 4 --epochs ${EPOCHS:-200} --auto-resume || FAILED=$((FAILED + 1))
echo "== memory, 24 slices"
python scripts/probe_memory.py --designs w2z1 --target 24 512 512 --z-down 0 0 1 1 --max-batch 8 || FAILED=$((FAILED + 1))
echo "== w2z1_sqrt24: train"
python scripts/train.py --out "$RUNS/w2z1_sqrt24" --design w2z1 --band-scale 0.5 --batch-size 4 --epochs ${EPOCHS:-200} --auto-resume --keep-slices 24 --target 24 512 512 --z-down 0 0 1 1 || FAILED=$((FAILED + 1))
echo "next runs finished, failed steps: $FAILED"
