#!/bin/bash
if [ $# -ne 1 ]; then
    echo "usage: bash jobs/lab_run.sh NAME jobs/lab_slo_runs.sh RUNS_FOLDER"
    exit 1
fi
cd "$(dirname "$0")/.."
RUNS=$1
FAILED=0
FLAT="--keep-slices 24 --z-down 0 0 1 1"
for RUN in ${SLO_RUNS:-xattn joint}; do
    case $RUN in
        xattn)
            echo "== memory, SLO cross-attention at levels 0 1 2, 256 tokens"
            python scripts/probe_memory.py --designs w2z1 --target 24 512 512 --z-down 0 0 1 1 --slo-size 256 --slo-tokens 16 --slo-xattn 0 1 2 --max-batch 8 || FAILED=$((FAILED + 1))
            echo "== w2z1_xattn: train"
            python scripts/train.py --out "$RUNS/w2z1_xattn" --design w2z1 --band-scale 0.5 --batch-size 4 --epochs ${EPOCHS:-200} --auto-resume $FLAT --target 24 512 512 --slo-size 256 --slo-tokens 16 --slo-xattn 0 1 2 || FAILED=$((FAILED + 1))
            ;;
        joint)
            echo "== memory, OCT and SLO in one volume"
            python scripts/probe_memory.py --designs w2z1 --target 32 512 512 --z-down 0 0 1 1 --max-batch 8 || FAILED=$((FAILED + 1))
            echo "== w2z1_joint: train"
            python scripts/train.py --out "$RUNS/w2z1_joint" --design w2z1 --band-scale 0.5 --batch-size 4 --epochs ${EPOCHS:-200} --auto-resume $FLAT --target 32 512 512 --slo-slice raster || FAILED=$((FAILED + 1))
            ;;
        *)
            echo "unknown run $RUN"
            FAILED=$((FAILED + 1))
            ;;
    esac
done
echo "SLO runs finished, failed steps: $FAILED"
