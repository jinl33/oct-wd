#!/bin/bash
if [ $# -lt 1 ]; then
    echo "usage: bash jobs/lab_run.sh NAME jobs/lab_label_fraction.sh RUN_FOLDER [RUN_FOLDER ...]"
    exit 1
fi
cd "$(dirname "$0")/.."
FAILED=0
for RUN in "$@"; do
    for TASK in ${TASKS:-righteye}; do
        for MODE in ${MODES:-probe}; do
            for FRACTION in ${FRACTIONS:-0.1 0.25 0.5 1}; do
                case "$FRACTION" in
                    1|1.0) SUBSETS=0 ;;
                    *) SUBSETS=${SEEDS:-0 1 2 3 4} ;;
                esac
                for SEED in $SUBSETS; do
                    for INIT in "" --random-init; do
                        echo "== $(basename "$RUN"): $TASK $MODE ${INIT:-pretrained} fraction $FRACTION subset $SEED"
                        python scripts/downstream.py --run "$RUN" --task "$TASK" --mode "$MODE" --label-fraction "$FRACTION" --subset-seed "$SEED" $INIT $DOWN_ARGS || FAILED=$((FAILED + 1))
                    done
                done
            done
        done
    done
    python scripts/summarize_downstream.py --run "$RUN"
done
echo "label-fraction runs finished, failed steps: $FAILED"
