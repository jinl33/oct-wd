#!/bin/bash
if [ $# -lt 1 ]; then
    echo "usage: bash jobs/lab_run.sh NAME jobs/lab_evaluate.sh RUN_FOLDER [RUN_FOLDER ...]"
    exit 1
fi
cd "$(dirname "$0")/.."
FAILED=0
for RUN in "$@"; do
    echo "== $(basename "$RUN"): sample"
    python scripts/sample.py --run "$RUN" $SAMPLE_ARGS || FAILED=$((FAILED + 1))
    for TASK in ${TASKS:-male righteye}; do
        for MODE in ${MODES:-probe finetune}; do
            for INIT in "" --random-init; do
                echo "== $(basename "$RUN"): $TASK $MODE ${INIT:-pretrained}"
                python scripts/downstream.py --run "$RUN" --task "$TASK" --mode "$MODE" $INIT $DOWN_ARGS || FAILED=$((FAILED + 1))
            done
        done
    done
done
echo "evaluation finished, failed steps: $FAILED"
