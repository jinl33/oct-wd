#!/bin/bash
if [ $# -ne 1 ]; then
    echo "usage: bash jobs/lab_run.sh NAME jobs/lab_seed_runs.sh RUNS_FOLDER"
    exit 1
fi
cd "$(dirname "$0")/.."
RUNS=$1
SEED=${SEED:-1}
FAILED=0
COMMON="--design w2z1 --band-scale 0.5 --batch-size 4 --epochs ${EPOCHS:-200} --auto-resume --keep-slices 24 --z-down 0 0 1 1 --target 24 512 512 --seed $SEED --split-seed 42"
ALIGN=$RUNS/w2z1_align_s$SEED
PLAIN=$RUNS/w2z1_sqrt24_s$SEED
echo "== $(basename "$ALIGN"): train"
python scripts/train.py --out "$ALIGN" $COMMON --slo-size 256 --slo-align 0.01 || FAILED=$((FAILED + 1))
echo "== $(basename "$PLAIN"): train"
python scripts/train.py --out "$PLAIN" $COMMON || FAILED=$((FAILED + 1))
echo "== split check against w2z1_sqrt24"
if python -c "import json,sys; s=[json.load(open(p)) for p in sys.argv[1:]]; sys.exit(0 if all(x == s[0] for x in s) else 1)" "$RUNS/w2z1_sqrt24/split.json" "$ALIGN/split.json" "$PLAIN/split.json"; then
    echo "same split as w2z1_sqrt24"
    echo "== eye probe at label fractions"
    bash jobs/lab_label_fraction.sh "$ALIGN" "$PLAIN" || FAILED=$((FAILED + 1))
    echo "== retrieval"
    python scripts/retrieval.py --run "$ALIGN" || FAILED=$((FAILED + 1))
else
    echo "could not confirm the split matches w2z1_sqrt24's; heads and retrieval not run"
    FAILED=$((FAILED + 1))
fi
echo "seed runs finished, failed steps: $FAILED"
