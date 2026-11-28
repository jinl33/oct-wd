#!/bin/bash
if [ $# -ne 1 ]; then
    echo "usage: bash jobs/lab_run.sh NAME jobs/lab_jepa_sigreg.sh RUNS_FOLDER"
    exit 1
fi
cd "$(dirname "$0")/.."
RUNS=$1
FAILED=0
RUN=$RUNS/w2z1_align_jepa_sigreg
echo "== $(basename "$RUN"): train"
python scripts/train.py --out "$RUN" --design w2z1 --band-scale 0.5 --batch-size 4 --epochs ${EPOCHS:-200} --auto-resume --keep-slices 24 --z-down 0 0 1 1 --target 24 512 512 --slo-size 256 --slo-align 0.01 --slo-align-loss jepa --slo-sigreg 0.05 || FAILED=$((FAILED + 1))
echo "== split check against w2z1_sqrt24"
if python -c "import json,sys; s=[json.load(open(p)) for p in sys.argv[1:]]; sys.exit(0 if all(x == s[0] for x in s) else 1)" "$RUNS/w2z1_sqrt24/split.json" "$RUN/split.json"; then
    echo "same split as w2z1_sqrt24"
    echo "== eye probe at label fractions"
    bash jobs/lab_label_fraction.sh "$RUN" || FAILED=$((FAILED + 1))
    echo "== retrieval"
    python scripts/retrieval.py --run "$RUN" || FAILED=$((FAILED + 1))
else
    echo "could not confirm the split matches w2z1_sqrt24's; heads and retrieval not run"
    FAILED=$((FAILED + 1))
fi
echo "SIGReg run finished, failed steps: $FAILED"
