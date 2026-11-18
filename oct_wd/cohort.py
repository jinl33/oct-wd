import csv
import math
import os
import random
import re

import numpy as np


IDENTIFIERS = ("anon_id", "datadir", "seriesdescription")
NOT_GROUPS = re.compile(r"date|time|mrn|path|dir|name|uid|(^|_)id$", re.I)
DATE_VALUE = re.compile(r"\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4}")


def default_root():
    root = os.environ.get("OCT_ROOT")
    if not root:
        raise SystemExit("OCT_ROOT is not set. See 'First-time setup' in README.md.")
    return root


def paths(root):
    return os.path.join(root, "oct_images"), os.path.join(root, "pilot_cohort_deidentified.csv")


def read_rows(path):
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
        return list(csv.DictReader(f))


def last_part(value):
    return value.strip().replace("\\", "/").rstrip("/").split("/")[-1]


def folder_names(images_dir):
    return {e.name for e in os.scandir(images_dir) if e.is_dir(follow_symlinks=False)}


def column_matches(rows, names):
    if not rows:
        return {}
    return {c: sum(last_part(r[c] or "") in names for r in rows) for c in rows[0] if c}


def find_folder_column(rows, names, min_share=0.95):
    matches = column_matches(rows, names)
    best = max(matches, key=matches.get) if matches else None
    if best is None or matches[best] < min_share * len(rows):
        raise SystemExit("no cohort column names the scan folders; best match count: %d of %d" % (matches.get(best, 0), len(rows)))
    return best


def bscan_numbers(folder):
    nums = []
    for n in os.listdir(folder):
        m = re.fullmatch(r"b(\d+)\.png", n, re.I)
        if m:
            nums.append(int(m.group(1)))
    return sorted(nums)


def list_scans(root, n_slices=25):
    images_dir, cohort_csv = paths(root)
    rows = read_rows(cohort_csv)
    names = folder_names(images_dir)
    column = find_folder_column(rows, names)
    scans = []
    counts = {"cohort_rows": len(rows), "folders": len(names), "no_folder": 0, "other_slice_count": 0, "bad_numbering": 0}
    for r in rows:
        name = last_part(r[column] or "")
        if name not in names:
            counts["no_folder"] += 1
            continue
        folder = os.path.join(images_dir, name)
        nums = bscan_numbers(folder)
        if len(nums) != n_slices:
            counts["other_slice_count"] += 1
        elif nums != list(range(1, n_slices + 1)):
            counts["bad_numbering"] += 1
        else:
            scans.append({"patient": r["anon_id"].strip(), "folder": folder, "n": n_slices, "eye": (r.get("righteye") or "").strip()[:8], "row": r})
    counts["kept"] = len(scans)
    return scans, counts


def split_by_patient(scans, val_fraction=0.1, test_fraction=0.2, seed=42):
    patients = sorted({s["patient"] for s in scans})
    random.Random(seed).shuffle(patients)
    n_test = round(len(patients) * test_fraction)
    n_val = round(len(patients) * val_fraction)
    test = set(patients[:n_test])
    val = set(patients[n_test: n_test + n_val])
    train = [s for s in scans if s["patient"] not in test and s["patient"] not in val]
    return train, [s for s in scans if s["patient"] in val], [s for s in scans if s["patient"] in test]


def label(scan, column):
    raw = (scan["row"].get(column) or "").strip().lower()
    if raw in ("true", "yes"):
        return 1.0
    if raw in ("false", "no"):
        return 0.0
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def group_labels(scans, column, max_groups=12, width=16):
    if column.strip().lower() in IDENTIFIERS or NOT_GROUPS.search(column):
        raise SystemExit("%s may identify scans or dates, so it cannot be a group" % column)
    if not scans or column not in scans[0]["row"]:
        raise SystemExit("the cohort file has no column %s" % column)
    raw = [(s["row"].get(column) or "").strip() for s in scans]
    present = sorted({v for v in raw if v})
    if not present:
        raise SystemExit("column %s is empty" % column)
    if any(DATE_VALUE.search(v) for v in present):
        raise SystemExit("column %s holds dates, so it cannot be a group" % column)
    if len(present) <= max_groups and len(present) <= max(2, len(raw) // 2):
        return [v[:width] or None for v in raw], sorted({v[:width] for v in present})
    values = [label(s, column) if v else None for s, v in zip(scans, raw)]
    if any(v and x is None for v, x in zip(raw, values)):
        raise SystemExit("column %s has %d different values and not all are numbers, so it cannot be a group" % (column, len(present)))
    edges = np.unique(np.quantile([x for x in values if x is not None], [0, 0.25, 0.5, 0.75, 1]))
    names = ["%.3g to %.3g" % (lo, hi) for lo, hi in zip(edges[:-1], edges[1:])]
    out = [None if x is None else names[min(max(int(np.searchsorted(edges, x, side="right")) - 1, 0), len(names) - 1)] for x in values]
    return out, names
