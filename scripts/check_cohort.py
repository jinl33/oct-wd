#!/usr/bin/env python3
import argparse
import collections
import csv
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oct_wd import cohort as co

SETTINGS = ["sloHorizontalFieldOfView", "sloPixelspacingX", "sloPixelspacingY", "sliceThickness",
            "bscanPixelspacingWidth", "bscanPixelspacingDepth", "BScanWidthOrRadius"]
TALLIES = [("hasOCT",), ("hasSLO",), ("righteye",), ("slox", "sloy"), ("bscanx", "bscany")]


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def rng(vals):
    if not vals:
        return "no values"
    return "min %.6g  median %.6g  max %.6g  (n=%d)" % (min(vals), statistics.median(vals), max(vals), len(vals))


def tally(title, counter):
    print(title)
    if len(counter) > 10 or any(len(str(k)) > 24 for k in counter):
        print("  (%d different values, not shown)" % len(counter))
        return
    for k, v in sorted(counter.items(), key=lambda kv: -kv[1]):
        print("  %-26s %d" % (k, v))


def direction(seq):
    d = [b - a for a, b in zip(seq, seq[1:])]
    if all(x > 0 for x in d):
        return "increasing"
    if all(x < 0 for x in d):
        return "decreasing"
    if all(x == 0 for x in d):
        return "constant"
    return "mixed"


def geometry(meta, slo=768):
    start_r = [float(m["pixel_start_row"]) for m in meta]
    start_c = [float(m["pixel_start_col"]) for m in meta]
    end_r = [float(m["pixel_end_row"]) for m in meta]
    end_c = [float(m["pixel_end_col"]) for m in meta]
    index = [int(float(m["bscan_index"])) for m in meta]
    steps = [abs(b - a) for a, b in zip(start_r, start_r[1:])]
    return {
        "index_ok": index == list(range(1, len(index) + 1)),
        "row_order": direction(start_r),
        "col_order": direction(start_c),
        "left_to_right": all(a < b for a, b in zip(start_c, end_c)),
        "in_bounds": all(0 <= v <= slo - 1 for v in start_r + start_c + end_r + end_c),
        "angle": statistics.median(math.degrees(math.atan2(er - sr, ec - sc)) for sr, sc, er, ec in zip(start_r, start_c, end_r, end_c)),
        "length": statistics.median(math.hypot(er - sr, ec - sc) for sr, sc, er, ec in zip(start_r, start_c, end_r, end_c)),
        "step": statistics.median(steps) if steps else 0.0,
        "uneven": bool(steps) and max(steps) - min(steps) > 2,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    root = ap.parse_args().root or co.default_root()
    images_dir, cohort_csv = co.paths(root)
    rows = co.read_rows(cohort_csv)
    names = co.folder_names(images_dir)
    print("cohort rows: %d   image folders: %d   distinct anon_id: %d" % (len(rows), len(names), len({r["anon_id"] for r in rows})))
    print("cohort columns that name a folder (rows matched):")
    matches = {c: n for c, n in co.column_matches(rows, names).items() if n}
    for c, n in matches.items():
        print("  %-26s %d" % (c, n))
    column = co.find_folder_column(rows, names)
    print("folders with no cohort row:", len(names - {co.last_part(r[column] or "") for r in rows}))

    groups = collections.defaultdict(list)
    no_folder = wrong_count = 0
    eye_order = collections.Counter()
    for r in rows:
        name = co.last_part(r[column] or "")
        if name not in names:
            no_folder += 1
            continue
        folder = os.path.join(images_dir, name)
        n = len(co.bscan_numbers(folder))
        wrong_count += num(r.get("nbscans")) != n
        groups[n].append((r, folder))
    print("cohort rows with no folder:", no_folder)
    print("rows where nbscans differs from the B-scan files:", wrong_count)

    for cols in TALLIES:
        tally(" / ".join(cols), collections.Counter(" / ".join(str(r.get(c, "")).strip() for c in cols) for r in rows))

    for n in sorted(groups):
        items = groups[n]
        print("\n=== scans with %d B-scans: %d ===" % (n, len(items)))
        for s in SETTINGS:
            vals = [v for v in (num(r.get(s)) for r, _ in items) if v is not None]
            print("  %-26s %s" % (s, rng(vals)))
        geo, bad = [], 0
        rows_vs_scans = 0
        for r, folder in items:
            try:
                with open(os.path.join(folder, "metadata.csv"), newline="", encoding="utf-8-sig", errors="replace") as f:
                    meta = list(csv.DictReader(f))
                g = geometry(meta)
            except (OSError, KeyError, ValueError, csv.Error, statistics.StatisticsError):
                bad += 1
                continue
            rows_vs_scans += len(meta) != n
            g["eye"] = str(r.get("righteye", "")).strip()[:8]
            geo.append(g)
        print("  metadata unreadable: %d   row count differs from B-scans: %d" % (bad, rows_vs_scans))
        if not geo:
            continue
        print("  bscan_index runs 1..N in order: %d of %d" % (sum(g["index_ok"] for g in geo), len(geo)))
        print("  all coordinates inside a 768 x 768 SLO: %d of %d" % (sum(g["in_bounds"] for g in geo), len(geo)))
        print("  lines drawn left to right: %d of %d" % (sum(g["left_to_right"] for g in geo), len(geo)))
        print("  uneven slice spacing (over 2 px): %d of %d" % (sum(g["uneven"] for g in geo), len(geo)))
        tally("  start row vs slice number", collections.Counter(g["row_order"] for g in geo))
        tally("  start column vs slice number", collections.Counter(g["col_order"] for g in geo))
        tally("  start row order by eye (righteye value)", collections.Counter("%s, eye %s" % (g["row_order"], g["eye"]) for g in geo))
        print("  line angle (degrees)        ", rng([g["angle"] for g in geo]))
        print("  line length (SLO pixels)    ", rng([g["length"] for g in geo]))
        print("  slice step (SLO pixels)     ", rng([g["step"] for g in geo]))


try:
    main()
except SystemExit:
    raise
except Exception as e:
    print("stopped early:", type(e).__name__)  # the message itself could contain a path
    sys.exit(1)
