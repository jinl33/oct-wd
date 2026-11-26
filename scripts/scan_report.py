import argparse
import csv
import glob
import json
import os
import re
import sys
import time

import numpy as np
import torch
from PIL import Image
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generative.networks.schedulers import DDIMScheduler

from oct_wd import cohort as co
from oct_wd import evaluate as ev
from oct_wd import wavelet as wv
from oct_wd.model import band_scale, joint_parts, load_checkpoint, options, predict, to_device
from oct_wd.volumes import as_batch, assemble, load_condition, oct_region

PARTS = ("train", "val", "test")
METRICS = (("psnr", "PSNR (dB)"), ("ssim", "SSIM"))


def to_pixels(x, band, axes, native, joint=False):
    return oct_region(wv.inverse(wv.unscaled(x.float(), band), axes)[0, 0].clamp(-1, 1), native, joint)


def to_uint8(x):
    return ((x.clamp(-1, 1) + 1) / 2 * 255).round().to(torch.uint8).cpu().numpy()


def finite(values):
    return [float(v) for v in values if np.isfinite(v)]


def spread(values, fmt):
    v = finite(values)
    if not v:
        return "no values"
    low, mid, high = np.percentile(v, [25, 50, 75])
    return (fmt + " (" + fmt + " to " + fmt + ")") % (mid, low, high)


def compare(groups):
    groups = [g for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return float("nan"), "too few scans for a test"
    try:
        if len(groups) == 2:
            return float(stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided").pvalue), "Mann-Whitney"
        return float(stats.kruskal(*groups).pvalue), "Kruskal-Wallis"
    except ValueError:
        return float("nan"), "no test, values identical"


def save_view(path, panels):
    canvas = Image.new("L", (sum(p.shape[1] for p in panels) + 10 * (len(panels) - 1), max(p.shape[0] for p in panels)))
    x = 0
    for p in panels:
        canvas.paste(Image.fromarray(p), (x, 0))
        x += p.shape[1] + 10
    canvas.save(path)


def safe_name(column):
    return re.sub(r"[^A-Za-z0-9]+", "_", column).strip("_") or "group"


def finish(fig, plt, out, name, top=0.95):
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(os.path.join(out, name), dpi=150)
    plt.close(fig)
    return name


def draw(out, rows, scans, labels, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError:
        print("matplotlib is not installed, so no plots were drawn; install it with: pip install matplotlib")
        return []
    written = []
    steps = args.steps
    if args.group:
        fig, axs = plt.subplots(1, len(args.group), figsize=(4.5 * len(args.group), 3.8), squeeze=False)
        for ax, c in zip(axs[0], args.group):
            values, names = labels[c]
            cats = names + (["missing"] if None in values else [])
            x = np.arange(len(cats))
            width = 0.8 / len(PARTS)
            for j, part in enumerate(PARTS):
                counts = [sum(1 for s, v in zip(scans, values) if (v or "missing") == n and s["part"] == part) for n in cats]
                ax.bar_label(ax.bar(x + (j - 1) * width, counts, width, label=part), fontsize=7)
            tilt = len(cats) > 4
            ax.set_xticks(x)
            ax.set_xticklabels(cats, rotation=30 if tilt else 0, ha="right" if tilt else "center")
            ax.set_title(c)
            ax.set_ylabel("scans")
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.margins(y=0.12)
        fig.suptitle("Scans per value and split part (%d scans)" % len(scans))
        fig.legend(*axs[0][0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, 0.9), ncol=len(PARTS), frameon=False)
        written.append(finish(fig, plt, out, "cohort.png", top=0.84))

    fig, axs = plt.subplots(2, len(steps), figsize=(3.8 * len(steps), 6.4), squeeze=False)
    for j, t in enumerate(steps):
        for i, (m, name) in enumerate(METRICS):
            ax = axs[i][j]
            v = finite(r["%s_t%d" % (m, t)] for r in rows)
            if v:
                ax.hist(v, bins=min(30, max(5, len(v) // 5)), color="C0", alpha=0.85)
                ax.axvline(np.median(v), color="black", lw=1, label="median")
            if m == "psnr":
                noisy = finite(r["noisy_psnr_t%d" % t] for r in rows)
                if noisy:
                    ax.text(0.02, 0.97, "noisy input: median %.2f dB" % np.median(noisy), transform=ax.transAxes, va="top", fontsize=8)
                ax.set_title("noise step %d" % t)
            ax.set_xlabel(name)
            ax.set_ylabel("scans")
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.margins(y=0.15)
    axs[0][0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Reconstruction of %d %s scans from one noisy step" % (len(rows), " and ".join(args.parts)))
    written.append(finish(fig, plt, out, "reconstruction.png"))

    rng = np.random.default_rng(0)
    for c in args.group:
        cats = [n for n in labels[c][1] + ["missing"] if any(r[c] == n for r in rows)]
        fig, axs = plt.subplots(2, len(steps), figsize=(max(3.8, 1.1 * len(cats) + 1.5) * len(steps), 6.8), squeeze=False)
        for j, t in enumerate(steps):
            for i, (m, name) in enumerate(METRICS):
                ax = axs[i][j]
                data = [finite(r["%s_t%d" % (m, t)] for r in rows if r[c] == n) for n in cats]
                pos = np.arange(1, len(cats) + 1)
                ax.boxplot(data, positions=pos, widths=0.5, showfliers=False)
                for p, d in zip(pos, data):
                    ax.scatter(p + rng.uniform(-0.15, 0.15, len(d)), d, s=8, alpha=0.5, color="C0")
                ax.set_xticks(pos)
                ax.set_xticklabels(["%s\nn=%d" % (n, len(d)) for n, d in zip(cats, data)], fontsize=8)
                ax.set_ylabel(name)
                ax.yaxis.set_major_locator(MaxNLocator(5))
                p_value, _ = compare([d for n, d in zip(cats, data) if n != "missing"])
                test = "p = %.2g" % p_value if np.isfinite(p_value) else "no test"
                ax.set_title(("noise step %d\n" % t if i == 0 else "") + test, fontsize=9)
        fig.suptitle("Reconstruction by %s" % c)
        written.append(finish(fig, plt, out, "reconstruction_%s.png" % safe_name(c)))
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--parts", nargs="+", default=["val", "test"], choices=PARTS, help="split parts to score; training scans would look better than they are")
    ap.add_argument("--steps", type=int, nargs="+", default=[50, 200, 500], help="each scan is noised to these steps and restored in one model call")
    ap.add_argument("--group", nargs="*", default=["male", "righteye"], help="cohort columns that split the counts and plots")
    ap.add_argument("--n-worst", type=int, default=4, help="views of the scans with the lowest PSNR at the middle step")
    ap.add_argument("--max-scans", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if any(not 0 < t < 1000 for t in args.steps):
        raise SystemExit("noise steps must be between 1 and 999")
    args.steps = sorted(set(args.steps))
    args.group = list(dict.fromkeys(args.group))
    start = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    model, ckpt, axes = load_checkpoint(os.path.join(args.run, "latest.pt"), device)
    model.eval()
    with open(os.path.join(args.run, "config.json")) as f:
        config = json.load(f)
    with open(os.path.join(args.run, "split.json")) as f:
        split = json.load(f)
    empty = [p for p in args.parts if not split.get(p)]
    if empty:
        raise SystemExit("this run's split has no %s scans" % " or ".join(empty))
    target, slo_size, band, saved = tuple(ckpt["target"]), ckpt.get("slo_size", 0), band_scale(ckpt), options(ckpt)
    slo_size = 0 if saved["align"] else slo_size
    joint = joint_parts(ckpt, axes) is not None
    slo_text = ("%d px" % slo_size + (", adapter" if saved["adapter"] else "") + (", cross-attention at levels %s" % " ".join(map(str, saved["xattn"])) if saved["xattn"] else "")) if slo_size else ("joint slice (%s); PSNR and SSIM on the B-scans only" % saved["slo_slice"] if joint else "not used")
    print("checkpoint epoch %d, design %s, SLO %s, band scaling %s, device %s" % (ckpt["epoch"], ckpt["design"], slo_text, "on" if band is not None else "off", device))

    scans, _ = co.list_scans(args.root or co.default_root(), config["n_slices"])
    part_of = {pid: p for p in PARTS for pid in split.get(p, [])}
    for s in scans:
        s["part"] = part_of.get(s["patient"])
    labels = {c: co.group_labels(scans, c) for c in args.group}
    chosen = [i for i, s in enumerate(scans) if s["part"] in args.parts]
    if args.max_scans > 0:
        chosen = chosen[: args.max_scans]
    if not chosen:
        raise SystemExit("no scans found for %s" % " and ".join(args.parts))
    print("scans in the cohort: %d (train %d, val %d, test %d); scored here: %d from %s" % (
        len(scans), *[sum(s["part"] == p for s in scans) for p in PARTS], len(chosen), " and ".join(args.parts)))
    for c in args.group:
        values, names = labels[c]
        print("%s, scans per value (train / val / test):" % c)
        for n in names + (["missing"] if None in values else []):
            print("  %-18s %s" % (n, " / ".join(str(sum(1 for s, v in zip(scans, values) if (v or "missing") == n and s["part"] == p)) for p in PARTS)))

    out = args.out or os.path.join(args.run, "scan_report")
    os.makedirs(out, exist_ok=True)
    for old in glob.glob(os.path.join(out, "worst_*.png")):
        os.remove(old)
    ddim = DDIMScheduler(num_train_timesteps=1000, schedule="linear_beta", prediction_type="sample", clip_sample=True)
    view_step = args.steps[len(args.steps) // 2]
    rows, worst, failed = [], [], 0
    with torch.no_grad():
        for k, i in enumerate(chosen):
            s = scans[i]
            try:
                real, full = assemble(s["folder"], s["n"], target, saved["keep"], saved["slo_slice"])
                slo = to_device(as_batch(load_condition(s["folder"], s["n"], slo_size, saved["adapter"], real.shape, target, saved["keep"])), device)
            except Exception as e:
                failed += 1
                print("scan %d of %d could not be read: %s" % (k + 1, len(chosen), type(e).__name__))
                continue
            native = real.shape
            ref = torch.from_numpy(real).to(device)
            x0 = wv.scaled(wv.forward(torch.from_numpy(full)[None, None].to(device), axes), band)
            noise = torch.randn(x0.shape, generator=torch.Generator().manual_seed(args.seed + i)).to(device)
            row = {"part": s["part"]}
            for c in args.group:
                row[c] = labels[c][0][i] or "missing"
            for t_value in args.steps:
                t = torch.full((1,), t_value, device=device, dtype=torch.long)
                noisy = ddim.add_noise(original_samples=x0, noise=noise, timesteps=t)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    pred = predict(model, noisy, t, slo)
                made = to_pixels(pred, band, axes, native, joint)
                seen = to_pixels(noisy, band, axes, native, joint)
                row["psnr_t%d" % t_value] = ev.psnr(made, ref)
                row["noisy_psnr_t%d" % t_value] = ev.psnr(seen, ref)
                row["ssim_t%d" % t_value] = ev.ssim(made, ref)
                if t_value == view_step and args.n_worst > 0:
                    mid = native[0] // 2
                    worst.append((row["psnr_t%d" % t_value], [to_uint8(seen[mid]), to_uint8(made[mid]), to_uint8(ref[mid])]))
                    worst.sort(key=lambda w: w[0])
                    del worst[args.n_worst:]
            rows.append(row)
            if (k + 1) % 25 == 0:
                print("scored %d of %d" % (k + 1, len(chosen)), flush=True)
    if not rows:
        raise SystemExit("no scan could be scored")
    print("scored %d scans, could not read %d, %.0f s" % (len(rows), failed, time.time() - start))

    fields = ["part"] + args.group + ["%s_t%d" % (m, t) for t in args.steps for m in ("psnr", "noisy_psnr", "ssim")]
    with open(os.path.join(out, "scans.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: "%.6g" % r[k] if isinstance(r[k], float) else r[k] for k in fields})

    print("noise is added in each run's own wavelet units, so compare groups within a run, not runs with each other")
    print("reconstruction from one noisy step; median (25th to 75th percentile):")
    print("  %-6s %-24s %-24s %s" % ("step", "PSNR, dB", "noisy input PSNR, dB", "SSIM"))
    for t in args.steps:
        print("  %-6d %-24s %-24s %s" % (t, spread([r["psnr_t%d" % t] for r in rows], "%.2f"), spread([r["noisy_psnr_t%d" % t] for r in rows], "%.2f"), spread([r["ssim_t%d" % t] for r in rows], "%.3f")))
    for c in args.group:
        cats = [n for n in labels[c][1] + ["missing"] if any(r[c] == n for r in rows)]
        print("by %s:" % c)
        print("  %-6s %-18s %-5s %-24s %s" % ("step", "value", "n", "PSNR, dB", "SSIM"))
        for t in args.steps:
            for n in cats:
                part = [r for r in rows if r[c] == n]
                print("  %-6d %-18s %-5d %-24s %s" % (t, n, len(part), spread([r["psnr_t%d" % t] for r in part], "%.2f"), spread([r["ssim_t%d" % t] for r in part], "%.3f")))
            tests = []
            for m, name in METRICS:
                p_value, how = compare([finite(r["%s_t%d" % (m, t)] for r in rows if r[c] == n) for n in cats if n != "missing"])
                tests.append("%s p %s" % (name.split()[0], "%.3g" % p_value if np.isfinite(p_value) else "-"))
            print("  %-6d %s (%s)" % (t, "   ".join(tests), how))
    if args.group:
        print("p-values are not corrected for the number of comparisons")

    for rank, (value, panels) in enumerate(worst):
        save_view(os.path.join(out, "worst_%03d.png" % rank), panels)
    if worst:
        print("lowest PSNR at step %d, in worst_000.png onwards (noisy input | reconstruction | original, middle B-scan): %s dB" % (view_step, ", ".join("%.2f" % v for v, _ in worst)))
    written = draw(out, rows, scans, labels, args)
    print("written to %s: scans.csv%s%s" % (os.path.basename(os.path.normpath(out)), "".join(", " + n for n in written), ", %d worst_*.png" % len(worst) if worst else ""))


try:
    main()
except SystemExit:
    raise
except Exception as e:
    print("stopped early:", type(e).__name__)
    sys.exit(1)
