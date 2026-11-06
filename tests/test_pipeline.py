import csv
import glob
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from oct_wd import alignment as al
from oct_wd import cohort as co
from oct_wd import volumes as vo
from oct_wd import wavelet as wv

torch.set_num_threads(2)
torch.manual_seed(0)
ENV = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
ENV.pop("OCT_ROOT", None)
LEAK_WORDS = ("FAKEUID", "FAKEPATIENT", "9.9.9", "2021-03-04")
DEPTH, WIDTH, SLO = 48, 64, 96


def write_png(path, a):
    a = np.asarray(a, dtype=np.uint8)
    h, w = a.shape

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)

    raw = b"".join(b"\0" + a[i].tobytes() for i in range(h))
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def make_scan(images, name, n, seed, skip=None):
    rng = np.random.default_rng(seed)
    d = os.path.join(images, name)
    os.makedirs(d)
    for b in range(1, n + 1):
        if b == skip:
            continue
        img = rng.integers(0, 60, (DEPTH, WIDTH))
        img[20:26] = 200 + b
        write_png(os.path.join(d, "b%d.png" % b), img)
    write_png(os.path.join(d, "slo.png"), rng.integers(0, 255, (SLO, SLO)))
    with open(os.path.join(d, "metadata.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bscan_index", "pixel_start_row", "pixel_start_col", "pixel_end_row", "pixel_end_col"])
        for b in range(1, n + 1):
            w.writerow([b, 2 + b, 10, 2 + b, 70])


def make_tree(root):
    images = os.path.join(root, "oct_images")
    os.makedirs(images)
    rows = []
    for i in range(12):
        name = "9.9.9.FAKEUID%d" % i
        make_scan(images, name, 25, i)
        rows.append(("FAKEPATIENT%d" % max(0, i - 1), name, 25))
    make_scan(images, "9.9.9.FAKEUID97", 97, 97)
    rows.append(("FAKEPATIENT97", "9.9.9.FAKEUID97", 97))
    make_scan(images, "9.9.9.FAKEUIDGAP", 26, 98, skip=7)
    rows.append(("FAKEPATIENTGAP", "9.9.9.FAKEUIDGAP", 25))
    with open(os.path.join(root, "pilot_cohort_deidentified.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["anon_id", "male", "righteye", "seriesDescription", "datadir", "gaze", "sloHorizontalFieldOfView", "slox", "sloy",
                    "sloPixelspacingX", "sloPixelspacingY", "nbscans", "bscanx", "bscany", "duration", "sliceThickness",
                    "bscanPixelspacingWidth", "bscanPixelspacingDepth", "BScanWidthOrRadius", "hasSLO", "hasOCT"])
        for pid, name, n in rows:
            w.writerow([pid, 1, 1, name, "/fake/" + name, "x", 30, SLO, SLO, 0.01, 0.01, n, WIDTH, DEPTH, 1, 0.2, 0.01, 0.004, 6, 1, 1])


def run(*cmd):
    p = subprocess.run([sys.executable] + list(cmd), cwd=REPO, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    return p.returncode, p.stdout


def no_leak(text, where):
    for word in LEAK_WORDS:
        assert word not in text, "%s printed %s" % (where, word)


def test_wavelet():
    x = torch.rand(2, 1, 32, 64, 64, dtype=torch.float64) * 2 - 1
    for name, axes in wv.DESIGNS.items():
        y = wv.forward(x, axes)
        assert y.shape[1] == wv.channels(axes), name
        assert tuple(y.shape[2:]) == wv.grid((32, 64, 64), axes), name
        assert (wv.inverse(y, axes) - x).abs().max() < 1e-10, name
        assert y[:, 0].abs().max() <= 1 + 1e-9, name + " lowest band left [-1, 1]"
    assert wv.channels(wv.DESIGNS["w2z1"]) == 32 and wv.check_unet_grid((32, 512, 512), wv.DESIGNS["w2z1"]) == (16, 128, 128)
    assert wv.channels(wv.DESIGNS["w1z1"]) == 8 and wv.check_unet_grid((32, 512, 512), wv.DESIGNS["w1z1"]) == (16, 256, 256)
    for bad in ((25, 512, 512), (24, 512, 512)):
        try:
            wv.check_unet_grid(bad, wv.DESIGNS["w2z1"])
        except ValueError:
            continue
        raise AssertionError("%s should be rejected" % (bad,))


def test_matches_brain_pipeline_wavelet():
    try:
        import pywt
    except ImportError:
        print("  (pywt not installed: comparison skipped)")
        return
    x = np.random.default_rng(0).random((16, 32, 16)) * 2 - 1
    packet = pywt.WaveletPacketND(x, "haar", axes=(-3, -2, -1), maxlevel=2)
    theirs = np.stack([n.data for n in packet.get_level(2)]) / 8.0
    ours = wv.forward(torch.from_numpy(x)[None], wv.DESIGNS["w2z2"]).numpy()
    assert theirs.shape == ours.shape
    for band in theirs:
        best = np.abs(np.abs(ours) - np.abs(band)).max(axis=(1, 2, 3)).min()
        assert best < 1e-10, "band not found in our output"


def test_slice_order(root):
    folder = os.path.join(root, "oct_images", "9.9.9.FAKEUID0")
    vol = vo.load_volume(folder, 25)
    assert vol.shape == (25, DEPTH, WIDTH) and vol.dtype == np.uint8
    assert list(vol[:, 22, 0]) == [200 + b for b in range(1, 26)], "slices are not in numeric order"


def test_scale_and_pad():
    vol = np.zeros((25, 496, 512), dtype=np.uint8)
    vol[:, 100:300] = 200
    x = vo.pad(vo.scale(vol), (32, 512, 512))
    assert x.shape == (32, 512, 512) and x.dtype == np.float32
    assert x.min() == -1 and x.max() == 1
    assert (x[:3] == -1).all() and (x[-4:] == -1).all() and (x[3:28, 108:308] == 1).all(), "padding is not centered"
    for fn, args in ((vo.pad, (np.zeros((33, 512, 512)), (32, 512, 512))), (vo.scale, (np.full((3, 3, 3), 7, np.uint8),))):
        try:
            fn(*args)
        except ValueError:
            continue
        raise AssertionError("bad input should raise")


def test_depth_shift():
    rng = np.random.default_rng(0)
    depth = np.arange(496, dtype=np.float64)
    p = np.exp(-((depth - 200) / 6) ** 2) + 0.5 * np.exp(-((depth - 260) / 10) ** 2) + 0.01 * rng.random(496)
    for true in (0, 3, 7, 15, 30, 90, -4, -22, -90):
        assert vo.depth_shift(p, np.roll(p, true)) == -true, true
    assert vo.depth_shift(p, p) == 0
    flat = np.full(496, 5.0)
    assert vo.depth_shift(flat, flat) == 0, "a flat profile must not report a jump"
    assert abs(vo.depth_shift(p[:48], p[:48])) == 0, "a short profile must not break the window"


def fake_aligned(seed):
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(768, 768))
    for axis in (0, 1):
        noise = al.uniform_filter1d(noise, 9, axis=axis)
    slo = np.clip(150 + noise / noise.std() * 40, 0, 255).astype(np.uint8)
    lines = np.array([[640 - 21.3 * k, 128, 640 - 21.3 * k, 639.33] for k in range(25)])
    samples = al.sample_slo(slo, lines, 512)
    vol = np.full((25, 64, 512), 10, dtype=np.uint8)
    for k in range(25):
        top = 30 + int(rng.integers(-4, 5))
        gain = rng.uniform(0.5, 1.0)
        vol[k, top : top + 10] = np.clip(samples[k] * gain, 0, 255).astype(np.uint8)
    return vol, slo, lines


def test_layer_depths():
    vol = np.full((5, 200, 64), 10, dtype=np.uint8)
    tops = [100, 100, 130, 130, 60]
    for k, top in enumerate(tops):
        vol[k, top : top + 9] = 200
    found = vo.layer_depths(vol)
    assert np.allclose(found, [top + 4 for top in tops], atol=1), found
    assert list(np.abs(np.diff(found))) == [0, 30, 0, 70], found


def test_alignment():
    vol, slo, lines = fake_aligned(0)
    view = al.en_face(vol)
    assert np.allclose(np.median(view, axis=1), 1.0, atol=0.05), "slices should end up equally bright"
    scores, best = al.alignment_scores(vol, slo, lines)
    assert scores["as_given"] > 0.8 and max(scores, key=scores.get) == "as_given", scores
    assert best[1:] == (0, 0), best
    scores, _ = al.alignment_scores(vol[::-1], slo, lines)
    assert max(scores, key=scores.get) == "slices_reversed", scores
    scores, _ = al.alignment_scores(vol[:, :, ::-1], slo, lines)
    assert max(scores, key=scores.get) == "mirrored", scores
    scores, _ = al.alignment_scores(vol[::-1, :, ::-1], slo, lines)
    assert max(scores, key=scores.get) == "both", scores
    moved = lines + np.array([12, -9, 12, -9])
    _, best = al.alignment_scores(vol, slo, moved)
    assert best[1:] == (-12, 9), best
    _, other_slo, _ = fake_aligned(1)
    scores, _ = al.alignment_scores(vol, other_slo, lines)
    assert abs(scores["as_given"]) < 0.2, scores


def test_cohort(root):
    scans, counts = co.list_scans(root, 25)
    assert counts["kept"] == 12 and counts["other_slice_count"] == 1 and counts["bad_numbering"] == 1, counts
    train, val = co.split_by_patient(scans, 0.25)
    assert not {s["patient"] for s in train} & {s["patient"] for s in val}
    assert len(train) + len(val) == 12 and val
    assert co.split_by_patient(scans, 0.25) == (train, val), "split must repeat"
    two = [s for s in scans if s["patient"] == "FAKEPATIENT0"]
    assert len(two) == 2 and len({s in train for s in two}) == 1, "same patient on both sides"


def test_scripts(root, tmp):
    code, out = run("scripts/check_cohort.py", "--root", root)
    assert code == 0, out
    no_leak(out, "check_cohort")
    assert "scans with 25 B-scans: 13" in out and "scans with 97 B-scans: 1" in out, out
    assert "bscan_index runs 1..N in order: 13 of 13" in out, out
    assert "  increasing                 13" in out, out
    print(out)

    qc = os.path.join(tmp, "qc")
    code, out = run("scripts/qc_views.py", "--root", root, "--out", qc, "--n-views", "2")
    assert code == 0, out
    no_leak(out, "qc_views")
    assert len(glob.glob(os.path.join(qc, "qc_[0-9]*.png"))) == 2, out
    assert len(glob.glob(os.path.join(qc, "qc_worst_*.png"))) == 12, out
    assert "scans with a slice under half their typical brightness: 0" in out and "share of pixels exactly 0" in out, out
    print(out)

    code, out = run("scripts/check_slo_alignment.py", "--root", root)
    assert code == 0, out
    no_leak(out, "check_slo_alignment")
    assert "scans checked: 12   failed: 0" in out, out
    print(out)

    code, out = run("scripts/probe_memory.py", "--target", "32", "64", "64", "--steps", "1", "--max-batch", "1")
    assert code == 0, out
    print(out)

    run_dir = os.path.join(tmp, "run")
    base = ["scripts/train.py", "--root", root, "--out", run_dir, "--target", "32", "64", "64", "--batch-size", "2", "--workers", "0", "--max-steps", "2", "--val-fraction", "0.25"]
    code, out = run(*base, "--epochs", "1")
    assert code == 0, out
    no_leak(out, "train")
    assert "train scans: 9   validation scans: 3" in out, out
    print(out)
    code, out = run(*base, "--epochs", "1")
    assert code != 0 and "already has a checkpoint" in out, "a second run must not overwrite: " + out
    code, out = run(*base, "--epochs", "5", "--resume", "--keep", "2")
    assert code == 0 and "resumed after epoch 1" in out and "epoch 5" in out, out
    no_leak(out, "train resume")
    kept = sorted(os.path.basename(p) for p in glob.glob(os.path.join(run_dir, "epoch_*.pt")))
    assert "epoch_0005.pt" in kept and "epoch_0004.pt" in kept and "epoch_0002.pt" not in kept, kept
    ckpt = torch.load(os.path.join(run_dir, "latest.pt"), map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == 5 and ckpt["design"] == "w2z1"
    assert all(torch.isfinite(v).all() for v in ckpt["model"].values()), "non-finite weights"

    code, out = run("scripts/sample.py", "--run", run_dir, "--root", root, "--n-samples", "2", "--ddim-steps", "3", "--n-eval", "3")
    assert code == 0, out
    no_leak(out, "sample")
    assert "checkpoint epoch 5, design w2z1" in out and "generated 2 of 2" in out and "images written: 4" in out, out
    written = sorted(os.path.basename(p) for p in glob.glob(os.path.join(run_dir, "samples", "*.png")))
    assert written == ["real_000.png", "real_001.png", "sample_000.png", "sample_001.png"], written
    print(out)

    bad = os.path.join(root, "oct_images", "9.9.9.FAKEUID3", "b5.png")
    with open(bad, "wb") as f:
        f.write(b"not a png")
    code, out = run("scripts/train.py", "--root", root, "--out", os.path.join(tmp, "run2"), "--target", "32", "64", "64", "--batch-size", "2", "--workers", "0", "--epochs", "1", "--val-fraction", "0.25")
    assert code != 0 and "failed: UnidentifiedImageError" in out, out
    no_leak(out, "train with a broken file")


def main():
    tmp = tempfile.mkdtemp(prefix="oct_wd_test_")
    root = os.path.join(tmp, "OCT-AI")
    os.makedirs(root)
    try:
        make_tree(root)
        for name, fn, args in (
            ("wavelet", test_wavelet, ()),
            ("same bands as brain pipeline", test_matches_brain_pipeline_wavelet, ()),
            ("slice order", test_slice_order, (root,)),
            ("scale and pad", test_scale_and_pad, ()),
            ("staircase measure", test_depth_shift, ()),
            ("bright layer measure", test_layer_depths, ()),
            ("SLO alignment measure", test_alignment, ()),
            ("cohort and split", test_cohort, (root,)),
            ("scripts end to end", test_scripts, (root, tmp)),
        ):
            print("--- %s" % name, flush=True)
            fn(*args)
            print("    OK", flush=True)
        print("ALL TESTS PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


main()
