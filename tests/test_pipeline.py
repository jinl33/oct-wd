import csv
import glob
import gzip
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib

import json

import numpy as np
import torch
from PIL import Image

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from generative.networks.schedulers import DDPMScheduler

from oct_wd import alignment as al
from oct_wd import cohort as co
from oct_wd import evaluate as ev
from oct_wd import model as mo
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
        w.writerow(["anon_id", "male", "righteye", "age", "seriesDescription", "datadir", "gaze", "sloHorizontalFieldOfView", "slox", "sloy",
                    "sloPixelspacingX", "sloPixelspacingY", "nbscans", "bscanx", "bscany", "duration", "sliceThickness",
                    "bscanPixelspacingWidth", "bscanPixelspacingDepth", "BScanWidthOrRadius", "hasSLO", "hasOCT"])
        for k, (pid, name, n) in enumerate(rows):
            w.writerow([pid, ["True", "False"][k % 2], k // 3 % 2, 40 + 3 * k, name, "/fake/" + name, "x", 30, SLO, SLO, 0.01, 0.01, n, WIDTH, DEPTH, 1, 0.2, 0.01, 0.004, 6, 1, 1])


def run(*cmd):
    p = subprocess.run([sys.executable] + list(cmd), cwd=REPO, env=ENV, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    return p.returncode, p.stdout


def no_leak(text, where):
    for word in LEAK_WORDS:
        assert word not in text, "%s printed %s" % (where, word)


def band_rows(text):
    lines = text.split("wavelet bands by number of detail axes")[1].splitlines()
    return [l.split()[2] for l in lines if l.split() and l.split()[0].isdigit()]


def test_wavelet():
    x = torch.rand(2, 1, 32, 64, 64, dtype=torch.float64) * 2 - 1
    for name, axes in wv.DESIGNS.items():
        y = wv.forward(x, axes)
        assert y.shape[1] == wv.channels(axes), name
        assert tuple(y.shape[2:]) == wv.grid((32, 64, 64), axes), name
        assert (wv.inverse(y, axes) - x).abs().max() < 1e-10, name
        assert y[:, 0].abs().max() <= 1 + 1e-9, name + " lowest band left [-1, 1]"
    y = wv.forward(x, wv.DESIGNS["w2z1"])
    band = torch.rand(32) + 0.01
    assert (wv.unscaled(wv.scaled(y, band), band) - y).abs().max() < 1e-6 and wv.scaled(y, None) is y
    assert (wv.scaled(y, band)[:, 5] - y[:, 5] / band[5].item()).abs().max() < 1e-6
    assert [round(v, 6) for v in wv.by_detail_axes(torch.full((32,), 2.0), 5)] == [2.0] * 6
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
    rng = np.random.default_rng(1)
    slo_dir = tempfile.mkdtemp(prefix="oct_wd_slo_")
    write_png(os.path.join(slo_dir, "slo.png"), rng.integers(0, 255, (96, 96)))
    slo = vo.load_slo(slo_dir, 64)
    shutil.rmtree(slo_dir, ignore_errors=True)
    assert slo.shape == (64, 64) and slo.dtype == np.float32 and slo.min() >= -1 and slo.max() <= 1, (slo.shape, slo.dtype)
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


def test_split_steps():
    rng = np.random.default_rng(0)
    depth = np.arange(496, dtype=np.float64)
    p = np.exp(-((depth - 200) / 6) ** 2) + 0.5 * np.exp(-((depth - 260) / 10) ** 2) + 0.01 * rng.random(496)

    def stack(path):
        return np.stack([np.roll(p, int(round(y))) for y in path])

    slices = np.arange(25)
    steps = vo.neighbor_steps(stack(5 * slices))
    assert (np.abs(steps) == 5).all(), steps
    slope, sudden = vo.split_steps(steps)
    assert sudden.max() == 0 and (np.abs(slope) == 5).all(), "a steady tilt must not read as a sudden jump"

    _, sudden = vo.split_steps(vo.neighbor_steps(stack(0.1 * (slices - 12) ** 2)))
    assert sudden.max() <= 1, "a curved retina must not read as a sudden jump: %s" % sudden

    spike = np.zeros(25)
    spike[12] = 30
    _, sudden = vo.split_steps(vo.neighbor_steps(stack(spike)))
    assert sudden.max() == 30 and (sudden >= 5).sum() == 2, "one slice out of line must read at its true size: %s" % sudden

    moved = (slices > 12) * 30.0
    _, sudden = vo.split_steps(vo.neighbor_steps(stack(moved)))
    assert sudden.max() == 30 and (sudden >= 5).sum() == 1, "a lasting jump must read at its true size: %s" % sudden

    tilted = 4 * slices + spike
    _, sudden = vo.split_steps(vo.neighbor_steps(stack(tilted)))
    assert sudden.max() == 30, "a jump on a tilted retina must read without the tilt: %s" % sudden


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
    train, val, test = co.split_by_patient(scans, 0.2, 0.2)
    sets = [{s["patient"] for s in part} for part in (train, val, test)]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]), "patient in two parts"
    assert len(train) + len(val) + len(test) == 12 and val and test
    assert [len(x) for x in sets] == [7, 2, 2], [len(x) for x in sets]
    assert co.split_by_patient(scans, 0.2, 0.2) == (train, val, test), "split must repeat"
    two = [s for s in scans if s["patient"] == "FAKEPATIENT0"]
    assert len(two) == 2 and len({(s in train, s in val) for s in two}) == 1, "same patient on both sides"
    old_train, old_val = co.split_by_patient(scans, 0.25, 0.0)[:2]
    assert co.split_by_patient(scans, 0.25, 0.0)[2] == [] and old_val, "a zero test share must leave test empty"
    big = [{"patient": "P%03d" % i} for i in range(531)]
    assert [len(x) for x in co.split_by_patient(big)] == [372, 53, 106], "default must be 70/10/20"
    cases = {"True": 1.0, "false": 0.0, "1": 1.0, "0": 0.0, " 2.5 ": 2.5, "": None, "nan": None, "abc": None}
    for raw, want in cases.items():
        assert co.label({"row": {"x": raw}}, "x") == want, raw
    assert co.label({"row": {}}, "x") is None


def test_models():
    for n in (2, 3, 7):
        assert all(vo.other_index(i, n) != i for i in range(n)), n
    x = torch.rand(2, 32, 16, 32, 32) * 2 - 1
    t = torch.tensor([3, 700])
    plain = mo.build_model(32).eval()
    enc = mo.Encoder(plain).eval()
    seen = {}
    plain.middle_block.register_forward_hook(lambda m, i, o: seen.update(mid=o))
    with torch.no_grad():
        full = plain(x=x, timesteps=t)
        feats = enc(x, t)
    assert full.shape == x.shape and feats.shape == (2, enc.dim) and enc.dim == 560, feats.shape
    assert torch.allclose(feats[:, -160:], seen["mid"].mean(dim=(2, 3, 4)), atol=1e-5), "encoder must match the U-Net's own bottleneck"
    n_enc = sum(p.numel() for p in enc.parameters())
    n_all = sum(p.numel() for p in plain.parameters())
    assert n_enc < 0.6 * n_all, (n_enc, n_all)
    cond = mo.build_model(32, 64).eval()
    with torch.no_grad():
        for prm in cond.parameters():
            if (prm == 0).all():
                prm.normal_(0, 0.02)
    slo = torch.rand(2, 64, 64) * 2 - 1
    tokens = cond.slo(slo, torch.tensor([True, False]))
    assert tokens.shape == (2, 64, mo.CONTEXT_DIM)
    assert torch.equal(tokens[1], cond.slo.null) and not torch.equal(tokens[0], cond.slo.null), "dropped SLO must become the null tokens"
    with torch.no_grad():
        own = mo.predict(cond, x, t, slo)
        swapped = mo.predict(cond, x, t, slo.flip(0))
        none = mo.predict(cond, x, t, None)
        feats = mo.Encoder(cond).eval()(x, t, slo)
    assert own.shape == x.shape and none.shape == x.shape and feats.shape == (2, 560)
    assert not torch.allclose(own, swapped), "the SLO must reach the U-Net"
    assert ev.auroc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == 0.75
    assert ev.auroc([1, 1], [0.1, 0.2]) != ev.auroc([1, 1], [0.1, 0.2]), "one class must give nan"
    low, high = ev.bootstrap("binary", np.array([0, 1] * 20), np.arange(40) % 2 + np.random.default_rng(0).normal(0, 0.5, 40), 200)
    assert 0.5 < low <= high <= 1.0, (low, high)
    r = ev.scores("regression", [1, 2, 3], [1, 2, 4])
    assert abs(r["mae"] - 1 / 3) < 1e-9 and r["pearson_r"] > 0.9
    assert ev.better("binary", 0.8, 0.7) and ev.better("regression", 1.0, 2.0) and ev.better("binary", 0.6, float("nan"))


def fails(fn, error=ValueError):
    try:
        fn()
    except error as e:
        return str(e)
    raise AssertionError("expected %s" % error.__name__)


def test_adapter_parts(root):
    lines = np.array([[10.0 + 10 * k, 20.0, 10.0 + 10 * k, 180.0] for k in range(4)])
    g = vo.slo_grid(lines, (100, 200), (4, 5, 9), (8, 5, 9))

    def x_of(col):
        return (col + 0.5) / 200 * 2 - 1

    def y_of(row):
        return (row + 0.5) / 100 * 2 - 1

    assert g.shape == (3, 8, 9) and g[2].sum() == 4 * 9 and (g[2, 2:6] == 1).all(), g[2]
    assert np.allclose(g[0, 2, [0, 8]], [x_of(20), x_of(180)]) and np.allclose(g[1, 2:6, 4], [y_of(r) for r in (10, 20, 30, 40)])
    assert np.allclose(g[1, [0, 1, 6, 7], 0], [y_of(r) for r in (-10, 0, 50, 60)]), "padded slices continue the raster"
    wide = vo.slo_grid(lines, (100, 200), (4, 5, 9), (4, 5, 13))
    assert wide[2].sum() == 4 * 9 and (wide[2][:, :2] == 0).all() and (wide[2][:, 11:] == 0).all() and np.isclose(wide[0, 0, 2], x_of(20))
    fails(lambda: vo.slo_grid(lines[:3], (100, 200), (4, 5, 9), (8, 5, 9)))

    w2 = wv.DESIGNS["w2z1"]
    assert wv.check_unet_grid((24, 64, 64), w2, (0, 0, 1, 1)) == (12, 16, 16)
    assert "(16, 16, 16)" in fails(lambda: wv.check_unet_grid((24, 64, 64), w2))
    fails(lambda: mo.build_model(32, z_down=(0, 1)))
    fails(lambda: mo.build_model(32, 0, adapter=True))

    flat = mo.build_model(32, z_down=(0, 0, 1, 1)).eval()
    assert set(flat.state_dict()) == set(mo.build_model(32).state_dict()), "the slice-axis option must keep the weight names"
    shapes = []
    hooks = [b.register_forward_hook(lambda m, i, o: shapes.append(tuple(o[0].shape[2:]))) for b in flat.down_blocks]
    x24 = torch.rand(1, 32, 12, 16, 16) * 2 - 1
    with torch.no_grad():
        out = flat(x=x24, timesteps=torch.tensor([10]))
    for h in hooks:
        h.remove()
    with torch.no_grad():
        f24 = mo.Encoder(flat).eval()(x24, torch.tensor([10]))
    assert out.shape == x24.shape and f24.shape == (1, 560), (out.shape, f24.shape)
    assert shapes == [(12, 8, 8), (12, 4, 4), (6, 2, 2), (3, 1, 1), (3, 1, 1)], shapes

    torch.manual_seed(0)
    a = mo.build_model(32, 64, adapter=True).eval()
    assert all((p == 0).all() for p in a.adapter.parameters()), "the adapter must start at zero"
    with torch.no_grad():
        for name, prm in a.named_parameters():
            if not name.startswith("adapter.") and (prm == 0).all():
                prm.normal_(0, 0.02)
    b = mo.build_model(32, 64).eval()
    b.load_state_dict({k: v for k, v in a.state_dict().items() if not k.startswith("adapter.")})
    lines25 = np.array([[3.0 + k, 10.0, 3.0 + k, 70.0] for k in range(25)])
    grid = torch.from_numpy(vo.slo_grid(lines25, (96, 96), (25, 128, 128), (32, 128, 128)))[None].expand(2, -1, -1, -1)
    img = torch.rand(2, 64, 64) * 2 - 1
    x = torch.rand(2, 32, 16, 32, 32) * 2 - 1
    t = torch.tensor([3, 700])
    with torch.no_grad():
        start = a(x, t, (img, grid))
        ref = b(x, t, img)
        fa0, fb = mo.Encoder(a).eval()(x, t, (img, grid)), mo.Encoder(b).eval()(x, t, img)
        assert torch.allclose(a.slo.to_tokens(a.slo.stages(img)[-1]), a.slo(img), atol=1e-6)
    assert torch.allclose(start, ref, atol=1e-5) and torch.allclose(fa0, fb, atol=1e-5), "a new adapter must start as the plain SLO model"
    fails(lambda: a(x, t, img))

    with torch.no_grad():
        for prm in a.adapter.parameters():
            prm.normal_(0, 0.05)
        moved_grid = grid.clone()
        moved_grid[:, 0] += 0.3
        own = a(x, t, (img, grid))
        moved = a(x, t, (img, moved_grid))
        none = a(x, t, None)
        dropped = a(x, t, (img, grid), torch.tensor([False, False]))
        half = a(x, t, (img, grid), torch.tensor([True, False]))
        enc = mo.Encoder(a).eval()
        fa_own, fa_none = enc(x, t, (img, grid)), enc(x, t, None)
        fb_own, fb_none = mo.Encoder(b).eval()(x, t, img), mo.Encoder(b).eval()(x, t, None)
        add = a.adapter(a.slo.stages(img), grid)(0, torch.zeros(2, 40, 16, 32, 32))
    assert not torch.allclose(own, ref, atol=1e-4) and not torch.allclose(own, moved, atol=1e-4), "the adapter and the scan lines must reach the output"
    assert torch.allclose(dropped, none, atol=1e-5) and torch.allclose(half[1], none[1], atol=1e-5) and torch.allclose(half[0], own[0], atol=1e-5), "a dropped SLO must give the no-SLO prediction"
    assert not torch.allclose(fa_own[:, :40], fa_none[:, :40], atol=1e-5), "the adapter must reach the first encoder level"
    assert torch.allclose(fb_own[:, :160], fb_none[:, :160], atol=1e-6), "cross-attention alone reaches only the deep levels"
    assert (add[:, :, [0, 14, 15]] == 0).all() and add[:, :, 2:14].abs().sum() > 0, "padded slices must get no SLO features"

    scans, _ = co.list_scans(root, 25)
    s = scans[0]
    assert np.array_equal(vo.load_volume(s["folder"], 25, 24), vo.load_volume(s["folder"], 25)[:24])
    fails(lambda: vo.load_volume(s["folder"], 25, 26))
    x0, (img0, g0), (other_img, other_grid) = vo.OctVolumes(scans[:3], (32, 64, 64), 64, other=True, adapter=True)[0]
    assert x0.shape == (1, 32, 64, 64) and img0.shape == (64, 64) and g0.shape == (3, 32, 64)
    assert torch.equal(g0, other_grid) and not torch.equal(img0, other_img), "the other-SLO test swaps the image, not the scan lines"
    assert g0[2].sum() == 25 * 64 and (g0[2, 3:28] == 1).all()
    assert torch.allclose(g0[1, 3:28, 0], torch.tensor([(2 + k + 0.5) / SLO * 2 - 1 for k in range(1, 26)]))
    assert torch.allclose(g0[0, 3, [0, WIDTH - 1]], torch.tensor([10.5 / SLO * 2 - 1, 70.5 / SLO * 2 - 1]))
    x24, (_, g24) = vo.OctVolumes(scans[:1], (24, 64, 64), 64, adapter=True, keep=24)[0]
    assert x24.shape == (1, 24, 64, 64) and g24.shape == (3, 24, 64) and bool(g24[2].all()) and torch.allclose(g24[:2], g0[:2, 3:27])
    plain = vo.OctVolumes(scans[:2], (32, 64, 64), 64, other=True)[0]
    assert len(plain) == 3 and plain[1].shape == (64, 64) and plain[2].shape == (64, 64), "the plain SLO model keeps its batch layout"


def test_adapter_runs(root, tmp):
    common = ["--root", root, "--batch-size", "2", "--workers", "0", "--max-steps", "2", "--val-fraction", "0.2", "--test-fraction", "0.2"]
    ada = os.path.join(tmp, "run_adapter")
    ada_base = ["scripts/train.py", *common, "--out", ada, "--target", "32", "64", "64", "--slo-size", "64", "--slo-adapter", "--band-scale", "0.5"]
    code, out = run(*ada_base, "--epochs", "1")
    assert code == 0 and "SLO 64 px, adapter" in out and "(own SLO)  other SLO" in out and "B-scans kept" not in out, out
    no_leak(out, "train with the SLO adapter")
    print(out)
    ckpt = torch.load(os.path.join(ada, "latest.pt"), map_location="cpu", weights_only=False)
    assert ckpt["slo_adapter"] is True and ckpt["keep_slices"] == 0 and ckpt["z_down"] == [1, 1, 1, 1]
    assert any(k.startswith("adapter.") for k in ckpt["model"]) and all(torch.isfinite(v).all() for v in ckpt["model"].values())
    code, out = run(*ada_base, "--epochs", "2", "--auto-resume")
    assert code == 0 and "resumed after epoch 1" in out, out
    code, out = run(*[a for a in ada_base if a != "--slo-adapter"], "--epochs", "3", "--auto-resume")
    assert code != 0 and "SLO adapter on" in out, out
    code, out = run("scripts/train.py", *common, "--out", os.path.join(tmp, "run_no_slo"), "--target", "32", "64", "64", "--slo-adapter", "--epochs", "1")
    assert code != 0 and "--slo-adapter needs --slo-size" in out, out
    code, out = run("scripts/sample.py", "--run", ada, "--root", root, "--n-samples", "1", "--ddim-steps", "2", "--n-eval", "2")
    assert code == 0 and "SLO 64 px, adapter" in out and "ratio, own SLO" in out and "from the SLO of real_000" in out, out
    no_leak(out, "sample with the SLO adapter")
    for extra in ([], ["--random-init"], ["--mode", "finetune", "--epochs", "1"]):
        code, out = run("scripts/downstream.py", "--run", ada, "--root", root, "--task", "righteye", "--workers", "0", "--n-boot", "50", *extra)
        assert code == 0 and "SLO 64 px, adapter" in out and "test auroc" in out, out
        no_leak(out, "downstream with the SLO adapter")
    with open(os.path.join(ada, "downstream", "righteye_probe_t0", "results.json")) as f:
        assert json.load(f)["slo_adapter"] is True

    flat = os.path.join(tmp, "run_24")
    flat_base = ["scripts/train.py", *common, "--out", flat, "--target", "24", "64", "64", "--keep-slices", "24", "--z-down", "0", "0", "1", "1"]
    code, out = run(*flat_base, "--epochs", "1")
    assert code == 0 and "wavelet grid (12, 16, 16)" in out and "B-scans kept: 24 of 25; slice axis halved at 2 of 4 downsamplings (0 0 1 1)" in out, out
    no_leak(out, "train on 24 slices")
    print(out)
    code, out = run(*flat_base[:-4], "0", "1", "1", "1", "--epochs", "2", "--auto-resume")
    assert code != 0 and "slice-axis halving 0 0 1 1" in out, out
    code, out = run(*flat_base[:-5], "--out", os.path.join(tmp, "run_24_iso"), "--epochs", "1")
    assert code != 0 and "must be divisible by (16, 16, 16)" in out, out
    code, out = run(*flat_base[:-4], "0", "1", "1", "1", "--out", os.path.join(tmp, "run_24_three"), "--epochs", "1")
    assert code != 0 and "must be divisible by (8, 16, 16)" in out, out
    code, out = run("scripts/train.py", *common, "--out", flat, "--target", "32", "64", "64", "--epochs", "2", "--auto-resume")
    assert code != 0 and "slices kept 24" in out, out
    code, out = run("scripts/train.py", *common, "--out", os.path.join(tmp, "run_short"), "--target", "24", "64", "64", "--z-down", "0", "0", "1", "1", "--epochs", "1")
    assert code != 0 and "fewer than the 25 B-scans kept" in out, out
    code, out = run("scripts/sample.py", "--run", flat, "--root", root, "--n-samples", "1", "--ddim-steps", "2", "--n-eval", "2")
    assert code == 0 and "generated 1 of 1" in out, out
    with Image.open(os.path.join(flat, "samples", "sample_000.png")) as im:
        assert im.size == (WIDTH + 24 * 8 + WIDTH + 20, DEPTH), im.size
    code, out = run("scripts/downstream.py", "--run", flat, "--root", root, "--task", "male", "--workers", "0", "--n-boot", "50")
    assert code == 0 and "test auroc" in out, out

    both = os.path.join(tmp, "run_adapter_24")
    code, out = run("scripts/train.py", *common, "--out", both, "--target", "24", "64", "64", "--keep-slices", "24", "--z-down", "0", "0", "1", "1", "--slo-size", "64", "--slo-adapter", "--band-scale", "0.5", "--epochs", "1")
    assert code == 0 and "SLO 64 px, adapter" in out and "B-scans kept: 24" in out, out
    no_leak(out, "train with the adapter on 24 slices")
    code, out = run("scripts/downstream.py", "--run", both, "--root", root, "--task", "righteye", "--mode", "finetune", "--epochs", "1", "--workers", "0", "--n-boot", "50")
    assert code == 0 and "test auroc" in out, out
    report = os.path.join(tmp, "report_adapter_24")
    code, out = run("scripts/scan_report.py", "--run", both, "--root", root, "--out", report, "--max-scans", "2", "--steps", "50", "--group", "--n-worst", "1")
    assert code == 0 and "SLO 64 px, adapter" in out and "scored 2 scans, could not read 0" in out, out
    no_leak(out, "scan report with the adapter on 24 slices")
    with Image.open(os.path.join(report, "worst_000.png")) as im:
        assert im.size == (3 * WIDTH + 20, DEPTH), im.size
    code, out = run("scripts/probe_memory.py", "--target", "24", "64", "64", "--steps", "1", "--max-batch", "1", "--designs", "w2z1", "--slo-size", "64", "--slo-adapter", "--z-down", "0", "0", "1", "1")
    assert code == 0 and "SLO 64 px, adapter" in out and "halved at 2 of 4" in out and "batch  1: OK" in out, out


def test_report_measures():
    a = torch.zeros(3, 40, 40, dtype=torch.float64)
    assert abs(ev.psnr(a + 0.1, a) - 10 * np.log10(4 / 0.01)) < 1e-9 and ev.psnr(a, a) == float("inf")
    x = torch.rand(3, 40, 40) * 2 - 1
    y = (x + 0.3 * torch.randn(3, 40, 40)).clamp(-1, 1)
    s = ev.ssim(x, y)
    assert abs(ev.ssim(x, x) - 1) < 1e-12 and 0 < s < 1, s
    try:
        from skimage.metrics import structural_similarity
    except (ImportError, ValueError):
        print("  (scikit-image not installed or broken: SSIM comparison skipped)")
    else:
        ref = np.mean([structural_similarity(x[k].double().numpy(), y[k].double().numpy(), data_range=2.0, gaussian_weights=True, sigma=1.5, use_sample_covariance=False) for k in range(3)])
        assert abs(ref - s) < 1e-9, (ref, s)
    scans = [{"row": {"male": ["True", "False"][k % 2], "age": str(20 + k), "gaze": "", "anon_id": "P%d" % k, "visit_date": "x", "when": "2021-01-0%d" % (k % 3 + 1), "eye": "LR"[k % 2] if k else ""}} for k in range(20)]
    values, names = co.group_labels(scans, "male")
    assert names == ["False", "True"] and values[:2] == ["True", "False"], (values, names)
    values, names = co.group_labels(scans, "age")
    assert names == ["20 to 24.8", "24.8 to 29.5", "29.5 to 34.2", "34.2 to 39"] and values[0] == names[0] and values[-1] == names[-1], names
    assert [values.count(n) for n in names] == [5, 5, 5, 5], values
    values, names = co.group_labels(scans, "eye")
    assert values[0] is None and names == ["L", "R"]
    for bad in ("anon_id", "visit_date", "when", "gaze", "not_there"):
        try:
            co.group_labels(scans, bad)
        except SystemExit as e:
            assert "P1" not in str(e) and "2021" not in str(e), e
            continue
        raise AssertionError("%s should be refused as a group" % bad)


def test_report(root, tmp):
    plain, cond = os.path.join(tmp, "report_plain"), os.path.join(tmp, "report_slo_band")
    train = ["scripts/train.py", "--root", root, "--target", "32", "64", "64", "--batch-size", "2", "--workers", "0", "--max-steps", "2", "--val-fraction", "0.2", "--test-fraction", "0.2", "--epochs", "1"]
    code, out = run(*train, "--out", plain)
    assert code == 0, out
    code, out = run(*train, "--out", cond, "--slo-size", "64", "--band-scale", "0.5")
    assert code == 0, out
    with open(os.path.join(plain, "split.json")) as f:
        split = json.load(f)
    scans, _ = co.list_scans(root, 25)
    held = [s for s in scans if s["patient"] in set(split["val"]) | set(split["test"])]
    trained = [s for s in scans if s["patient"] in set(split["train"])]

    report = os.path.join(tmp, "report_out")
    args = ["scripts/scan_report.py", "--run", plain, "--root", root, "--steps", "500", "50", "--group", "male", "righteye", "age", "--n-worst", "2", "--out", report]
    code, out = run(*args)
    assert code == 0, out
    no_leak(out, "scan_report")
    assert "checkpoint epoch 1, design w2z1, SLO not used, band scaling off" in out, out
    assert "scored %d scans, could not read 0" % len(held) in out and "scored here: %d from val and test" % len(held) in out, out
    assert "male, scans per value (train / val / test):" in out and "by righteye:" in out and "by age:" in out and "  50     PSNR p " in out, out
    assert "lowest PSNR at step 500" in out and "2 worst_*.png" in out, out
    with open(os.path.join(report, "scans.csv")) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(held) and {r["part"] for r in rows} <= {"val", "test"}, rows
    assert list(rows[0]) == ["part", "male", "righteye", "age", "psnr_t50", "noisy_psnr_t50", "ssim_t50", "psnr_t500", "noisy_psnr_t500", "ssim_t500"], list(rows[0])
    for r in rows:
        assert all(np.isfinite(float(r[k])) for k in r if "_t" in k), r
        assert -1 <= float(r["ssim_t50"]) <= 1 and float(r["noisy_psnr_t50"]) > float(r["noisy_psnr_t500"]), r
    with open(os.path.join(report, "scans.csv")) as f:
        text = f.read()
    no_leak(text, "scans.csv")
    made = sorted(os.listdir(report))
    try:
        import matplotlib
        plots = ["cohort.png", "reconstruction.png", "reconstruction_age.png", "reconstruction_male.png", "reconstruction_righteye.png"]
    except ImportError:
        plots = []
    assert made == sorted(plots + ["scans.csv", "worst_000.png", "worst_001.png"]), made
    with Image.open(os.path.join(report, "worst_000.png")) as im:
        assert im.size == (3 * WIDTH + 20, DEPTH), im.size
    print(out)

    code, again = run(*args[:-1], report + "_again")
    with open(os.path.join(report + "_again", "scans.csv")) as f:
        assert code == 0 and f.read() == text, "a second report must give the same numbers"

    code, out = run("scripts/scan_report.py", "--run", plain, "--root", root, "--parts", "train", "--steps", "100", "--group", "--n-worst", "0", "--out", report + "_train")
    assert code == 0 and "scored %d scans" % len(trained) in out and "by " not in out and "lowest PSNR" not in out, out
    no_leak(out, "scan_report on training scans")

    code, out = run("scripts/scan_report.py", "--run", cond, "--root", root, "--steps", "200", "--max-scans", "2", "--out", report + "_cond")
    assert code == 0 and "SLO 64 px, band scaling on" in out and "scored 2 scans" in out, out
    no_leak(out, "scan_report with SLO and band scaling")
    print(out)

    for extra, want in ((["--group", "anon_id"], "cannot be a group"), (["--group", "seriesDescription"], "cannot be a group"),
                        (["--group", "datadir"], "cannot be a group"), (["--steps", "0"], "between 1 and 999")):
        code, out = run("scripts/scan_report.py", "--run", plain, "--root", root, "--out", report + "_bad", *extra)
        assert code != 0 and want in out, out
        no_leak(out, "scan_report refusal")


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

    stack = os.path.join(tmp, "stack")
    code, out = run("scripts/stack_views.py", "--root", root, "--out", stack, "--n-views", "2")
    assert code == 0, out
    no_leak(out, "stack_views")
    views = sorted(glob.glob(os.path.join(stack, "stack_*.png")))
    assert len(views) == 2, out
    with Image.open(views[0]) as im:
        assert im.size == (25 * 16 * 3 + 20, DEPTH), im.size
    assert "scans checked: 12   failed: 0" in out and "scans by largest sudden jump" in out and "view 001:" in out, out
    print(out)

    dense = os.path.join(tmp, "dense")
    code, out = run("scripts/stack_views.py", "--root", root, "--out", dense, "--n-slices", "97")
    assert code == 0, out
    no_leak(out, "stack_views dense")
    assert "as 25 slices, scans by largest sudden jump" in out and "view 000:" in out and "as 25 slices:" in out, out
    with Image.open(os.path.join(dense, "stack_000.png")) as im:
        assert im.size == (97 * 4 * 3 + 20, DEPTH), im.size
    print(out)

    nii = os.path.join(tmp, "nifti")
    for n_slices, n_views in (("25", "2"), ("97", "1")):
        code, out = run("scripts/export_nifti.py", "--root", root, "--out", nii, "--n-slices", n_slices, "--n-views", n_views)
        assert code == 0, out
        no_leak(out, "export_nifti")
        assert "spacing values missing 0" in out, out
        print(out)
    files = sorted(os.listdir(nii))
    assert files == ["n25_000.nii.gz", "n25_001.nii.gz", "n97_000.nii.gz"], files
    for name, n in (("n25_000.nii.gz", 25), ("n97_000.nii.gz", 97)):
        with gzip.open(os.path.join(nii, name), "rb") as f:
            raw = f.read()
        assert len(raw) == 352 + n * DEPTH * WIDTH and raw[344:348] == b"n+1\x00", len(raw)
        data = np.frombuffer(raw[352:], dtype=np.uint8).reshape(n, DEPTH, WIDTH)
        assert (data[:, 22, 0] == (201 + np.arange(n)) % 256).all(), "slices must keep their order"
        try:
            import nibabel
        except ImportError:
            continue
        img = nibabel.load(os.path.join(nii, name))
        assert img.shape == (WIDTH, DEPTH, n), img.shape
        assert (np.asarray(img.dataobj)[0, 22, :] == (201 + np.arange(n)) % 256).all(), "axes must be across, depth, slice"
        expected = np.array([[0.004, 0, 0, 0], [0, 0, 0.2, 0], [0, -0.01, 0, 0], [0, 0, 0, 1]])
        assert np.allclose(img.affine, expected, atol=1e-6), img.affine
        assert np.allclose(img.header.get_qform(), img.header.get_sform(), atol=1e-6), "qform and sform must agree"

    code, out = run("scripts/check_slo_alignment.py", "--root", root)
    assert code == 0, out
    no_leak(out, "check_slo_alignment")
    assert "scans checked: 12   failed: 0" in out, out
    print(out)

    code, out = run("scripts/probe_memory.py", "--target", "32", "64", "64", "--steps", "1", "--max-batch", "1")
    assert code == 0, out
    print(out)

    code, out = run("scripts/probe_memory.py", "--target", "32", "64", "64", "--steps", "1", "--max-batch", "1", "--designs", "w2z1", "--slo-size", "64")
    assert code == 0 and "SLO 64 px" in out and "batch  1: OK" in out, out

    run_dir = os.path.join(tmp, "run")
    base = ["scripts/train.py", "--root", root, "--out", run_dir, "--target", "32", "64", "64", "--batch-size", "2", "--workers", "0", "--max-steps", "2", "--val-fraction", "0.2", "--test-fraction", "0.2"]
    code, out = run(*base, "--epochs", "1")
    assert code == 0, out
    no_leak(out, "train")
    scans, _ = co.list_scans(root, 25)
    parts = co.split_by_patient(scans, 0.2, 0.2)
    assert "train scans: %d   validation scans: %d   test scans (never used here): %d" % tuple(len(p) for p in parts) in out, out
    with open(os.path.join(run_dir, "split.json")) as f:
        saved = json.load(f)
    assert sorted(saved) == ["test", "train", "val"] and len(saved["test"]) == 2, saved.keys()
    print(out)
    code, out = run(*base, "--epochs", "1")
    assert code != 0 and "already has a checkpoint" in out, "a second run must not overwrite: " + out
    code, out = run(*base, "--epochs", "2", "--auto-resume", "--test-fraction", "0.3")
    assert code != 0 and "split differs" in out, "a changed split must not continue a run: " + out
    code, out = run(*base, "--epochs", "2", "--auto-resume", "--design", "w1z1")
    assert code != 0 and "trained with design w2z1" in out, out
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
    assert "checkpoint epoch 5, design w2z1, SLO not used" in out and "generated 2 of 2" in out and "images written: 4" in out, out
    assert "wavelet bands by number of detail axes" in out and "noise added at step 0, 10, 50: 0.010, 0.047, 0.173" in out and "band scaling off" in out, out
    plain_bands = band_rows(out)
    written = sorted(os.path.basename(p) for p in glob.glob(os.path.join(run_dir, "samples", "*.png")))
    assert written == ["real_000.png", "real_001.png", "sample_000.png", "sample_001.png"], written
    print(out)

    for extra, want in (([], "test auroc"), (["--random-init"], "random init"), (["--mode", "finetune", "--epochs", "2"], "epoch 2: train loss")):
        code, out = run("scripts/downstream.py", "--run", run_dir, "--root", root, "--task", "male", "--workers", "0", "--n-boot", "50", *extra)
        assert code == 0 and want in out and "decoder removed" in out and "positives" in out, out
        no_leak(out, "downstream")
        print(out)
    code, out = run("scripts/downstream.py", "--run", run_dir, "--root", root, "--task", "age", "--kind", "regression", "--workers", "0", "--n-boot", "50", "--t", "50")
    assert code == 0 and "test mae" in out, out
    with open(os.path.join(run_dir, "downstream", "age_probe_t50", "results.json")) as f:
        res = json.load(f)
    assert res["splits"]["test"]["n"] in (2, 3) and "mae" in res["test"] and len(res["test_ci95"]) == 2, res
    code, out = run("scripts/downstream.py", "--run", run_dir, "--root", root, "--task", "age", "--workers", "0")
    assert code != 0 and "is not 0/1" in out, out

    few = ["scripts/downstream.py", "--run", run_dir, "--root", root, "--workers", "0", "--n-boot", "50"]
    code, out = run(*few, "--task", "male", "--label-fraction", "0.5", "--subset-seed", "1")
    assert code == 0 and "label fraction 0.5, subset 1:" in out and "test auroc" in out, out
    no_leak(out, "downstream with fewer labels")
    with open(os.path.join(run_dir, "downstream", "male_probe_t0_f0.5_s1", "results.json")) as f:
        res = json.load(f)
    with open(os.path.join(run_dir, "downstream", "male_probe_t0", "results.json")) as f:
        full = json.load(f)
    assert res["label_fraction"] == 0.5 and 0 < res["splits"]["train"]["n"] < full["splits"]["train"]["n"], (res["splits"], full["splits"])
    assert res["splits"]["test"] == full["splits"]["test"] and res["splits"]["val"] == full["splits"]["val"]
    code, out = run(*few, "--task", "male", "--mode", "finetune", "--epochs", "1", "--label-fraction", "0.5", "--subset-seed", "1")
    assert code == 0 and "finetune for 2 epochs, validating every 2" in out and "epoch 2: train loss" in out and "epoch 1:" not in out, out
    code, out = run(*few, "--task", "righteye", "--label-fraction", "0.01")
    assert code != 0 and "one class only" in out, out
    code, out = run(*few, "--task", "male", "--label-fraction", "0")
    assert code != 0 and "must be above 0" in out, out
    code, out = run("scripts/summarize_downstream.py", "--run", run_dir)
    assert code == 0 and "random init" in out and "finetune" in out and " 0.5 " in out, out
    no_leak(out, "summary")
    print(out)

    slo_dir = os.path.join(tmp, "run_slo")
    slo_base = ["scripts/train.py", "--root", root, "--out", slo_dir, "--target", "32", "64", "64", "--batch-size", "2", "--workers", "0", "--max-steps", "2", "--slo-size", "64", "--accum", "2", "--val-fraction", "0.2", "--test-fraction", "0.2"]
    code, out = run(*slo_base, "--epochs", "1", "--auto-resume")
    assert code == 0 and "SLO 64 px" in out and "(own SLO)  other SLO" in out and "no SLO" in out, out
    code, out = run(*slo_base, "--epochs", "2", "--auto-resume")
    assert code == 0 and "resumed after epoch 1" in out, out
    no_leak(out, "train with SLO")
    print(out)
    code, out = run("scripts/sample.py", "--run", slo_dir, "--root", root, "--n-samples", "1", "--ddim-steps", "2", "--n-eval", "2")
    assert code == 0 and "ratio, own SLO" in out and "from the SLO of real_000" in out, out
    no_leak(out, "sample with SLO")
    print(out)
    code, out = run("scripts/downstream.py", "--run", slo_dir, "--root", root, "--task", "righteye", "--workers", "0", "--n-boot", "50")
    assert code == 0 and "SLO 64 px" in out, out
    no_leak(out, "downstream with SLO")
    print(out)

    band_dir = os.path.join(tmp, "run_band")
    band_base = ["scripts/train.py", "--root", root, "--out", band_dir, "--target", "32", "64", "64", "--batch-size", "2", "--workers", "0", "--val-fraction", "0.2", "--test-fraction", "0.2", "--band-scale"]
    code, out = run(*band_base, "--epochs", "1")
    assert code == 0 and "band scaling measured on" in out, out
    no_leak(out, "train with band scaling")
    print(out)
    ckpt = torch.load(os.path.join(band_dir, "latest.pt"), map_location="cpu", weights_only=False)
    with open(os.path.join(band_dir, "split.json")) as f:
        train_ids = set(json.load(f)["train"])
    vols = vo.OctVolumes([s for s in scans if s["patient"] in train_ids], (32, 64, 64))
    x = wv.scaled(wv.forward(torch.stack([vols[i] for i in range(len(vols))]), wv.DESIGNS["w2z1"]), torch.tensor(ckpt["band_scale"]))
    rms = x.pow(2).mean(dim=(0, 2, 3, 4)).sqrt()
    assert len(ckpt["band_scale"]) == 32 and ((rms - 1).abs() < 1e-3).all(), rms
    with open(os.path.join(band_dir, "config.json")) as f:
        assert json.load(f)["band_rms"] == ckpt["band_scale"] == ckpt["band_rms"] and ckpt["band_power"] == 1.0
    code, out = run(*band_base, "--epochs", "2", "--auto-resume")
    assert code == 0 and "resumed after epoch 1" in out and "band scaling from latest.pt" in out, out
    code, out = run(*band_base[:-1], "--epochs", "3", "--auto-resume")
    assert code != 0 and "band scaling on" in out, out
    code, out = run(*base, "--epochs", "6", "--auto-resume", "--band-scale")
    assert code != 0 and "band scaling off" in out, out
    code, out = run("scripts/sample.py", "--run", band_dir, "--root", root, "--n-samples", "2", "--ddim-steps", "3", "--n-eval", "3")
    assert code == 0 and "band scaling on" in out and "generated 2 of 2" in out, out
    assert band_rows(out) == plain_bands and len(plain_bands) == 6, (band_rows(out), plain_bands)
    no_leak(out, "sample with band scaling")
    print(out)
    for extra in ([], ["--random-init"], ["--mode", "finetune", "--epochs", "1"]):
        code, out = run("scripts/downstream.py", "--run", band_dir, "--root", root, "--task", "male", "--workers", "0", "--n-boot", "50", *extra)
        assert code == 0 and "band scaling on" in out and "test auroc" in out, out
        no_leak(out, "downstream with band scaling")
    with open(os.path.join(band_dir, "downstream", "male_finetune_t0", "results.json")) as f:
        assert json.load(f)["band_scale"] is True

    sqrt_dir = os.path.join(tmp, "run_band_sqrt")
    sqrt_base = band_base[:4] + [sqrt_dir] + band_base[5:]
    code, out = run(*sqrt_base, "0.5", "--epochs", "1")
    assert code == 0 and "band scaling measured on" in out and "power 0.5" in out, out
    no_leak(out, "train with square-root band scaling")
    half = torch.load(os.path.join(sqrt_dir, "latest.pt"), map_location="cpu", weights_only=False)
    assert half["band_power"] == 0.5 and half["band_rms"] == ckpt["band_scale"], (half["band_rms"], ckpt["band_scale"])
    assert (torch.tensor(half["band_scale"]) - torch.tensor(ckpt["band_scale"]).sqrt()).abs().max() < 1e-6
    with open(os.path.join(sqrt_dir, "config.json")) as f:
        assert json.load(f)["band_rms"] == half["band_rms"]
    code, out = run(*sqrt_base, "0.5", "--epochs", "2", "--auto-resume")
    assert code == 0 and "resumed after epoch 1" in out and "band scaling from latest.pt, power 0.5" in out, out
    code, out = run(*sqrt_base, "--epochs", "3", "--auto-resume")
    assert code != 0 and "band scaling on, power 0.5" in out, out
    code, out = run(*sqrt_base, "2", "--epochs", "3", "--auto-resume")
    assert code != 0 and "power from 0 to 1" in out, out
    code, out = run("scripts/sample.py", "--run", sqrt_dir, "--root", root, "--n-samples", "2", "--ddim-steps", "3", "--n-eval", "3")
    assert code == 0 and "band scaling on" in out and band_rows(out) == plain_bands, out
    no_leak(out, "sample with square-root band scaling")

    lab_env = dict(ENV, OCT_ROOT=root, PATH=os.path.dirname(sys.executable) + os.pathsep + ENV.get("PATH", ""), TASKS="male",
                   SAMPLE_ARGS="--n-samples 1 --ddim-steps 2 --n-eval 2", DOWN_ARGS="--workers 0 --n-boot 50 --epochs 1")
    p = subprocess.run(["bash", "jobs/lab_evaluate.sh", run_dir, slo_dir], cwd=REPO, env=lab_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    out = p.stdout
    assert p.returncode == 0 and "evaluation finished, failed steps: 0" in out and out.count("test auroc") == 8 and out.count("generated 1 of 1") == 2, out
    no_leak(out, "lab evaluation")
    done = sorted(os.listdir(os.path.join(slo_dir, "downstream")))
    assert {"male_probe_t0", "male_probe_random_t0", "male_finetune_t0", "male_finetune_random_t0"} <= set(done), done
    p = subprocess.run(["bash", "jobs/lab_label_fraction.sh", run_dir], cwd=REPO, env=dict(lab_env, FRACTIONS="0.5 1", SEEDS="1", DOWN_ARGS="--workers 0 --n-boot 50"), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    out = p.stdout
    assert p.returncode == 0 and "label-fraction runs finished, failed steps: 0" in out and out.count("test auroc") == 4 and "fraction 0.5 subset 1" in out, out
    no_leak(out, "label-fraction runs")

    bad = os.path.join(root, "oct_images", "9.9.9.FAKEUID3", "b5.png")
    with open(bad, "wb") as f:
        f.write(b"not a png")
    code, out = run("scripts/train.py", "--root", root, "--out", os.path.join(tmp, "run2"), "--target", "32", "64", "64", "--batch-size", "2", "--workers", "0", "--epochs", "1", "--val-fraction", "0.25", "--test-fraction", "0")
    assert code != 0 and "failed: UnidentifiedImageError" in out, out
    no_leak(out, "train with a broken file")


def test_slo_parts(root):
    x = torch.rand(2, 32, 16, 32, 32) * 2 - 1
    t = torch.tensor([3, 700])
    slo = torch.rand(2, 64, 64) * 2 - 1
    base = mo.build_model(32, 64).eval()
    with torch.no_grad():
        for prm in base.parameters():
            if (prm == 0).all():
                prm.normal_(0, 0.02)
    wide = mo.build_model(32, 64, xattn=(0, 1, 2)).eval()
    loaded = wide.load_state_dict(base.state_dict(), strict=False)
    assert not loaded.unexpected_keys and loaded.missing_keys and all(k.startswith("xattn.") for k in loaded.missing_keys), loaded
    with torch.no_grad():
        start = mo.predict(base, x, t, slo)
        assert start.abs().mean() > 1e-3 and torch.allclose(start, mo.predict(wide, x, t, slo), atol=1e-5), "zero-initialised cross-attention must start as the SLO model without it"
        for name, prm in wide.xattn.named_parameters():
            if ".norm." not in name:
                prm.normal_(0, 0.5)
        own, swapped = mo.predict(wide, x, t, slo), mo.predict(wide, x, t, slo.flip(0))
        enc = mo.Encoder(wide, slo_features=True).eval()
        feats = enc(x, t, slo)
        feats_swapped = enc(x, t, slo.flip(0))
    assert not torch.allclose(own, swapped), "the added cross-attention must carry the SLO"
    assert enc.xattn is wide.xattn and feats.shape == (2, 560 + 160) == (2, enc.dim), feats.shape
    assert not torch.allclose(feats[:, :560], feats_swapped[:, :560]) and not torch.allclose(feats[:, 560:], feats_swapped[:, 560:])
    big = mo.build_model(32, 64, tokens=16, dim=96)
    assert big.slo(slo).shape == (2, 256, 96)
    assert fails(lambda: mo.build_model(32, 0, xattn=(1,))).startswith("the SLO adapter and SLO cross-attention need")
    assert fails(lambda: mo.build_model(32, 64, xattn=(5,))).startswith("--slo-xattn levels run from 0 to 4")
    assert mo.options({})["xattn"] == () and mo.options({})["tokens"] == 8 and mo.options({})["slo_slice"] is None

    h = torch.arange(4, dtype=torch.float32).view(1, 1, 4, 1, 1)
    assert mo.pool(h, ((0, 12), (12, 13)), 16).tolist() == [[1.0, 3.0]]
    assert mo.pool(torch.arange(16, dtype=torch.float32).view(1, 1, 16, 1, 1), ((0, 12), (12, 13)), 16).tolist() == [[5.5, 12.0]]
    axes = wv.DESIGNS["w2z1"]
    assert mo.joint_parts({"keep_slices": 24, "slo_slice": "raster"}, axes) == ((0, 12), (12, 13))
    assert mo.joint_parts({"keep_slices": 24}, axes) is None
    plain = mo.build_model(32).eval()
    with torch.no_grad():
        split = mo.Encoder(plain, zparts=((0, 12), (12, 13))).eval()(x, t)
    assert split.shape == (2, 1120), split.shape

    stepped = mo.build_model(32, part=(12, 13)).eval()
    with torch.no_grad():
        for prm in stepped.unet.parameters():
            if (prm == 0).all():
                prm.normal_(0, 0.02)
    assert isinstance(stepped, mo.PartNoiseUNet) and stepped.unet.conv_in.conv.in_channels == 33
    level = stepped.step.level(torch.tensor([-1, 0, 999]))
    assert level[0] == 0 and 0 < level[1] < 0.02 and level[2] > 0.99, level
    m, inject = stepped.step.inputs(x, t, torch.tensor([-1, 999]))
    assert m.shape == (2, 33, 16, 32, 32) and torch.equal(m[:, :32], x)
    assert (m[0, 32, 12] == 0).all() and torch.allclose(m[1, 32, 12], level[2].expand(32, 32)) and torch.allclose(m[0, 32, :12], stepped.step.level(t[:1]).expand(12, 32, 32))
    emb = torch.randn(2, stepped.unet.time_embed[-1].out_features)
    assert torch.equal(inject.time(emb), emb), "the SLO step embedding must start at zero"
    with torch.no_grad():
        shared = mo.predict(stepped, x, t)
        assert torch.allclose(shared, mo.predict(stepped, x, t, None, t)), "no SLO step means the OCT's step"
        assert not torch.allclose(shared, mo.predict(stepped, x, t, None, torch.tensor([-1, -1])))
        for prm in stepped.step.mlp.parameters():
            prm.normal_(0, 0.5)
        assert not torch.allclose(inject.time(emb), emb)
        sfeat = mo.Encoder(stepped, zparts=((0, 12), (12, 13))).eval()(x, t)
    assert sfeat.shape == (2, 1120) and torch.isfinite(sfeat).all()
    assert fails(lambda: mo.build_model(32, 64, part=(12, 13))).startswith("an own noise step for the SLO slice needs the joint model")
    assert mo.model_args({"design": "w2z1", "keep_slices": 24, "slo_slice": "raster", "slo_own_step": True})["part"] == (12, 13)
    assert mo.model_args({"design": "w2z1", "keep_slices": 24, "slo_slice": "raster"})["part"] is None
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_script", os.path.join(REPO, "scripts", "train.py"))
    tr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tr)
    torch.manual_seed(1)
    tt = torch.randint(0, 1000, (20000,))
    ts = tr.slo_steps(tt, 0.25, 0.25)
    clean, same = (ts == -1).float().mean().item(), (ts == tt).float().mean().item()
    assert abs(clean - 0.25) < 0.02 and abs(same - 0.25) < 0.02 and ts.min() >= -1 and ts.max() <= 999, (clean, same)
    sched = DDPMScheduler(num_train_timesteps=1000, schedule="linear_beta", prediction_type="sample")
    x0, noise = torch.rand(2, 32, 16, 4, 4), torch.randn(2, 32, 16, 4, 4)
    noisy = sched.add_noise(original_samples=x0, noise=noise, timesteps=t)
    given = tr.with_slo(noisy, x0[:, :, 12:13], noise[:, :, 12:13], torch.tensor([-1, 500]), sched, 12, 13)
    assert torch.equal(given[0, :, 12:13], x0[0, :, 12:13]) and torch.equal(given[:, :, :12], noisy[:, :, :12]) and torch.equal(given[:, :, 13:], noisy[:, :, 13:])
    assert torch.allclose(given[1:, :, 12:13], sched.add_noise(original_samples=x0[1:, :, 12:13], noise=noise[1:, :, 12:13], timesteps=torch.tensor([500])))

    folder = sorted(glob.glob(os.path.join(root, "oct_images", "*FAKEUID3")))[0]
    vol, full = vo.assemble(folder, 25, (32, 64, 64), keep=24, slo_slice="raster")
    assert vol.shape == (24, DEPTH, WIDTH) and full.shape == (32, 64, 64) and full.dtype == np.float32
    assert np.array_equal(vo.oct_region(full, vol.shape, True), vol) and np.array_equal(full[24], full[25]) and (full[26:] == -1).all()
    assert np.array_equal(vo.slo_region(full, vol.shape), full[24]) and full[24].min() >= -1 and full[24].max() <= 1
    c = wv.forward(torch.from_numpy(full)[None, None], axes)
    alone = wv.forward(torch.from_numpy(full[24:26].copy())[None, None], axes)
    assert torch.allclose(c[:, :, 12:13], alone, atol=1e-6), "the SLO must sit in its own wavelet slice"
    lines = al.read_lines(folder)[:24]
    rows, cols = vo.raster_positions(lines, 64, 64)
    for k in (0, 11, 23):
        r = vo.raster_row(k, 24, 64)
        assert abs(rows[r, 0] - lines[k, 0]) <= 0.5 and abs(cols[r, 0] - lines[k, 1]) < 1e-6 and abs(cols[r, -1] - lines[k, 3]) < 1e-6
    whole = vo.load_slo_slice(folder, 25, (64, 64), 24, "full")
    assert whole.shape == (64, 64) and not np.array_equal(whole, full[24])
    _, centred = vo.assemble(folder, 25, (32, 64, 64), keep=24)
    assert np.array_equal(vo.oct_region(centred, vol.shape, False), vol)
    assert fails(lambda: vo.joint_volume(np.zeros((25, DEPTH, WIDTH), np.float32), full[24], (32, 64, 64))).startswith("the joint volume needs")
    assert fails(lambda: vo.joint_volume(np.zeros((24, DEPTH, WIDTH), np.float32), full[24], (24, 64, 64))).startswith("the joint volume needs")


def test_slo_runs(root, tmp):
    common = ["--root", root, "--batch-size", "2", "--workers", "0", "--max-steps", "2", "--val-fraction", "0.2", "--test-fraction", "0.2"]
    wide = os.path.join(tmp, "run_xattn")
    wide_base = ["scripts/train.py", *common, "--out", wide, "--target", "24", "64", "64", "--keep-slices", "24", "--z-down", "0", "0", "1", "1", "--slo-size", "64", "--slo-tokens", "4", "--slo-dim", "96", "--slo-xattn", "2", "0", "1", "--band-scale", "0.5"]
    code, out = run(*wide_base, "--epochs", "1")
    assert code == 0 and "SLO 64 px, 16 tokens of 96, cross-attention at levels 0 1 2" in out and "(own SLO)  other SLO" in out, out
    no_leak(out, "train with SLO cross-attention")
    print(out)
    ckpt = torch.load(os.path.join(wide, "latest.pt"), map_location="cpu", weights_only=False)
    assert ckpt["slo_xattn"] == [0, 1, 2] and ckpt["slo_tokens"] == 4 and ckpt["slo_dim"] == 96 and ckpt["slo_slice"] is None
    assert any(k.startswith("xattn.0.") for k in ckpt["model"]) and all(torch.isfinite(v).all() for v in ckpt["model"].values())
    code, out = run(*wide_base, "--epochs", "2", "--auto-resume")
    assert code == 0 and "resumed after epoch 1" in out, out
    code, out = run(*wide_base[:-5], "1", "--band-scale", "0.5", "--epochs", "3", "--auto-resume")
    assert code != 0 and "cross-attention levels 0 1 2" in out, out
    code, out = run("scripts/train.py", *common, "--out", os.path.join(tmp, "run_xattn_no_slo"), "--target", "32", "64", "64", "--slo-xattn", "1", "--epochs", "1")
    assert code != 0 and "--slo-xattn needs --slo-size" in out, out
    code, out = run("scripts/sample.py", "--run", wide, "--root", root, "--n-samples", "1", "--ddim-steps", "2", "--n-eval", "2")
    assert code == 0 and "cross-attention at levels 0 1 2" in out and "ratio, own SLO" in out and "OCT generated from a given real SLO, 1 scans" in out and "from SLO   own SLO" in out, out
    no_leak(out, "sample with SLO cross-attention")
    for extra in ([], ["--slo-features"], ["--slo-features", "--random-init"], ["--slo-features", "--mode", "finetune", "--epochs", "1"]):
        code, out = run("scripts/downstream.py", "--run", wide, "--root", root, "--task", "righteye", "--workers", "0", "--n-boot", "50", *extra)
        size = 720 if "--slo-features" in extra else 560
        assert code == 0 and "cross-attention at levels 0 1 2" in out and "feature size %d" % size in out and "test auroc" in out, out
        no_leak(out, "downstream with SLO cross-attention")
    assert os.path.exists(os.path.join(wide, "downstream", "righteye_probe_slofeat_t0", "results.json"))
    code, out = run("scripts/downstream.py", "--run", wide, "--root", root, "--task", "age", "--kind", "regression", "--slo-features", "--workers", "0", "--n-boot", "50")
    assert code == 0 and "test mae" in out, out
    report = os.path.join(tmp, "report_xattn")
    code, out = run("scripts/scan_report.py", "--run", wide, "--root", root, "--out", report, "--max-scans", "2", "--steps", "50", "--group", "--n-worst", "1")
    assert code == 0 and "scored 2 scans, could not read 0" in out, out
    code, out = run("scripts/probe_memory.py", "--target", "24", "64", "64", "--steps", "1", "--max-batch", "1", "--designs", "w2z1", "--slo-size", "64", "--slo-tokens", "4", "--slo-xattn", "0", "1", "2", "--z-down", "0", "0", "1", "1")
    assert code == 0 and "16 tokens of 160, cross-attention at levels 0 1 2" in out and "batch  1: OK" in out, out

    joint = os.path.join(tmp, "run_joint")
    joint_base = ["scripts/train.py", *common, "--out", joint, "--target", "32", "64", "64", "--keep-slices", "24", "--z-down", "0", "0", "1", "1", "--slo-slice", "raster", "--band-scale", "0.5"]
    code, out = run(*joint_base, "--epochs", "1")
    assert code == 0 and "SLO joint, own slice after the B-scans (raster)" in out and "wavelet grid (16, 16, 16)" in out and "OCT with own SLO slice" in out and "with another scan's" in out, out
    no_leak(out, "train the joint model")
    print(out)
    ckpt = torch.load(os.path.join(joint, "latest.pt"), map_location="cpu", weights_only=False)
    assert ckpt["slo_slice"] == "raster" and ckpt["slo_size"] == 0 and not any(k.startswith("slo.") for k in ckpt["model"])
    code, out = run(*[a for a in joint_base if a not in ("--slo-slice", "raster")], "--epochs", "2", "--auto-resume")
    assert code != 0 and "SLO slice raster" in out, out
    code, out = run(*joint_base[:-2], "--slo-size", "64", "--out", os.path.join(tmp, "run_joint_slo"), "--epochs", "1")
    assert code != 0 and "leave out --slo-size" in out, out
    code, out = run("scripts/train.py", *common, "--out", os.path.join(tmp, "run_joint_odd"), "--target", "32", "64", "64", "--slo-slice", "raster", "--epochs", "1")
    assert code != 0 and "even number of B-scans" in out, out
    code, out = run("scripts/train.py", *common, "--out", os.path.join(tmp, "run_joint_full"), "--target", "24", "64", "64", "--keep-slices", "24", "--z-down", "0", "0", "1", "1", "--slo-slice", "full", "--epochs", "1")
    assert code != 0 and "two free slices" in out, out
    code, out = run("scripts/sample.py", "--run", joint, "--root", root, "--n-samples", "2", "--ddim-steps", "2", "--n-eval", "2")
    assert code == 0 and "joint slice (raster)" in out and "SLO slice, intensity" in out and "OCT en-face view against the SLO" in out and "images written: 8" in out, out
    assert "similarity between different scans" in out and "generated  en-face views" in out and "SLOs" in out and "given real SLO" not in out, out
    no_leak(out, "sample from the joint model")
    print(out)
    with Image.open(os.path.join(joint, "samples", "sample_000_slo.png")) as im:
        assert im.size == (64, 64), im.size
    with Image.open(os.path.join(joint, "samples", "sample_000.png")) as im:
        assert im.size == (WIDTH + 24 * 8 + WIDTH + 20, DEPTH), im.size
    for extra, size in (([], 1120), (["--joint-pool", "oct"], 560), (["--joint-pool", "slo", "--random-init"], 560), (["--mode", "finetune", "--epochs", "1"], 1120)):
        code, out = run("scripts/downstream.py", "--run", joint, "--root", root, "--task", "male", "--workers", "0", "--n-boot", "50", *extra)
        assert code == 0 and "joint slice (raster)" in out and "feature size %d" % size in out and "test auroc" in out, out
        no_leak(out, "downstream on the joint model")
    code, out = run("scripts/downstream.py", "--run", joint, "--root", root, "--task", "male", "--slo-features", "--workers", "0")
    assert code != 0 and "--slo-features needs a model with an SLO encoder" in out, out
    report = os.path.join(tmp, "report_joint")
    code, out = run("scripts/scan_report.py", "--run", joint, "--root", root, "--out", report, "--max-scans", "2", "--steps", "50", "--group", "--n-worst", "1")
    assert code == 0 and "PSNR and SSIM on the B-scans only" in out and "scored 2 scans, could not read 0" in out, out
    no_leak(out, "scan report on the joint model")

    step = os.path.join(tmp, "run_joint_step")
    step_base = [a if a != joint else step for a in joint_base] + ["--slo-own-step"]
    code, out = run(*step_base, "--epochs", "1")
    assert code == 0 and "own noise step for the SLO (clean 0.25, same step 0.25, independent 0.5)" in out and "OCT given a clean SLO: own" in out and "SLO all noise" in out, out
    no_leak(out, "train the joint model with an own SLO step")
    print(out)
    ckpt = torch.load(os.path.join(step, "latest.pt"), map_location="cpu", weights_only=False)
    assert ckpt["slo_own_step"] and ckpt["slo_clean"] == 0.25 and any(k.startswith("step.mlp.") for k in ckpt["model"]) and all(torch.isfinite(v).all() for v in ckpt["model"].values())
    code, out = run(*step_base, "--epochs", "2", "--auto-resume")
    assert code == 0 and "resumed after epoch 1" in out, out
    code, out = run(*step_base[:-1], "--epochs", "3", "--auto-resume")
    assert code != 0 and "trained with an own noise step" in out, out
    code, out = run(*[a for a in step_base if a not in ("--slo-slice", "raster")][:-1] + ["--out", os.path.join(tmp, "run_step_noslice"), "--slo-own-step", "--epochs", "1"])
    assert code != 0 and "--slo-own-step needs --slo-slice" in out, out
    code, out = run(*step_base, "--out", os.path.join(tmp, "run_step_shares"), "--slo-clean", "0.7", "--slo-same", "0.5", "--epochs", "1")
    assert code != 0 and "add up to at most 1" in out, out
    code, out = run("scripts/sample.py", "--run", step, "--root", root, "--n-samples", "2", "--ddim-steps", "2", "--n-eval", "2", "--n-given", "2")
    assert code == 0 and "joint slice (raster), own noise step" in out and "B-scans only" in out and "ratio, own SLO clean" in out and "ratio, SLO all noise" in out, out
    assert "OCT generated from a given real SLO, 2 scans" in out and "from SLO   own SLO" in out and "from SLO   en-face views" in out and "images written: 10" in out, out
    no_leak(out, "sample from the joint model with an own SLO step")
    print(out)
    with Image.open(os.path.join(step, "samples", "given_001.png")) as im:
        assert im.size == (WIDTH + 24 * 8 + WIDTH + 20, DEPTH), im.size
    for extra, size in (([], 1120), (["--joint-pool", "oct", "--random-init"], 560), (["--mode", "finetune", "--epochs", "1"], 1120)):
        code, out = run("scripts/downstream.py", "--run", step, "--root", root, "--task", "male", "--workers", "0", "--n-boot", "50", *extra)
        assert code == 0 and "feature size %d" % size in out and "test auroc" in out, out
        no_leak(out, "downstream on the joint model with an own SLO step")
    code, out = run("scripts/scan_report.py", "--run", step, "--root", root, "--out", os.path.join(tmp, "report_step"), "--max-scans", "2", "--steps", "50", "--n-worst", "1")
    assert code == 0 and "scored 2 scans, could not read 0" in out, out


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
            ("slope vs sudden jump measure", test_split_steps, ()),
            ("bright layer measure", test_layer_depths, ()),
            ("SLO alignment measure", test_alignment, ()),
            ("cohort and split", test_cohort, (root,)),
            ("models, encoder and metrics", test_models, ()),
            ("SLO adapter and slice-axis option", test_adapter_parts, (root,)),
            ("PSNR, SSIM and groups", test_report_measures, ()),
            ("scan report", test_report, (root, tmp)),
            ("SLO adapter and 24-slice runs", test_adapter_runs, (root, tmp)),
            ("SLO cross-attention levels and joint OCT+SLO volume", test_slo_parts, (root,)),
            ("SLO cross-attention and joint model runs", test_slo_runs, (root, tmp)),
            ("scripts end to end", test_scripts, (root, tmp)),
        ):
            print("--- %s" % name, flush=True)
            fn(*args)
            print("    OK", flush=True)
        print("ALL TESTS PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


main()
