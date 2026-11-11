import argparse
import copy
import json
import os
import random
import sys
import time

import numpy as np
import torch

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generative.networks.schedulers import DDPMScheduler

from oct_wd import cohort as co
from oct_wd import evaluate as ev
from oct_wd import wavelet as wv
from oct_wd.model import Encoder, band_scale, build_model, load_checkpoint
from oct_wd.volumes import OctVolumes

WEIGHT_DECAYS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="pretraining run folder with latest.pt, config.json and a split.json that has a test list")
    ap.add_argument("--root", default=None)
    ap.add_argument("--task", required=True, help="cohort column to predict, for example male or righteye")
    ap.add_argument("--kind", default="binary", choices=("binary", "regression"))
    ap.add_argument("--mode", default="probe", choices=("probe", "finetune"))
    ap.add_argument("--random-init", action="store_true", help="same architecture without pretrained weights, as a control")
    ap.add_argument("--t", type=int, default=0, help="noise step the encoder sees")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--label-fraction", type=float, default=1.0, help="share of training patients whose labels the head sees; finetune epochs scale up so the number of training steps stays about the same")
    ap.add_argument("--subset-seed", type=int, default=0, help="which random subset of training patients, when --label-fraction is below 1")
    ap.add_argument("--out", default=None)
    return ap.parse_args()


class Labelled(torch.utils.data.Dataset):
    def __init__(self, volumes, y):
        self.volumes = volumes
        self.y = y

    def __len__(self):
        return len(self.volumes)

    def __getitem__(self, i):
        return self.volumes[i], self.y[i]


def labelled(scans, task, kind):
    pairs = [(s, co.label(s, task)) for s in scans]
    kept = [(s, y) for s, y in pairs if y is not None]
    y = np.array([v for _, v in kept], dtype=np.float32)
    if kind == "binary" and not np.isin(y, (0.0, 1.0)).all():
        raise SystemExit("column %s is not 0/1: %d values outside" % (task, int((~np.isin(y, (0.0, 1.0))).sum())))
    return [s for s, _ in kept], y, len(pairs) - len(kept)


def label_subset(scans, y, fraction, seed):
    patients = sorted({s["patient"] for s in scans})
    random.Random(seed).shuffle(patients)
    keep = set(patients[: max(1, round(len(patients) * fraction))])
    idx = [i for i, s in enumerate(scans) if s["patient"] in keep]
    return [scans[i] for i in idx], y[idx], len(keep), len(patients)


def encoder_input(batch, axes, t, sched, gen, device, slo_size, band):
    vol, slo = (batch, None) if not slo_size else (batch[0], batch[1].to(device))
    x = wv.scaled(wv.forward(vol.to(device), axes), band)
    tt = torch.full((x.shape[0],), t, device=device, dtype=torch.long)
    if t > 0:
        noise = torch.randn(x.shape, generator=gen).to(device)
        x = sched.add_noise(original_samples=x, noise=noise, timesteps=tt)
    return x, tt, slo


def loss_fn(kind, out, y):
    if kind == "binary":
        return torch.nn.functional.binary_cross_entropy_with_logits(out, y)
    return torch.nn.functional.mse_loss(out, y)


@torch.no_grad()
def features(enc, scans, args, axes, target, slo_size, sched, device, use_amp, band):
    enc.eval()
    loader = torch.utils.data.DataLoader(OctVolumes(scans, target, slo_size), batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    out = []
    for batch in loader:
        x, tt, slo = encoder_input(batch, axes, args.t, sched, gen, device, slo_size, band)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out.append(enc(x, tt, slo).cpu().numpy())
    return np.concatenate(out)


def fit_probe(kind, xtr, ytr, xva, yva, seed, back):
    best = (float("nan"), None, None)
    X, Y = torch.from_numpy(xtr), torch.from_numpy(ytr)
    for wd in WEIGHT_DECAYS:
        torch.manual_seed(seed)
        head = torch.nn.Linear(X.shape[1], 1)
        opt = torch.optim.Adam(head.parameters(), lr=1e-2)
        for _ in range(500):
            opt.zero_grad()
            loss = loss_fn(kind, head(X)[:, 0], Y) + wd * head.weight.pow(2).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred = back(head(torch.from_numpy(xva))[:, 0].numpy())
        value = ev.scores(kind, yva, pred)[ev.MAIN[kind]]
        if best[1] is None or ev.better(kind, value, best[0]):
            best = (value, wd, head)
    return best


@torch.no_grad()
def predict_all(enc, head, loader, args, axes, slo_size, sched, device, use_amp, band):
    enc.eval()
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    out = []
    for batch, _ in loader:
        x, tt, slo = encoder_input(batch, axes, args.t, sched, gen, device, slo_size, band)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out.append(head(enc(x, tt, slo))[:, 0].float().cpu().numpy())
    return np.concatenate(out)


def main():
    args = parse()
    if not 0 < args.label_fraction <= 1:
        raise SystemExit("--label-fraction must be above 0 and at most 1")
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    with open(os.path.join(args.run, "config.json")) as f:
        config = json.load(f)
    with open(os.path.join(args.run, "split.json")) as f:
        split = json.load(f)
    if "test" not in split:
        raise SystemExit("this run has no test split; pretrain with --test-fraction above 0")
    model, ckpt, axes = load_checkpoint(os.path.join(args.run, "latest.pt"), device)
    target, slo_size, band = tuple(ckpt["target"]), ckpt.get("slo_size", 0), band_scale(ckpt)
    if args.random_init:
        torch.manual_seed(args.seed)
        model = build_model(wv.channels(axes), slo_size).to(device)
    enc = Encoder(model).to(device)
    del model
    name = "%s_%s%s_t%d%s" % (args.task, args.mode, "_random" if args.random_init else "", args.t, "_f%g_s%d" % (args.label_fraction, args.subset_seed) if args.label_fraction < 1 else "")
    out_dir = args.out or os.path.join(args.run, "downstream", name)
    os.makedirs(out_dir, exist_ok=True)
    print("task %s (%s), %s, encoder %s, design %s, SLO %s, band scaling %s, noise step %d, feature size %d, device %s" % (
        args.task, args.kind, args.mode, "random init" if args.random_init else "pretrained epoch %d" % ckpt["epoch"],
        ckpt["design"], "%d px" % slo_size if slo_size else "not used", "on" if band is not None else "off", args.t, enc.dim, device))
    print("encoder parameters: %d (decoder removed)" % sum(p.numel() for p in enc.parameters()))

    scans, _ = co.list_scans(args.root or co.default_root(), config["n_slices"])
    parts, labels, report = {}, {}, {}
    for part in ("train", "val", "test"):
        members = set(split[part])
        parts[part], labels[part], missing = labelled([s for s in scans if s["patient"] in members], args.task, args.kind)
        if part == "train" and args.label_fraction < 1:
            parts[part], labels[part], kept, total = label_subset(parts[part], labels[part], args.label_fraction, args.subset_seed)
            print("label fraction %g, subset %d: %d of %d training patients" % (args.label_fraction, args.subset_seed, kept, total))
            if args.kind == "binary" and len(set(labels[part].tolist())) < 2:
                raise SystemExit("the training subset has one class only; use another --subset-seed")
        report[part] = {"n": len(parts[part]), "missing_label": missing}
        if args.kind == "binary":
            report[part]["positives"] = int(labels[part].sum())
        print("%-5s scans with a label: %d  missing label: %d%s" % (part, len(parts[part]), missing, "  positives: %d" % labels[part].sum() if args.kind == "binary" else ""))
        if not parts[part]:
            raise SystemExit("no labelled scans in %s" % part)

    mean, std = (0.0, 1.0) if args.kind == "binary" else (float(labels["train"].mean()), float(labels["train"].std() or 1.0))

    def back(p):
        return p if args.kind == "binary" else p * std + mean

    sched = DDPMScheduler(num_train_timesteps=1000, schedule="linear_beta", prediction_type="sample")
    started = time.time()
    result = dict(vars(args), checkpoint_epoch=ckpt["epoch"], design=ckpt["design"], slo_size=slo_size, band_scale=band is not None, splits=report)
    if args.mode == "probe":
        feats = {p: features(enc, parts[p], args, axes, target, slo_size, sched, device, use_amp, band) for p in parts}
        mu, sd = feats["train"].mean(axis=0), feats["train"].std(axis=0) + 1e-6
        feats = {p: ((v - mu) / sd).astype(np.float32) for p, v in feats.items()}
        ytr = (labels["train"] - mean) / std
        val_value, wd, head = fit_probe(args.kind, feats["train"], ytr.astype(np.float32), feats["val"], labels["val"], args.seed, back)
        with torch.no_grad():
            test_pred = back(head(torch.from_numpy(feats["test"]))[:, 0].numpy())
        result["chosen_weight_decay"] = wd
        torch.save({"head": head.state_dict(), "mean": mu, "std": sd, "label_mean": mean, "label_std": std}, os.path.join(out_dir, "probe.pt"))
    else:
        head = torch.nn.Sequential(torch.nn.LayerNorm(enc.dim), torch.nn.Linear(enc.dim, 1)).to(device)
        loaders = {}
        for p in parts:
            y = ((labels[p] - mean) / std).astype(np.float32)
            loaders[p] = torch.utils.data.DataLoader(Labelled(OctVolumes(parts[p], target, slo_size), y), batch_size=args.batch_size, shuffle=p == "train", drop_last=p == "train" and len(parts[p]) > args.batch_size, num_workers=args.workers)
        opt = torch.optim.AdamW([{"params": enc.parameters(), "lr": args.lr}, {"params": head.parameters(), "lr": args.head_lr}])
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        gen = torch.Generator(device="cpu").manual_seed(args.seed)
        val_value, best_epoch, best_state = float("nan"), 0, None
        every = 1 if args.label_fraction >= 1 else max(1, round(1 / args.label_fraction))
        epochs = args.epochs * every
        if every > 1:
            print("finetune for %d epochs, validating every %d, so the training steps match about %d epochs on all labels" % (epochs, every, args.epochs))
        total, steps = 0.0, 0
        for epoch in range(1, epochs + 1):
            enc.train()
            head.train()
            for batch, y in loaders["train"]:
                x, tt, slo = encoder_input(batch, axes, args.t, sched, gen, device, slo_size, band)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = head(enc(x, tt, slo))[:, 0].float()
                loss = loss_fn(args.kind, out, y.to(device))
                if not torch.isfinite(loss):
                    raise SystemExit("non-finite loss at epoch %d" % epoch)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                total += loss.item()
                steps += 1
            if epoch % every:
                continue
            value = ev.scores(args.kind, labels["val"], back(predict_all(enc, head, loaders["val"], args, axes, slo_size, sched, device, use_amp, band)))[ev.MAIN[args.kind]]
            print("epoch %d: train loss %.4f  val %s %.4f" % (epoch, total / max(steps, 1), ev.MAIN[args.kind], value), flush=True)
            total, steps = 0.0, 0
            if best_state is None or ev.better(args.kind, value, val_value):
                val_value, best_epoch = value, epoch
                best_state = (copy.deepcopy(enc.state_dict()), copy.deepcopy(head.state_dict()))
        enc.load_state_dict(best_state[0])
        head.load_state_dict(best_state[1])
        test_pred = back(predict_all(enc, head, loaders["test"], args, axes, slo_size, sched, device, use_amp, band))
        result.update(chosen_epoch=best_epoch, epochs_run=epochs, validate_every=every)
        torch.save({"encoder": best_state[0], "head": best_state[1], "label_mean": mean, "label_std": std}, os.path.join(out_dir, "finetuned.pt"))

    test = ev.scores(args.kind, labels["test"], test_pred)
    low, high = ev.bootstrap(args.kind, labels["test"], test_pred, args.n_boot, args.seed)
    result.update(val=val_value, test=test, test_ci95=[low, high], seconds=round(time.time() - started))
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(result, f, indent=1)
    key = ev.MAIN[args.kind]
    print("validation %s: %.4f" % (key, val_value))
    print("test %s: %.4f (95%% CI %.4f to %.4f)   %s" % (key, test[key], low, high, "   ".join("%s %.4f" % (k, v) for k, v in test.items() if k != key)))
    print("results written to %s" % os.path.basename(out_dir))


try:
    main()
except SystemExit:
    raise
except Exception as e:
    print("stopped early:", type(e).__name__)
    sys.exit(1)
