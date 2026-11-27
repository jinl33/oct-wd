import argparse
import json
import os
import sys

import numpy as np
import torch

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oct_wd import cohort as co
from oct_wd import wavelet as wv
from oct_wd.model import band_scale, load_checkpoint, options, to_device
from oct_wd.volumes import OctVolumes


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, nargs="+", help="pretraining run folders trained with --slo-align, each with latest.pt, config.json and split.json")
    ap.add_argument("--root", default=None)
    ap.add_argument("--part", default="test", choices=("val", "test"))
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--n-perm", type=int, default=10000, help="random pairings for the p-value of the mean rank")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


@torch.no_grad()
def embeddings(model, loader, axes, band, device, use_amp):
    out = {k: [] for k in ("o", "v", "valid", "go", "gs")}
    for vol, cond in loader:
        x0 = wv.scaled(wv.forward(vol.to(device), axes), band)
        with torch.amp.autocast("cuda", enabled=use_amp):
            parts = model.match(x0, to_device(cond, device))
        for k, a in zip(out, parts):
            out[k].append(a.cpu())
    return {k: torch.cat(v) for k, v in out.items()}


def place_matrix(o, v, valid):
    n = len(o)
    sim, shared = torch.empty(n, n), torch.empty(n, n)
    for i in range(n):
        both = (valid[i][None] & valid).float()
        cos = torch.einsum("cd,jcd->jc", o[i], v)
        shared[i] = both.sum(1)
        sim[i] = (cos * both).sum(1) / shared[i].clamp_min(1)
    return sim, int((shared == 0).sum())


def other_place(o, v, valid):
    both = (valid & valid.flip(1)).float()
    cos = (o * v.flip(1)).sum(-1)
    return ((cos * both).sum(1) / both.sum(1).clamp_min(1)).mean().item()


def ranks(sim):
    order = sim.argsort(dim=1, descending=True)
    r = torch.empty_like(order)
    r.scatter_(1, order, torch.arange(1, sim.shape[1] + 1).expand_as(order).contiguous())
    return r


def summary(sim, n_perm, gen):
    n = len(sim)
    r = ranks(sim)
    own = r.diagonal().float()
    rows = torch.arange(n)[None].expand(n_perm, n)
    pairings = torch.stack([torch.randperm(n, generator=gen) for _ in range(n_perm)])
    random_means = r[rows, pairings].float().mean(1)
    p = (1 + (random_means <= own.mean()).sum().item()) / (1 + n_perm)
    return dict(top1=int((own == 1).sum()), top5=int((own <= 5).sum()), median_rank=float(np.median(own.numpy())), mean_rank=own.mean().item(), p=p)


def main():
    args = parse()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    gen = torch.Generator().manual_seed(args.seed)
    for run in args.run:
        name = os.path.basename(os.path.normpath(run))
        model, ckpt, axes = load_checkpoint(os.path.join(run, "latest.pt"), device)
        saved = options(ckpt)
        if not saved["align"]:
            raise SystemExit("%s is not an aligned model (trained without --slo-align)" % name)
        model.eval()
        with open(os.path.join(run, "config.json")) as f:
            config = json.load(f)
        with open(os.path.join(run, "split.json")) as f:
            members = set(json.load(f)[args.part])
        scans, _ = co.list_scans(args.root or co.default_root(), config["n_slices"])
        chosen = [s for s in scans if s["patient"] in members]
        if len(chosen) < 2:
            raise SystemExit("%s: fewer than 2 %s scans found" % (name, args.part))
        data = OctVolumes(chosen, tuple(ckpt["target"]), ckpt["slo_size"], adapter=True, keep=saved["keep"], slo_slice=saved["slo_slice"])
        loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        e = embeddings(model, loader, axes, band_scale(ckpt), device, use_amp)
        n = len(chosen)
        place, empty = place_matrix(e["o"], e["v"], e["valid"])
        whole = e["go"] @ e["gs"].t()
        off = ~torch.eye(n, dtype=torch.bool)
        result = dict(run=name, checkpoint_epoch=ckpt["epoch"], align_loss=saved["align_loss"], part=args.part, scans=n, cells=int(e["o"].shape[1]), n_perm=args.n_perm,
                      place_own=place.diagonal().mean().item(), place_own_other_place=other_place(e["o"], e["v"], e["valid"]), place_other=place[off].mean().item(),
                      whole_own=whole.diagonal().mean().item(), whole_other=whole[off].mean().item(), pairs_without_shared_cells=empty, retrieval={})
        print("%s: checkpoint epoch %d, aligned encoder (%s), %s scans: %d, cells per scan: %d" % (name, ckpt["epoch"], saved["align_loss"], args.part, n, result["cells"]))
        print("same-place cosine, mean over scans: own SLO %.3f  own SLO at the place turned 180 degrees %.3f  other scans' SLO %.3f" % (result["place_own"], result["place_own_other_place"], result["place_other"]))
        print("whole-scan cosine: own SLO %.3f  other scans' SLO %.3f" % (result["whole_own"], result["whole_other"]))
        if empty:
            print("pairs with no place in common (scored 0): %d" % empty)
        print("retrieval among %d %s scans (expected by chance: 1 scan ranked first, %d in the top 5, median rank %.1f; p from %d random pairings, one-sided on the mean rank)" % (n, args.part, min(5, n), (n + 1) / 2, args.n_perm))
        print("  %-11s %-11s %-6s %-6s %-12s %-10s %s" % ("measure", "direction", "top-1", "top-5", "median rank", "mean rank", "p"))
        for measure, sim in (("same place", place), ("whole scan", whole)):
            for direction, m in (("OCT to SLO", sim), ("SLO to OCT", sim.t())):
                s = summary(m, args.n_perm, gen)
                result["retrieval"]["%s, %s" % (measure, direction)] = s
                print("  %-11s %-11s %-6d %-6d %-12.1f %-10.1f %.4f" % (measure, direction, s["top1"], s["top5"], s["median_rank"], s["mean_rank"], s["p"]))
        out = os.path.join(run, "retrieval")
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "retrieval_%s.json" % args.part), "w") as f:
            json.dump(result, f, indent=1)
        print("results written to retrieval/retrieval_%s.json" % args.part)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
