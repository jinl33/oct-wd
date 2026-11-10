import numpy as np
from scipy.stats import rankdata

MAIN = {"binary": "auroc", "regression": "mae"}


def auroc(y, s):
    y = np.asarray(y) > 0.5
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def scores(kind, y, pred):
    y, pred = np.asarray(y, dtype=np.float64), np.asarray(pred, dtype=np.float64)
    if kind == "binary":
        return {"auroc": auroc(y, pred), "accuracy": float(((pred > 0) == (y > 0.5)).mean())}
    r = float(np.corrcoef(y, pred)[0, 1]) if len(y) > 1 and y.std() > 0 and pred.std() > 0 else float("nan")
    return {"mae": float(np.abs(y - pred).mean()), "pearson_r": r}


def better(kind, a, b):
    if not np.isfinite(a):
        return False
    if not np.isfinite(b):
        return True
    return a > b if kind == "binary" else a < b


def bootstrap(kind, y, pred, n=1000, seed=0):
    y, pred = np.asarray(y), np.asarray(pred)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        v = scores(kind, y[i], pred[i])[MAIN[kind]]
        if np.isfinite(v):
            values.append(v)
    if not values:
        return float("nan"), float("nan")
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))
