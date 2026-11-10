"""Dataset loading and client splits. The whole dataset sits on the device to avoid DataLoader overhead."""
import os

import numpy as np
import torch
from torchvision import datasets
from pathlib import Path

ROOT = os.environ.get("SPPFD_DATA", str(Path(__file__).resolve().parents[1] / "data"))
_CACHE = {}


def load(name, dev):
    key = (name, str(dev))
    if key in _CACHE:
        return _CACHE[key]
    if name == "cifar10":
        tr, te = datasets.CIFAR10(ROOT, True), datasets.CIFAR10(ROOT, False)
        X = (torch.tensor(tr.data).float() / 255).permute(0, 3, 1, 2)
        y = torch.tensor(tr.targets)
        Xt = (torch.tensor(te.data).float() / 255).permute(0, 3, 1, 2)
        yt = torch.tensor(te.targets)
        m = X.mean((0, 2, 3), keepdim=True)
        s = X.std((0, 2, 3), keepdim=True)
        X, Xt = (X - m) / s, (Xt - m) / s
    else:
        tr, te = datasets.FashionMNIST(ROOT, True), datasets.FashionMNIST(ROOT, False)
        X = (tr.data.float() / 255).unsqueeze(1)
        y = tr.targets
        Xt = (te.data.float() / 255).unsqueeze(1)
        yt = te.targets
    out = (X.to(dev), y.to(dev), Xt.to(dev), yt.to(dev))
    _CACHE[key] = out
    return out


def _split(y, n, props, seed, dev):
    g = torch.Generator().manual_seed(seed)
    yc = y.cpu()
    ncls = int(yc.max()) + 1
    sh = [[] for _ in range(n)]
    for c in range(ncls):
        ci = torch.where(yc == c)[0]
        ci = ci[torch.randperm(len(ci), generator=g)]
        cut = torch.tensor(np.cumsum(props[c])[:-1] * len(ci), dtype=torch.long)
        for k, part in enumerate(torch.tensor_split(ci, cut)):
            sh[k].append(part)
    out = []
    for s in sh:
        s = torch.cat(s) if s else torch.arange(0)
        out.append(s.to(dev))
    return out


def split(y, n, dist, seed, dev):
    """IID uniform split or a seed-fixed Dirichlet label skew."""
    if dist == "iid":
        g = torch.Generator().manual_seed(seed)
        return [s.to(dev) for s in torch.tensor_split(torch.randperm(len(y), generator=g), n)]
    alpha = float(dist.replace("dir", ""))
    rng = np.random.default_rng(seed)
    for _ in range(10_000):
        props = np.stack([rng.dirichlet(np.full(n, alpha)) for _ in range(int(y.max()) + 1)])
        out = _split(y, n, props, seed, dev)
        if min(map(len, out)) >= 32:
            return out
    raise RuntimeError("failed to draw a Dirichlet split with at least 32 samples per client")


def split_pair(y, yt, n, dist, seed, dev):
    """Train and test share the per-client class mixture so worst-client accuracy can be measured."""
    if dist == "iid":
        return split(y, n, dist, seed, dev), split(yt, n, dist, seed + 1, dev)
    alpha = float(dist.replace("dir", ""))
    rng = np.random.default_rng(seed)
    for _ in range(10_000):
        props = np.stack([rng.dirichlet(np.full(n, alpha)) for _ in range(int(y.max()) + 1)])
        train = _split(y, n, props, seed, dev)
        test = _split(yt, n, props, seed + 1, dev)
        if min(map(len, train)) >= 32 and min(map(len, test)) >= 1:
            return train, test
    raise RuntimeError("failed to draw a non-empty paired Dirichlet split")
