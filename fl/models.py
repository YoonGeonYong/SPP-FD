"""Flat parameter-vector interface. Every detector sees only 1-D vectors."""
import math

import torch
import torch.nn.functional as F

NETS = {
    "fashion": [(16, 1, 3, 3), (16,), (32, 16, 3, 3), (32,), (128, 800), (128,), (10, 128), (10,)],
    "cifar10": [(32, 3, 3, 3), (32,), (64, 32, 3, 3), (64,), (128, 2304), (128,), (10, 128), (10,)],
}


def dim(ds):
    return sum(math.prod(s) for s in NETS[ds])


def unflat(w, ds):
    out, i = [], 0
    for s in NETS[ds]:
        n = math.prod(s)
        out.append(w[i:i + n].view(s))
        i += n
    return out


def fwd(w, x, ds):
    c1, b1, c2, b2, f1, d1, f2, d2 = unflat(w, ds)
    h = F.max_pool2d(F.relu(F.conv2d(x, c1, b1)), 2)
    h = F.max_pool2d(F.relu(F.conv2d(h, c2, b2)), 2)
    h = F.relu(F.linear(h.flatten(1), f1, d1))
    return F.linear(h, f2, d2)


def init(ds, seed, dev):
    g = torch.Generator(device=dev).manual_seed(seed)
    parts = []
    for s in NETS[ds]:
        if len(s) == 1:
            parts.append(torch.zeros(s[0], device=dev))
        else:
            parts.append(torch.randn(s, generator=g, device=dev) * math.sqrt(2 / math.prod(s[1:])))
    return torch.cat([p.flatten() for p in parts])


def train_local(w0, X, y, cfg, seed, ds, buf=None):
    """Local SGD from w0; buf carries momentum. Returns the update w - w0."""
    w = w0.clone().requires_grad_(True)
    g = torch.Generator(device=w.device).manual_seed(seed)
    m = buf if buf is not None else torch.zeros_like(w0)
    for _ in range(cfg.local_epoch):
        perm = torch.randperm(len(y), generator=g, device=w.device)
        for s in range(0, len(y), cfg.batch):
            idx = perm[s:s + cfg.batch]
            (gr,) = torch.autograd.grad(F.cross_entropy(fwd(w, X[idx], ds), y[idx]), w)
            with torch.no_grad():
                if cfg.momentum > 0:
                    m.mul_(cfg.momentum).add_(gr)
                    w -= cfg.lr * m
                else:
                    w -= cfg.lr * gr
    return w.detach() - w0


@torch.no_grad()
def evaluate(w, X, y, ds, bs=2000):
    hit, loss = 0, 0.0
    for s in range(0, len(y), bs):
        logits, target = fwd(w, X[s:s + bs], ds), y[s:s + bs]
        hit += (logits.argmax(1) == target).sum().item()
        loss += F.cross_entropy(logits, target, reduction="sum").item()
    return hit / len(y), loss / len(y)


def accuracy(w, X, y, ds, bs=2000):
    return evaluate(w, X, y, ds, bs)[0]
