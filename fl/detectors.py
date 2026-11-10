"""Detectors. Each emits a continuous score per client-round (higher = more suspicious), so AUROC is defined.

Common interface
    d.step(ctx) -> (scores: dict[k]->float, flags: set[k])
ctx: t, us(update), w, wg_hist, eps(fn), hist, prev_u, cfg, D, dev
"""
import math

import torch


# ---------------------------------------------------------------- shared helpers
def _cos(a, b):
    return float(a @ b) / (float(a.norm()) * float(b.norm()) + 1e-12)


def _robust_z(vals):
    t = torch.tensor(vals, dtype=torch.float64)
    med = t.median()
    mad = (t - med).abs().median() * 1.4826 + 1e-12
    return med.item(), mad.item()


class Base:
    name = "base"

    def __init__(self, cfg, D, dev):
        self.cfg, self.D, self.dev = cfg, D, dev

    def step(self, ctx):
        return {k: 0.0 for k in ctx["us"]}, set()


class NoDetector(Base):
    name = "none"


# ---------------------------------------------------------------- proposed method (SPP-FD)
class SPPFD(Base):
    """SPP-FD detector (the paper's server-side procedure, one call per round via step()).

    S1 exact-zero guard, S2 secret probe projection, S3 peer-relative trend deviation,
    S4 temporal alignment. A client-round is flagged if any available signal fires (OR); the
    continuous score is the maximum of the signals divided by their thresholds (S1 -> 1e9).
    The server noise itself is added in fl/runner.py (eps_fn and the broadcast step).
    """
    name = "spp_fd"
    PAIRS = [(1, 2), (2, 3), (3, 4), (4, 5)]

    def __init__(self, cfg, D, dev):
        super().__init__(cfg, D, dev)
        self.on = set(cfg.signals.split(","))
        required_history = max(max(i, j) for i, j in self.PAIRS)
        if "S2" in self.on and cfg.history < required_history:
            raise ValueError(
                f"history={cfg.history} is smaller than the S2 probe span {required_history}"
            )
        self.awin = []

    def _s2(self, ctx, k, u, nu):
        pr = ctx["hist"][k][::-1][:self.cfg.history]
        if len(pr) < self.cfg.history:
            return None, [None] * len(self.PAIRS)
        zs = []
        for i, j in self.PAIRS:
            p = ctx["eps"](pr[i - 1], k) - ctx["eps"](pr[j - 1], k)
            zs.append(math.sqrt(self.D) * float(u @ p) / (nu * float(p.norm()) + 1e-12))
        return max(map(abs, zs)), zs

    def step(self, ctx):
        us, dg = ctx["us"], ctx["dg"]
        raw, sc, fl = {}, {}, set()
        for k, u in us.items():
            nu = float(u.norm())
            if nu == 0:
                raw[k] = dict(s1=True, s2=0.0, a=0.0, s4=None,
                              **{f"s2_pair{i}": None for i in range(1, 5)})
                continue
            s2, pairs = (self._s2(ctx, k, u, nu) if "S2" in self.on else
                         (0.0, [None] * len(self.PAIRS)))
            raw[k] = dict(
                s1=False,
                s2=s2,
                a=_cos(u, dg) if dg is not None else 0.0,
                s4=(math.sqrt(self.D) * _cos(u, ctx["prev_u"][k])
                    if ("S4" in self.on and k in ctx["prev_u"]
                        and float(ctx["prev_u"][k].norm()) > 0) else None),
                **{f"s2_pair{i}": v for i, v in enumerate(pairs, 1)},
            )
        ev = [k for k in us if not raw[k]["s1"]]
        if dg is not None:
            self.awin.append([raw[k]["a"] for k in ev])
            pool = [v for row in self.awin[-self.cfg.s3_rounds:] for v in row]
            med, mad = _robust_z(pool) if pool else (0.0, 1.0)
        else:
            med, mad = 0.0, 1.0
        for k in us:
            r = raw[k]
            z3 = (r["a"] - med) / mad
            r["z3"] = z3
            hit = ("S1" in self.on and r["s1"]) or \
                  ("S2" in self.on and r["s2"] is not None and r["s2"] > self.cfg.tau2) or \
                  ("S3" in self.on and dg is not None and abs(z3) > self.cfg.tau3) or \
                  ("S4" in self.on and r["s4"] is not None and r["s4"] < self.cfg.tau4)
            # continuous score: each signal divided by its threshold, then the maximum
            comp = [r["s2"] / self.cfg.tau2 if ("S2" in self.on and r["s2"] is not None) else 0.0,
                    abs(z3) / self.cfg.tau3 if ("S3" in self.on and dg is not None) else 0.0,
                    (self.cfg.tau4 / max(r["s4"], 1e-6)
                     if ("S4" in self.on and r["s4"] is not None) else 0.0)]
            sc[k] = 1e9 if (r["s1"] and "S1" in self.on) else max(comp)
            if hit:
                fl.add(k)
            ctx["extra"][k] = r
        return sc, fl


REGISTRY = {d.name: d for d in [NoDetector, SPPFD]}


def build(cfg, D, dev):
    return REGISTRY[cfg.detector](cfg, D, dev)
