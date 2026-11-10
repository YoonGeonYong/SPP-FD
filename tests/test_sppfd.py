"""CPU checks of the SPP-FD signals, thresholds and attack formulas. Run: python tests/test_sppfd.py"""
import sys
from pathlib import Path
from statistics import NormalDist

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fl import attacks, detectors  # noqa: E402
from fl.config import Cfg  # noqa: E402
from fl.runner import eps_fn  # noqa: E402

D, DEV = 2000, "cpu"
CFG = Cfg(device=DEV, sigma=.005)
EPS = eps_fn(0, {t: .005 for t in range(1, 12)}, D, DEV)


def step(u, signals="S1,S2,S3,S4", prev=None, k=0):
    det = detectors.SPPFD(Cfg(device=DEV, sigma=.005, signals=signals), D, DEV)
    ctx = dict(t=6, us={k: u}, dg=None, hist={k: [1, 2, 3, 4, 5]},
               prev_u={} if prev is None else {k: prev}, eps=EPS, extra={})
    scores, flags = det.step(ctx)
    return scores[k], k in flags, ctx["extra"][k]


def test_thresholds():
    nd = NormalDist()
    assert (CFG.tau2, CFG.tau3, CFG.tau4) == (3.023, 3.5, 2.326)
    assert round(nd.inv_cdf(1 - 0.01 / 8), 3) == 3.023       # Bonferroni over 4 probes, two-sided
    assert round(nd.inv_cdf(0.99), 3) == 2.326


def test_s1_zero_update():
    score, flagged, _ = step(torch.zeros(D))
    assert flagged and score == 1e9


def test_s2_flags_received_difference_reuse():
    u = EPS(6, 0) - EPS(5, 0)                  # F4-like update with no global change
    s2 = step(u, signals="S2")[2]["s2"]
    assert s2 > 0.9 * D ** .5 / 2              # large-d limit sqrt(d)/2
    assert step(u, signals="S2")[1]


def test_s2_quiet_on_independent_update():
    g = torch.Generator().manual_seed(1)
    flags = [step(torch.randn(D, generator=g), signals="S2")[1] for _ in range(60)]
    assert sum(flags) / len(flags) < 0.1       # reference level is about 1%


def test_s4_temporal_alignment():
    g = torch.Generator().manual_seed(2)
    v = torch.randn(D, generator=g)
    assert not step(v, signals="S4", prev=v)[1]                       # identical -> sqrt(d)
    flags = [step(torch.randn(D, generator=g), signals="S4", prev=torch.randn(D, generator=g))[1]
             for _ in range(60)]
    assert sum(flags) / len(flags) > 0.9       # independent directions fall below tau4


def test_f4_scale_is_median_of_past_norms():
    g = torch.Generator().manual_seed(3)
    m = {1: torch.randn(D, generator=g), 2: torch.randn(D, generator=g), 3: torch.randn(D, generator=g)}
    c = dict(rng=g, D=D, dev=DEV, t=3, recv_models=m, recv_history=[2], recv_norms=[1., 2., 3.])
    u = attacks.make("F4", c, 0)               # paper F4: m * D / (|D| + delta)
    assert abs(float(u.norm()) - 2.0) < 1e-4
    v = m[3] - m[2]
    assert float(u @ v) / (float(u.norm()) * float(v.norm())) > 1 - 1e-6


def test_server_noise_only_with_spp_fd():
    try:
        Cfg(detector="none", attack="F4", sigma=.005).validate_protocol()
    except ValueError:
        return
    raise AssertionError("server perturbation must be restricted to SPP-FD and clean controls")


def test_only_spp_fd_is_registered():
    assert set(detectors.REGISTRY) == {"none", "spp_fd"}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
