"""Attack implementations.

The runner builds only the minimal context each attack family may see, so this module never
receives the server global model, other clients' histories or the challenge seed.
Received-history attacks get ``recv_models``/``recv_history`` that hold only the attacker's own
state, and ``recv_norms`` holds past scales recorded before the current round.
"""
import torch

# Exact inputs permitted to F7. No cfg, server RNG, scores or honest updates.
F7_INPUTS = {"t", "k", "D", "dev", "recv_models", "recv_history", "recv_norms"}


def f7_ready(c):
    return len(c["recv_history"]) >= 7


def f7_make(c):
    """F7, detector-aware stale reuse: repeat the direction of the difference between the first
    two received models, scaled to the recent received-difference norm median. It uses the
    attacker's own received models only."""
    if set(c) != F7_INPUTS:
        raise ValueError("unexpected F7 inputs")
    if not f7_ready(c):
        return None
    history, received = c["recv_history"], c["recv_models"]
    unit = lambda x: x / (x.norm() + 1e-12)
    anchor = unit(received[history[1]] - received[history[0]])
    return recv_scale(c) * unit(anchor)


RECV = {"F3", "F4", "F5", "F6"}  # attacks that use received-model history
TRAIN = {"none"}                             # performs real training
F3_GAMMA = 1.0                               # phi(t) = sigma_a t^-gamma; primary value of the literature panel {0.5,1,2}


def recv_delta(c, k):
    """Received-model difference against the previous participation round: Delta_g + (eps^t - eps^r)."""
    history = c["recv_history"]
    return c["recv_models"][c["t"]] - c["recv_models"][history[-1]]


def anchor_sigma(c):
    """sigma_a of F3: std of the update between the first two received rounds, estimated once."""
    r0, r1 = c["recv_history"][0], c["recv_history"][1]
    return float((c["recv_models"][r1] - c["recv_models"][r0]).std())


def recv_scale(c):
    """Median of recent past received-delta norms. The caller records the current round afterwards."""
    values = list(c.get("recv_norms", ()))
    if not values:
        return 0.0
    values = sorted(values[-5:])
    return values[len(values) // 2]


def ready(name, c, k):
    """Per-attack history readiness, fixed by the experiment protocol."""
    if name == "F4-ref":
        return c.get("dg") is not None
    if name == "F5-ref":
        return c.get("dg") is not None and c["n_dg"] >= 5
    if name == "F3":
        return len(c["recv_history"]) >= 2
    if name in RECV:
        return bool(c["recv_history"])
    return True


def make(name, c, k):
    """Build the attack update. Returning None sends the client down the honest-training path."""
    g, D, dev = c["rng"], c["D"], c["dev"]
    if not ready(name, c, k):
        return None
    if name in TRAIN:
        return None
    if name == "F1":
        return torch.zeros(D, device=dev)
    if name == "F2":
        return (torch.rand(D, generator=g, device=dev) * 2 - 1) * c["r_rand"]
    if name == "F4-ref":
        return c["lam"] * c["dg"]
    if name == "F5-ref":
        return c["lam"] * c["dg"] + torch.randn(D, generator=g, device=dev) * (c["kappa"] * c["s_t"])
    if name == "F3":
        # decaying stochastic perturbation; sigma_a is fixed at the anchor rounds
        return torch.randn(D, generator=g, device=dev) * (anchor_sigma(c) * c["t"] ** -F3_GAMMA)
    v = recv_delta(c, k)
    med = recv_scale(c)
    if name == "F4":
        return med * v / (v.norm() + 1e-12)
    if name == "F5":
        return v + torch.randn(D, generator=g, device=dev) * (c["kappa"] * med / D ** 0.5)
    if name == "F6":
        return v / (v.norm() + 1e-12)        # replaced by the shared direction in collude()
    raise ValueError(name)


def collude(us, atk, c):
    """F6 collusion: shared mean direction, pooled prior norm scale and camouflage noise."""
    histories = c["norm_history"]
    col = [k for k in atk if k in us and histories.get(k)]
    if not col:
        return
    directions = {k: us[k].clone() for k in col}
    vals = sorted(x for kk in col for x in histories[kk][-5:])
    med = vals[len(vals) // 2]
    shared = torch.stack([directions[kk] for kk in col]).mean(0)
    for k in col:
        g = c["camouflage_rngs"][k]
        us[k] = med * shared / (shared.norm() + 1e-12) + \
            torch.randn(c["D"], generator=g, device=c["dev"]) * (c["kappa"] * med / c["D"] ** 0.5)
