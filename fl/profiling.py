"""Optional E5 input replay. Capture/replay is separate from reported learning time."""
import copy
import math
import statistics
import time

import torch

from .detectors import _cos, _robust_z


class ReplayInputs:
    """Pass as runner.run(profile_hook=...). Keeps immutable tensors, copied mutable state.

    Inputs are retained in memory for one run; process each run before capturing
    the next one. The runner never mutates submitted tensors after this hook.
    """
    def __init__(self):
        self.rounds = []

    def __call__(self, cfg, detector, ctx):
        if cfg.detector != "spp_fd":
            raise ValueError("E5 replay is defined for SPP-FD")
        snapshot = {k: ctx[k] for k in ("t", "w", "dg", "eps")}
        snapshot.update(us=dict(ctx["us"]), prev_u=dict(ctx["prev_u"]),
                        hist={k: list(v) for k, v in ctx["hist"].items()}, extra={})
        self.rounds.append((copy.deepcopy(detector), snapshot))


def profile_round(detector, snapshot, warmup=20, repeats=100):
    """Synchronized wall time; state restoration excluded from each timed region.

    Per-signal timings are auxiliary isolated computations with the same inputs;
    their sum is not the overall detector measurement. No learning, evaluation,
    diagnostics or disk writes are included.
    """
    if warmup < 0 or repeats < 1:
        raise ValueError("invalid profiling repetition counts")
    dev, cfg = detector.dev, detector.cfg
    device = torch.device(dev)
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    def fresh():
        return dict(snapshot, hist={k: list(v) for k, v in snapshot["hist"].items()},
                    prev_u=dict(snapshot["prev_u"]), extra={})
    def measure(factory):
        values = []
        for i in range(warmup + repeats):
            operation = factory()
            sync()
            start = time.perf_counter()
            operation()
            sync()
            elapsed = time.perf_counter() - start
            if i >= warmup:
                values.append(elapsed)
        ordered = sorted(values)
        return dict(median_secs=statistics.median(values),
                    p95_secs=ordered[math.ceil(.95 * repeats) - 1], samples_secs=values)
    def detection():
        restored, ctx = copy.deepcopy(detector), fresh()
        return lambda: restored.step(ctx)
    def history():
        ctx = fresh()
        def update():
            for k, u in ctx["us"].items():
                ctx["hist"][k].append(ctx["t"])
                ctx["prev_u"][k] = u
        return update
    clients, w, t = list(snapshot["us"]), snapshot["w"], snapshot["t"]
    def original():
        return {k: w.clone() for k in clients}
    def noisy():
        return {k: w + snapshot["eps"](t, k) for k in clients}
    def aggregate():
        return w + torch.stack(list(snapshot["us"].values())).mean(0)
    def s1():
        return {k: float(u.norm()) == 0 for k, u in snapshot["us"].items()}
    def s2():
        return {k: detector._s2(snapshot, k, u, float(u.norm()))
                for k, u in snapshot["us"].items() if float(u.norm()) > 0}
    def s3():
        if snapshot["dg"] is None:
            return {}
        aligns = {k: _cos(u, snapshot["dg"]) if float(u.norm()) else 0.
                  for k, u in snapshot["us"].items()}
        current = [aligns[k] for k, u in snapshot["us"].items() if float(u.norm()) > 0]
        pool = [x for row in (detector.awin + [current])[-cfg.s3_rounds:] for x in row]
        median, mad = _robust_z(pool) if pool else (0., 1.)
        return {k: (a - median) / mad for k, a in aligns.items()}
    def s4():
        return {k: math.sqrt(detector.D) * _cos(u, snapshot["prev_u"][k])
                for k, u in snapshot["us"].items()
                if float(u.norm()) > 0 and k in snapshot["prev_u"]
                and float(snapshot["prev_u"][k].norm()) > 0}
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        expected = copy.deepcopy(detector).step(fresh())
        assert copy.deepcopy(detector).step(fresh()) == expected
        result = dict(original_prepare=measure(lambda: original),
                      noisy_prepare=measure(lambda: noisy), detector=measure(detection),
                      history=measure(history), aggregation=measure(lambda: aggregate))
        result["signals_auxiliary"] = {name: measure(lambda fn=fn: fn)
                                       for name, fn in (("S1", s1), ("S2", s2), ("S3", s3), ("S4", s4))}
        result["empty_timer"] = measure(lambda: lambda: None)
        result["four_timers_floor_fraction_of_detector"] = (
            4 * result["empty_timer"]["median_secs"] / result["detector"]["median_secs"])
    # Paired repetition differences retain negative measured overhead, if any.
    added = [n - o + d + h for n, o, d, h in zip(
        result["noisy_prepare"]["samples_secs"], result["original_prepare"]["samples_secs"],
        result["detector"]["samples_secs"], result["history"]["samples_secs"])]
    result["server_added"] = dict(median_secs=statistics.median(added),
                                  p95_secs=sorted(added)[math.ceil(.95 * repeats) - 1])
    result.update(round=t, warmup=warmup, repeats=repeats,
                  detector_flags=sorted(expected[1]),
                  auxiliary_scope="isolated computations; includes repeated norm/loop work",
                  timer_scope="four empty synchronized timers estimate a measurement floor, not all instrumentation cost",
                  excluded="state restoration, learning, evaluation, diagnostics, files")
    return result
