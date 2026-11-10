"""FL loop. Deterministic seeds; returns client-round records."""
import math
import os
import statistics
import time

import torch

from . import attacks, data, detectors, models
from .metrics import summarize_active


def eps_fn(seed, sigma_hist, D, dev):
    def f(t, k):
        if sigma_hist[t] == 0:
            return torch.zeros(D, device=dev)
        g = torch.Generator(device=dev).manual_seed((seed * 9973 + t) * 97 + k)
        return torch.randn(D, generator=g, device=dev) * sigma_hist[t]
    return f


def _participants(n_client, participation, seed, t):
    """Bernoulli participation with deterministic, uniform empty-round fallback."""
    if participation >= 1.0:
        return list(range(n_client))
    pg = torch.Generator().manual_seed((seed * 613 + t) * 7)
    act = [k for k in range(n_client)
           if float(torch.rand(1, generator=pg)) < participation]
    if not act:
        act = [int(torch.randint(n_client, (1,), generator=pg))]
    return act


def _attack_context(name, *, t, k, rng, D, dev, lam, kappa, r_rand,
                    n_dg, dg, s_t, recv_models, recv_history, recv_norms):
    """Build the smallest context permitted for one attack family."""
    c = dict(t=t, k=k, rng=rng, D=D, dev=dev)
    if name == "F2":
        c["r_rand"] = r_rand
    if name in {"F4-ref", "F5-ref"}:
        c["lam"] = lam
    if name in {"F5-ref", "F5"}:
        c["kappa"] = kappa
    if name == "F4-ref":
        c["dg"] = dg
    elif name == "F5-ref":
        c.update(dg=dg, s_t=s_t, n_dg=n_dg)
    elif name == "F3":
        c.update(recv_models=recv_models, recv_history=recv_history)
    elif name in attacks.RECV:
        c.update(recv_models=recv_models, recv_history=recv_history,
                 recv_norms=recv_norms)
    return c


def run(cfg, on_round=None, *, server_seed=None, attack_seed=None, profile_hook=None,
        observe_noise=False):
    cfg.validate_protocol()
    independent = cfg.attack == "F7"      # F7 uses independent server and attack seeds
    if independent:
        # Keep the unchanged eps_fn integer derivation within torch's uint64 seed range.
        if not all(type(s) is int and 0 <= s < 2 ** 40 for s in (server_seed, attack_seed)):
            raise ValueError("F7 requires separate server_seed and attack_seed runtime integers")
        if len({cfg.seed, server_seed, attack_seed}) != 3:
            raise ValueError("F7 runtime seeds must be distinct from each other and cfg.seed")
    elif server_seed is not None or attack_seed is not None:
        raise ValueError("independent runtime seeds apply to F7 only")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(cfg.seed)
    torch.use_deterministic_algorithms(True)
    dev = torch.device(cfg.device)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
    ds = cfg.dataset
    D = models.dim(ds)
    X, y, Xt, yt, = data.load(ds, dev)
    sh, sht = data.split_pair(y, yt, cfg.n_client, cfg.dist, cfg.seed, dev)
    atk = set(range(int(cfg.n_client * cfg.fr_frac))) if cfg.attack != "none" else set()
    det = detectors.build(cfg, D, dev)

    w = models.init(ds, cfg.seed, dev)
    Wg, winit, sigma_hist = {}, {}, {}
    hist = {k: [] for k in range(cfg.n_client)}
    recv_models = {k: {} for k in range(cfg.n_client)}
    recvn = {k: [] for k in range(cfg.n_client)}
    mbuf = {k: torch.zeros(D, device=dev) for k in range(cfg.n_client)}
    prev_u, dgn, rec, accs = {}, [], [], []
    eps = eps_fn(server_seed if independent else cfg.seed, sigma_hist, D, dev)
    attack_secs = 0.0
    attacker_training_secs = 0.0
    noise_ratios = []
    client_secs = detector_secs = probe_secs = evaluation_secs = diagnostic_secs = 0.0
    round_metrics = []
    upload_bytes = download_bytes = 0
    client_acc = []
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    for t in range(1, cfg.rounds + 1):
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        round_t0 = time.perf_counter()
        Wg[t] = w.clone()
        sigma_hist[t] = cfg.noise_sigma
        dg = Wg[t - 1] - Wg[t - 2] if t > 2 else None
        if dg is not None:
            dgn.append(float(dg.norm()))
        s_t = statistics.median(dgn[-5:]) / math.sqrt(D) if dgn else 0.0

        act = _participants(cfg.n_client, cfg.participation, cfg.seed, t)

        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        probe_t0 = time.perf_counter()
        for k in act:
            winit[(t, k)] = w + eps(t, k) if sigma_hist[t] > 0 else w.clone()
            recv_models[k][t] = winit[(t, k)]
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        probe_elapsed = time.perf_counter() - probe_t0
        probe_secs += probe_elapsed
        download_bytes += len(act) * D * 4
        if observe_noise:
            model_norm = float(w.norm())
            noise_ratios.append(dict(t=t, model_norm=model_norm,
                client_ratios={str(k): float((winit[(t, k)] - w).norm()) / model_norm
                               if model_norm else None for k in act}))

        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        client_t0 = time.perf_counter()
        us, active_attack, train_samples = {}, {}, {}
        current_recv_norms = {}
        for k in act:
            prior = hist[k]
            if prior:
                current_recv_norms[k] = float(
                    (recv_models[k][t] - recv_models[k][prior[-1]]).norm())
            requested = k in atk and t >= cfg.attack_from
            attack_t0 = time.perf_counter()
            if cfg.attack == "F7":
                c = dict(t=t, k=k, D=D, dev=dev, recv_models=dict(recv_models[k]),
                         recv_history=list(hist[k]), recv_norms=list(recvn[k]))
                on = requested and attacks.f7_ready(c)
                u = attacks.f7_make(c) if on else None
            else:
                c = _attack_context(
                    cfg.attack, t=t, k=k,
                    rng=torch.Generator(device=dev).manual_seed(
                        ((attack_seed if independent else cfg.seed) * 31 + t) * 131 + k),
                    D=D, dev=dev, lam=cfg.lam, kappa=cfg.kappa, r_rand=cfg.r_rand,
                    n_dg=len(dgn), dg=dg, s_t=s_t, recv_models=dict(recv_models[k]),
                    recv_history=list(hist[k]), recv_norms=list(recvn[k]))
                on = requested and attacks.ready(cfg.attack, c, k)
                u = attacks.make(cfg.attack, c, k) if on else None
            if independent and on:
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)
                attack_secs += time.perf_counter() - attack_t0
            active_attack[k] = on
            if u is None:                                   # honest-training path
                idx = sh[k]
                w0 = winit[(t, k)]
                if independent and k in atk:
                    if dev.type == "cuda":
                        torch.cuda.synchronize(dev)
                    training_t0 = time.perf_counter()
                u = models.train_local(w0, X[idx], y[idx], cfg, (cfg.seed * 7 + t) * 17 + k, ds,
                                       mbuf[k] if cfg.momentum > 0 else None)
                if independent and k in atk:
                    if dev.type == "cuda":
                        torch.cuda.synchronize(dev)
                    attacker_training_secs += time.perf_counter() - training_t0
                train_samples[k] = len(idx)
            if u is not None and k not in train_samples:
                train_samples[k] = 0
            us[k] = u
        if cfg.attack == "F6":
            active_atk = {k for k in atk if active_attack.get(k, False)}
            camouflage_rngs = {
                k: torch.Generator(device=dev).manual_seed((cfg.seed * 31 + t) * 131 + k)
                for k in active_atk if k in us
            }
            collusion_context = dict(
                D=D, dev=dev, kappa=cfg.kappa,
                norm_history={k: list(recvn[k]) for k in active_atk if k in us},
                camouflage_rngs=camouflage_rngs)
            attacks.collude(us, active_atk, collusion_context)
        for k, norm in current_recv_norms.items():
            recvn[k].append(norm)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        client_elapsed = time.perf_counter() - client_t0
        client_secs += client_elapsed

        extra = {}
        ctx = dict(t=t, us=us, w=w, w_prev=Wg.get(t - 1), dg=dg, winit=winit, eps=eps,
                   hist=hist, prev_u=prev_u, extra=extra)
        if profile_hook is not None:
            profile_hook(cfg, det, ctx)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        detector_t0 = time.perf_counter()
        scores, flags = det.step(ctx)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        detector_elapsed = time.perf_counter() - detector_t0
        detector_secs += detector_elapsed
        diagnostic_t0 = time.perf_counter()
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        diagnostic_elapsed = time.perf_counter() - diagnostic_t0
        diagnostic_secs += diagnostic_elapsed
        upload_bytes += len(us) * D * 4

        keep = [k for k in us if k not in flags] if cfg.filt else list(us)
        fallback = not keep
        if not keep:
            keep = list(us)

        round_start = len(rec)
        for k in us:
            e = extra.get(k, {})
            clean = lambda b: (b if isinstance(b, bool) or b is None else
                               (float(b) if math.isfinite(float(b)) else None))
            rec.append(dict(t=t, k=k, attacker=k in atk, atk=active_attack[k],
                            score=float(scores.get(k, 0.0)),
                            flag=k in flags, kept=k in keep, aggregation_fallback=fallback,
                            un=float(us[k].norm()),
                            dg=float(dg.norm()) if dg is not None else 0.0,
                            pn=sigma_hist[t] * math.sqrt(2 * D),
                            n_train=train_samples[k], n_nominal=len(sh[k]),
                            train_fraction=train_samples[k] / len(sh[k]),
                            **{f"x_{a}": clean(b) for a, b in e.items()}))

        for k in us:
            if cfg.detector == "spp_fd" and "S4" in cfg.signals:
                prev_u[k] = us[k]
            hist[k].append(t)
            keep_rounds = set(hist[k][:2]) | {hist[k][-1]}   # F3 anchor + previous participation round
            recv_models[k] = {r: recv_models[k][r] for r in keep_rounds
                              if r in recv_models[k]}
        w = w + torch.stack([us[k] for k in keep]).mean(0)
        for q in [q for q in winit if q[0] < t - 6]:
            del winit[q]
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        evaluation_t0 = time.perf_counter()
        accuracy, loss = models.evaluate(w, Xt, yt, ds)
        accs.append((t, accuracy))
        if t == cfg.rounds:
            client_acc = [models.accuracy(w, Xt[idx], yt[idx], ds) for idx in sht]
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        evaluation_elapsed = time.perf_counter() - evaluation_t0
        evaluation_secs += evaluation_elapsed
        rm = summarize_active(rec[round_start:], 1)
        rm.pop("per_client_fpr")
        rm.pop("attack_from")
        rm["metric_scope"] = "this_round: positive=atk; negative=not_attacker"
        round_metrics.append(dict(t=t, accuracy=accuracy, loss=loss,
                                  n_participants=len(us), n_flagged=len(flags),
                                  aggregation_fallback=fallback,
                                  sigma_actual=sigma_hist[t], **rm,
                                  round_secs=time.perf_counter() - round_t0,
                                  client_secs=client_elapsed, detector_secs=detector_elapsed,
                                  probe_secs=probe_elapsed, diagnostic_secs=diagnostic_elapsed,
                                  evaluation_secs=evaluation_elapsed))
        if on_round is not None:
            on_round(rec[round_start:], round_metrics[-1])
    # Estimate of the detector state, not peak process memory.
    storage = (cfg.n_client * D * 4 + cfg.n_client * cfg.history * 8) if cfg.detector == "spp_fd" else 0
    cost = dict(round_metrics=round_metrics, probe_secs=probe_secs,
                evaluation_secs=evaluation_secs, diagnostic_secs=diagnostic_secs,
                client_secs=client_secs,
                client_secs_per_round=client_secs / cfg.rounds,
                detector_secs=detector_secs,
                detector_secs_per_round=detector_secs / cfg.rounds,
                peak_gpu_bytes=(torch.cuda.max_memory_allocated(dev) if dev.type == "cuda" else 0),
                upload_bytes=upload_bytes,
                extra_upload_bytes=0,
                download_bytes=download_bytes,
                total_communication_bytes=upload_bytes + download_bytes,
                detector_storage_bytes_est=storage,
                client_acc=client_acc,
                worst_honest_client_acc=min((a for k, a in enumerate(client_acc) if k not in atk),
                                           default=None),
                worst_client_acc=min(client_acc) if client_acc else None)
    if independent:
        cost["attack_generation_secs"] = attack_secs
        cost["attacker_training_secs"] = attacker_training_secs
        cost["attacker_trained_client_rounds"] = sum(r["attacker"] and r["n_train"] > 0 for r in rec)
    if observe_noise:
        cost["observed_noise_to_model_norm"] = noise_ratios
    return rec, accs, cost
