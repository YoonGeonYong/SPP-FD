"""Config cross-product -> parallel runs. Results go under results_current/<source hash>."""
import itertools
import csv
import fcntl
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

from .config import Cfg
from .metrics import summarize, summarize_active

RESULT_SCHEMA = 5


@lru_cache(maxsize=1)
def harness_hash():
    """Reject stale caches when the source that defines the experiment changes."""
    repo = Path(__file__).resolve().parents[1]
    paths = [repo / "experiments.py", *sorted((repo / "fl").glob("*.py"))]
    h = hashlib.sha256()
    for path in paths:
        h.update(str(path.relative_to(repo)).encode())
        h.update(b"\0")
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


ROOT = str(Path(__file__).resolve().parents[1] / "results_current" / harness_hash())


def outdir(cfg):
    return os.path.join(ROOT, cfg.hash())


def done(cfg):
    d = outdir(cfg)
    summary_path, records_path = (os.path.join(d, name)
                                  for name in ("summary.json", "records.json"))
    if not all(os.path.exists(p) for p in (summary_path, records_path)):
        return False
    try:
        with open(summary_path) as f:
            summary = json.load(f)
        with open(records_path, "rb") as f:
            raw = f.read()
            checksum = hashlib.sha256(raw).hexdigest()
        from .runner import _participants
        records = json.loads(raw)
        expected = {(t, k) for t in range(1, cfg.rounds + 1)
                    for k in _participants(cfg.n_client, cfg.participation, cfg.seed, t)}
        complete = (len(records) == len(expected) and
                    {(r["t"], r["k"]) for r in records} == expected and
                    [r["t"] for r in summary["round_metrics"]] == list(range(1, cfg.rounds + 1)))
        with open(os.path.join(d, "round_metrics.csv"), "rb") as f:
            round_checksum = hashlib.sha256(f.read()).hexdigest()
        return (complete and summary.get("status") == "complete" and
                summary.get("round_metrics_sha256") == round_checksum and
                summary.get("schema_version") == RESULT_SCHEMA and
                summary.get("records_sha256") == checksum and
                summary.get("config_hash") == cfg.hash() and
                summary.get("harness_hash") == harness_hash() and
                summary["cfg"].get("device") == cfg.device and
                Cfg(**summary["cfg"]).hash() == cfg.hash())
    except (OSError, AttributeError, KeyError, TypeError, ValueError):
        return False


def _dump(path, value):
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(value, f, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def run_one(cfg, force=False, *, run_kwargs=None):
    cfg.validate_protocol()
    if cfg.attack == "F7" and not all(k in (run_kwargs or {}) for k in ("server_seed", "attack_seed")):
        raise ValueError("F7 execution requires registered runtime seeds")
    os.makedirs(ROOT, exist_ok=True)
    # Same-config writers must not overwrite each other's round artifacts.
    with open(os.path.join(ROOT, f".{cfg.hash()}.lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _run_one(cfg, force, run_kwargs=run_kwargs)


def _run_one(cfg, force, *, run_kwargs=None):
    from .runner import run
    d = outdir(cfg)
    seeds = {k: v for k, v in (run_kwargs or {}).items() if k in ("server_seed", "attack_seed")}
    runtime_digest = hashlib.sha256(json.dumps(seeds, sort_keys=True).encode()).hexdigest() if seeds else None
    if done(cfg) and not force:
        cached = json.load(open(os.path.join(d, "summary.json")))
        if cached.get("runtime_seed_sha256") != runtime_digest:
            raise ValueError("cached runtime seeds differ from the registered F7 seeds")
        return cached
    if os.path.exists(d):
        archive = os.path.join(ROOT, "attempts", f"{cfg.hash()}-{time.time_ns()}")
        os.makedirs(os.path.dirname(archive), exist_ok=True)
        os.rename(d, archive)
    os.makedirs(d)
    t0 = time.time()
    records_path = os.path.join(d, "records.json")
    rounds_path = os.path.join(d, "round_metrics.csv")
    state_path = os.path.join(d, "state.json")
    state = dict(status="running", cfg=cfg.to_dict(), harness_hash=harness_hash(), last_round=0)
    _dump(state_path, state)
    saved = []

    def on_round(rows, metrics):
        saved.extend(rows)
        _dump(records_path, saved)
        with open(rounds_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics))
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(metrics)
            f.flush()
            os.fsync(f.fileno())
        state["last_round"] = metrics["t"]
        _dump(state_path, state)

    try:
        rec, accs, cost = run(cfg, on_round=on_round, **(run_kwargs or {}))
        if rec != saved or state["last_round"] != cfg.rounds:
            raise ValueError("incomplete round callback output")
        s = _summarize_run(cfg, rec, accs, cost, t0)
        if runtime_digest is not None:
            s["runtime_seed_sha256"] = runtime_digest
        s["records_bytes"] = os.path.getsize(records_path)
        s["records_sha256"] = hashlib.sha256(Path(records_path).read_bytes()).hexdigest()
        s["round_metrics_sha256"] = hashlib.sha256(Path(rounds_path).read_bytes()).hexdigest()
        _dump(os.path.join(d, "summary.json"), s)
        if not done(cfg):
            raise ValueError("result completeness check failed")
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        _dump(state_path, state)
        raise
    state["status"] = "complete"
    _dump(state_path, state)
    return s


def _summarize_run(cfg, rec, accs, cost, t0):
    s = summarize_active(rec, cfg.attack_from)
    s["all_round_diagnostic"] = summarize(rec, cfg.attack_from)
    # Scores from different rounds need not have the same calibration.
    for key in ("auroc", "auprc", "recall_at_fpr01", "recall_at_fpr05"):
        values = [r[key] for r in cost["round_metrics"]
                  if r["t"] >= cfg.attack_from and r[key] is not None]
        s["pooled_" + key] = s[key]
        s[key] = sum(values) / len(values) if values else None
        s[key + "_n_rounds"] = len(values)
    s["ranking_scope"] = "mean of defined active-round metrics; pooled_* are auxiliary"
    s.update(schema_version=RESULT_SCHEMA, status="complete", cfg=cfg.to_dict(), config_hash=cfg.hash(),
             protocol="server_noise" if cfg.noise_sigma > 0 else "standard",
             sigma_resolved=cfg.noise_sigma,
             harness_hash=harness_hash(),
             pairing_id=cfg.pairing_hash(),
             trajectory_id=cfg.trajectory_hash(),
             acc=accs, acc_final=accs[-1][1] if accs else None,
             secs=round(time.time() - t0, 1), **cost)
    return s


def grid(base: Cfg, **axes):
    keys = list(axes)
    return [replace(base, **dict(zip(keys, v))) for v in itertools.product(*[axes[k] for k in keys])]


def main(cfgs, shard=0, nshard=1):
    todo = [c for i, c in enumerate(cfgs) if i % nshard == shard]
    for i, c in enumerate(todo):
        if done(c):
            continue
        s = run_one(c)
        auc = s.get("auroc") if s.get("auroc") is not None else float("nan")
        recall = s.get("recall") if s.get("recall") is not None else float("nan")
        fpr = s.get("fpr") if s.get("fpr") is not None else float("nan")
        print(f"[{shard}] {i+1}/{len(todo)} {c.detector:11} {c.attack:6} {c.dist:7} "
              f"s{c.seed} auroc={auc:.3f} "
              f"rec={recall:.3f} fpr={fpr:.4f} "
              f"acc={s.get('acc_final')} {s['secs']}s", flush=True)


if __name__ == "__main__":
    sys.exit("use experiments.py")
