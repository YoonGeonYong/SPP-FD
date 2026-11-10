"""Evaluation metrics: threshold-free (AUROC/AUPRC) plus operating points."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def recall_at_fpr(y, s, target):
    fpr, tpr, _ = roc_curve(y, s)
    ok = tpr[fpr <= target]
    return float(ok.max()) if len(ok) else 0.0


def summarize(rec, attack_from=None):
    y = np.array([r["atk"] for r in rec], bool)
    s = np.array([r["score"] for r in rec], float)
    f = np.array([r["flag"] for r in rec], bool)
    kept = np.array([r["kept"] for r in rec], bool)
    if not np.isfinite(s).all():
        raise ValueError("detector produced non-finite score")
    out = dict(n=len(rec), n_atk=int(y.sum()))
    if y.any() and (~y).any():
        out.update(auroc=float(roc_auc_score(y, s)),
                   auprc=float(average_precision_score(y, s)),
                   recall_at_fpr01=recall_at_fpr(y, s, 0.01),
                   recall_at_fpr05=recall_at_fpr(y, s, 0.05))
    else:
        out.update(auroc=None, auprc=None, recall_at_fpr01=None, recall_at_fpr05=None)
    survivor = ~f
    by_client = {}
    for k in sorted({r["k"] for r in rec}):
        q = np.array([r["k"] == k and not r["atk"] for r in rec])
        if q.any():
            by_client[str(k)] = float(f[q].mean())
    out.update(recall=float(f[y].mean()) if y.any() else None,
               fpr=float(f[~y].mean()) if (~y).any() else None,
               survivor_contamination=float(y[survivor].mean()) if survivor.any() else None,
               n_survivor=int(survivor.sum()),
               aggregation_contamination=float(y[kept].mean()) if kept.any() else None,
               n_aggregated=int(kept.sum()),
               per_client_fpr=by_client,
               worst_client_fpr=max(by_client.values()) if by_client else None)
    if attack_from is not None:
        t = np.array([r["t"] for r in rec])
        pre, post = (~y) & (t < attack_from), (~y) & (t >= attack_from)
        active_survivor = survivor & (t >= attack_from)
        active_aggregated = kept & (t >= attack_from)
        out.update(fpr_pre=float(f[pre].mean()) if pre.any() else None,
                   fpr_post=float(f[post].mean()) if post.any() else None,
                   survivor_contamination_active=(float(y[active_survivor].mean())
                                                  if active_survivor.any() else None),
                   aggregation_contamination_active=(float(y[active_aggregated].mean())
                                                     if active_aggregated.any() else None))
    return out


def summarize_active(rec, attack_from):
    """Detection: actual attacks vs permanent honest clients, after attack_from.

    Aggregation contamination still uses every actual aggregated client, including
    designated attackers who trained honestly while waiting for attack history.
    """
    active = [r for r in rec if r["t"] >= attack_from]
    evaluated = [r for r in active if r["atk"] or not r["attacker"]]
    out = summarize(evaluated)
    for field, select in (("survivor", lambda r: not r["flag"]),
                          ("aggregation", lambda r: r["kept"])):
        rows = [r for r in active if select(r)]
        out[field + "_contamination"] = sum(r["atk"] for r in rows) / len(rows) if rows else None
        out["n_survivor" if field == "survivor" else "n_aggregated"] = len(rows)
    out.update(n_honest=sum(not r["attacker"] for r in evaluated),
               n_not_ready=sum(r["attacker"] and not r["atk"] for r in active),
               tp=sum(r["atk"] and r["flag"] for r in evaluated),
               fp=sum(not r["attacker"] and r["flag"] for r in evaluated),
               fn=sum(r["atk"] and not r["flag"] for r in evaluated),
               tn=sum(not r["attacker"] and not r["flag"] for r in evaluated),
               metric_scope="t>=attack_from: positive=atk; negative=not_attacker",
               attack_from=attack_from)
    return out
