# SPP-FD: Free-Rider Detection in Federated Learning with Server-Private Noise

Code for the paper *SPP-FD: Free-Rider Detection in Federated Learning with Server-Private Noise*
(Geon Yong Yoon, Neetesh Saxena, Bong Jun Choi), accepted at the IEEE BigData 2026 Special Session
on Federated Learning on Big Data.

SPP-FD is a server-side detector for deployments where a trusted server may inspect individual
client updates. The server adds client-specific Gaussian noise to each broadcast model, keeps the
seed, and combines four signals: a zero-update guard (S1), secret probe projection (S2),
peer-relative trend deviation (S3) and temporal alignment (S4).

## What this repository contains

- `fl/`: the federated-learning simulator, the SPP-FD detector (`fl/detectors.py`, class `SPPFD`),
  the attack generators (`fl/attacks.py`), metrics, and the sweep runner with result caching.
- `experiments.py`: the SPP-FD experiment grids (`main`, `main_cifar`, `filt`, `noise_cost`, `smoke*`).
- `tests/test_sppfd.py`: CPU checks of the signals, thresholds and attack formulas.

## What it does not contain

- **Baseline adapters** (STD-DAGMM, Delta-DAGMM, FLDetector, WEF-Defense, CosDefense) are not
  released. The baseline rows and columns of the paper's comparison tables (detection comparison,
  filtering by method, detection cost) therefore cannot be reproduced from this repository.
- Raw results, datasets and the manuscript.
- Scripts that regenerate the paper's tables, and drivers for the extension experiments (noise-scale
  sweep, 100-client and partial-participation runs, partial training). This repository is meant for
  running and building on SPP-FD, not for reproducing the paper's results end to end.

## Where the algorithm is

The method has two parts: the server adds client-specific noise to each broadcast model, and the
detector scores the returned updates. Everything is in `fl/`.

| Paper | Code |
| --- | --- |
| Noisy broadcast `W_init = W_g + eps`, `eps ~ N(0, sigma^2 I)`, seed kept by the server | `fl/runner.py`: `eps_fn` (noise from the seed, regenerable per round and client) and the broadcast step `winit[(t, k)] = w + eps(t, k)` |
| Update `u = W_sub - W_init` | `fl/models.py` `train_local` returns it; forged by `fl/attacks.py` for free riders |
| S1 zero-update guard | `fl/detectors.py`, `SPPFD.step`: `nu == 0` |
| S2 probe projection: `max_j |sqrt(d) cos(u, eps^{r_j} - eps^{r_{j+1}})|` over the last 5 participations, flag if `> tau2` | `SPPFD._s2`, `PAIRS`, called from `step` |
| S3 peer-relative trend: alignment of `u` with `W_g^{t-1} - W_g^{t-2}`, robust z-score (median/MAD, 3-round pool), flag if `abs(z) > tau3` | `SPPFD.step` (`a`, `z3`, `awin`, `_robust_z`) |
| S4 temporal alignment: `sqrt(d) cos(u, u_prev)`, flag if `< tau4` | `SPPFD.step` (`s4`, `prev_u`) |
| Final flag (OR) and ranking score | end of `SPPFD.step` (`hit`, `comp`, `sc`) |
| Equal-weight aggregation, history update after detection | `fl/runner.py`: `det.step(ctx)`, then `hist`/`prev_u` updates, then `w = w + mean(updates)` |
| Thresholds `(3.023, 3.5, 2.326)`, `sigma = 0.005` | `fl/config.py` (`tau2`, `tau3`, `tau4`) and `experiments.py` |

## Names

Detector, attack and config IDs are the paper's names.

| Paper | Code |
| --- | --- |
| SPP-FD | detector `"spp_fd"`, class `SPPFD` |
| No-detection control | detector `"none"` |
| F1 zero, F2 uniform random, F3 decaying Gaussian | attacks `"F1"`, `"F2"`, `"F3"` |
| F4 received-model difference | attack `"F4"` |
| F5 noisy received-model difference | attack `"F5"` |
| F6 collusion | attack `"F6"` |
| F4-ref, F5-ref (noise-disclosure variants) | attacks `"F4-ref"`, `"F5-ref"` |
| F7 detector-aware stale reuse | attack `"F7"` (needs `attack_from=8`, full participation) |
| S2 history length H, S3 pooled rounds | `Cfg.history`, `Cfg.s3_rounds` |

F7 needs two distinct integer runtime seeds (`server_seed`, `attack_seed`, each below 2**40 and
different from `Cfg.seed`), passed to `fl.runner.run` or `fl.sweep.run_one`:

```python
cfg = Cfg(device="cpu", dataset="fashion", dist="iid", seed=1000, rounds=9, attack_from=8,
          attack="F7", detector="spp_fd", sigma=.005)
rec, accs, cost = run(cfg, server_seed=2**33 + 5, attack_seed=2**33 + 7)
```

F7 development candidates are not included.

The paper's raw results were produced with internal IDs. Mapping, for anyone comparing with them:
`nifd_cr` is `spp_fd`; attacks `f1 f2 f3 f6 f7 f8 f4ref f5ref` are `F1 F2 F3 F4 F5 F6 F4-ref F5-ref`;
`adaptive_stale_mix` with `lam=0` is `F7`; `warm` is `history`; `s3win` is `s3_rounds`.
Config hashes change with the names, so result folders are not interchangeable.

Defaults match the paper: 20 clients, 20% free riders, 30 rounds, attacks from round 6, SGD with
learning rate 0.05, batch 64, one local epoch, `sigma = 0.005`, thresholds
`(tau2, tau3, tau4) = (3.023, 3.5, 2.326)`, ten seeds (20 to 29).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.12
pip install -r requirements.txt
python -c "from torchvision import datasets as d; [f('data', train=t, download=True) for f in (d.FashionMNIST, d.CIFAR10) for t in (True, False)]"
```

Set `SPPFD_DATA=/path/to/data` to use another data directory.

## Usage

```bash
python tests/test_sppfd.py                         # CPU checks
python experiments.py smoke                        # dry run: lists the 72 smoke configs
python experiments.py smoke --run --device cuda:0  # trains them (12 rounds each)
```

Results go to `results_current/<source hash>/<config hash>/` (`summary.json`, `records.json`,
`round_metrics.csv`). `summary.json` holds recall, FPR, AUROC and recall at 1% FPR.

One run from Python:

```python
from fl.config import Cfg
from fl.runner import run
from fl.metrics import summarize_active

cfg = Cfg(device="cpu", dataset="fashion", dist="iid", seed=1000, rounds=8,
          attack="F4", detector="spp_fd", sigma=.005)
rec, accs, cost = run(cfg)
print(summarize_active(rec, cfg.attack_from)["recall"])
```

## Provenance and reproducibility

- This code is derived from the internal research harness that produced the paper's results. The
  main campaign ran on an earlier snapshot (source hash `841a62a82c9a9a6e`) that differs from the
  current harness only by added adaptive-attack code (the F7 runtime-seed path).
- The baseline-free tree was compared with the full harness on CPU: with identical configs and
  seeds, the client-round records, accuracies and cost fields were identical (see
  "Verification" below). GPU runs were not repeated for this repository.
- CPU and CUDA use different random streams, so numbers from a CPU run do not match the paper's GPU
  numbers bit for bit. Runs are deterministic on a given device and library version.
- The simulator does not enforce the secrecy of the server seed; seed secrecy is an operational
  assumption, as stated in the paper.

## Verification

Checked on CPU (FashionMNIST, 20 clients, 8 to 9 rounds, seed 1000, `sigma = 0.005`). A SHA-256
digest of the client-round records, accuracies, per-round metrics and cost fields (timings
excluded) was compared between the full internal harness (with its internal IDs) and this tree.
The 11 configs gave identical digests: SPP-FD under F1, F2, F3 (Dirichlet split), F4, F5 (with
filtering), F6, F4-ref, F5-ref and F7; SPP-FD without attack; and the no-detector control with
noise. For the comparison, renamed record fields were mapped back (`x_z3` to `x_g3`) and the
removed collusion-donor diagnostics were dropped. Against the frozen campaign snapshot
`841a62a82c9a9a6e`, four configs (F4, F6, F5 with filtering, no attack) also matched before the
renaming. Not checked: GPU runs, CIFAR-10, other seeds, and the paper's numbers themselves.

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Citation

Citation details will be added when the proceedings entry is available.
