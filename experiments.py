#!/usr/bin/env python
"""SPP-FD experiment grids. Default is a dry run; pass --run to train."""
import argparse
from dataclasses import replace

from fl.config import Cfg
from fl.sweep import grid, main

BASE = Cfg(device="cuda:0", tag="native-broadcast-v1")
# Attack and detector IDs are the paper's names. The detector-aware F7 needs registered runtime
# seeds (see README) and is not in these grids.
ATTACKS = ["none", "F1", "F2", "F3", "F4-ref", "F5-ref", "F4", "F5", "F6"]
DETS = ["none", "spp_fd"]           # "none" = no-detection control
SEEDS = list(range(20, 30))


def comparison_grid(**axes):
    axes = dict(dataset=["fashion", "cifar10"], dist=["iid", "dir0.1"],
                detector=DETS, attack=ATTACKS, seed=SEEDS) | axes
    # Only SPP-FD perturbs the broadcast models (sigma=.005); the control sees the original model.
    return [replace(c, sigma=.005 if c.detector == "spp_fd" else 0.)
            for c in grid(BASE, **axes)]


def noise_controls(seeds=SEEDS, rounds=30):
    """Clean runs with server noise but no detector: accuracy cost of the noise."""
    return grid(replace(BASE, detector="none", sigma=.005), dataset=["fashion", "cifar10"],
                dist=["iid", "dir0.1"], seed=seeds, rounds=[rounds])


EXP = {
    "main": lambda: comparison_grid(dataset=["fashion"]),
    "main_cifar": lambda: comparison_grid(dataset=["cifar10"]),
    "filt": lambda: comparison_grid(detector=DETS[1:], filt=[True]),
    "noise_cost": noise_controls,
    "smoke": lambda: comparison_grid(seed=[1000], rounds=[12]),
    "smoke_filt": lambda: comparison_grid(seed=[1000], rounds=[12], detector=DETS[1:],
                                          attack=["none", "F5-ref", "F5", "F6"], filt=[True]),
    "smoke_noise_cost": lambda: noise_controls([1000], 12),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exp", choices=EXP)
    parser.add_argument("shard", type=int, nargs="?", default=0)
    parser.add_argument("nshard", type=int, nargs="?", default=1)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--device", default=BASE.device, help="torch device, e.g. cuda:0 or cpu")
    args = parser.parse_args()
    if not 0 <= args.shard < args.nshard:
        parser.error("require 0 <= shard < nshard")
    cfgs = [replace(c, device=args.device) for c in EXP[args.exp]()]
    for cfg in cfgs:
        cfg.validate_protocol()
    print(f"{args.exp}: {len(cfgs)} configs, shard {args.shard}/{args.nshard}, "
          f"mode={'run' if args.run else 'dry-run'}", flush=True)
    if args.run:
        main(cfgs, args.shard, args.nshard)
