"""Experiment config: a frozen dataclass hashed to an ID. Identical configs share a result directory."""
import hashlib
import json
import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Cfg:
    # --- environment ---
    dataset: str = "fashion"          # fashion | cifar10
    device: str = "cuda:0"            # training device
    seed: int = 0
    # --- FL ---
    n_client: int = 20
    fr_frac: float = 0.20
    dist: str = "iid"                 # iid | dir0.5 | dir0.1
    rounds: int = 30
    attack_from: int = 6
    participation: float = 1.0
    local_epoch: int = 1
    batch: int = 64
    lr: float = 0.05
    momentum: float = 0.0             # >0 keeps a buffer across rounds
    # --- challenge ---
    sigma: float | None = None        # auto: SPP-FD .005, others 0; serialized explicitly in grids
    # --- attack ---
    attack: str = "none"
    lam: float = 1.0
    kappa: float = 0.5
    r_rand: float = 1e-3
    # --- detector ---
    detector: str = "spp_fd"          # none | spp_fd
    filt: bool = False
    history: int = 5                  # S2 history length H (participations)
    s3_rounds: int = 3                # rounds pooled for S3 peer statistics
    signals: str = "S1,S2,S3,S4"      # for ablations
    tau2: float = 3.023
    tau3: float = 3.5
    tau4: float = 2.326
    tag: str = ""

    @property
    def noise_sigma(self) -> float:
        return self.sigma if self.sigma is not None else (0.005 if self.detector == "spp_fd" else 0.0)

    def validate_protocol(self):
        if self.attack == "F7" and (self.participation != 1.0 or self.attack_from != 8):
            raise ValueError("F7 requires full participation and attack_from=8")
        if not math.isfinite(self.noise_sigma) or self.noise_sigma < 0:
            raise ValueError("sigma must be finite and nonnegative")
        if self.noise_sigma > 0 and self.detector != "spp_fd" and not (
                self.detector == "none" and self.attack == "none"):
            raise ValueError("Server perturbation is restricted to SPP-FD and clean no-detector controls")

    def hash(self) -> str:
        d = asdict(self)
        d.pop("device")               # only the logical config ID is shared; the cache also checks the exact device
        return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]

    def pairing_hash(self) -> str:
        """ID grouping runs with the same exogenous randomness, data and attack condition (paired design)."""
        d = asdict(self)
        for key in ("device", "detector", "filt", "history", "s3_rounds", "signals",
                    "tau2", "tau3", "tau4", "sigma"):
            d.pop(key)
        return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]

    def trajectory_hash(self) -> str:
        """Unfiltered runs share a trajectory only when the exogenous conditions and the delivery protocol both match."""
        if not self.filt:
            d = dict(pairing=self.pairing_hash(), sigma=self.noise_sigma)
            return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]
        d = asdict(self)
        d.pop("device")
        return hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:12]

    def to_dict(self):
        return asdict(self)
