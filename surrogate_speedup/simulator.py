"""
simulator.py
------------
Mock high-fidelity simulator that mimics a 2-input BF3 plasma PIC-MCC run.

Key behaviours
--------------
* Each call sleeps for `cost_seconds` (default 2.0 s) to represent real compute.
* The underlying physics-like function is deterministic given `random_seed`.
* Heteroscedastic noise: σ(x) is larger near chamber boundaries.
* Call counter exposed so callers can track total simulator invocations.
"""

import time
import numpy as np
from typing import Optional, Tuple


class PICMCCSimulator:
    """
    Mock PIC-MCC simulator for a 2-parameter BF3 process window.

    Parameters
    ----------
    cost_seconds : float
        Wall-clock delay per evaluation (simulates real compute cost).
    noise_scale  : float
        Global noise multiplier.
    random_seed  : int
        Seed for reproducibility.
    """

    def __init__(
        self,
        cost_seconds: float = 2.0,
        noise_scale: float = 1.0,
        random_seed: int = 42,
    ):
        self.cost_seconds = cost_seconds
        self.noise_scale  = noise_scale
        self.rng          = np.random.default_rng(random_seed)
        self.call_count   = 0
        self.total_time   = 0.0

    # ------------------------------------------------------------------
    def _physics_response(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Deterministic 'physics' — mimics ion-flux response surface."""
        return (
            1.8 * np.exp(-((x1 - 0.30) ** 2 + (x2 - 0.65) ** 2) / 0.08)
            + 1.2 * np.exp(-((x1 - 0.70) ** 2 + (x2 - 0.35) ** 2) / 0.12)
            - 0.4 * np.exp(-((x1 - 0.50) ** 2 + (x2 - 0.50) ** 2) / 0.25)
            + 0.25 * np.sin(3.2 * np.pi * x1) * np.cos(2.1 * np.pi * x2)
        )

    def _noise_std(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Heteroscedastic noise — larger near boundaries and high-power corner."""
        boundary = 0.12 * (
            np.exp(-9 * x1)
            + np.exp(-9 * (1 - x1))
            + np.exp(-9 * x2)
            + np.exp(-9 * (1 - x2))
        )
        hotspot = 0.10 * np.exp(-((x1 - 0.9) ** 2 + (x2 - 0.1) ** 2) / 0.04)
        return self.noise_scale * np.sqrt(np.clip(boundary + hotspot + 0.008, 0.005, 1.0))

    # ------------------------------------------------------------------
    def run(
        self, x1: float, x2: float, *, sleep: bool = True
    ) -> Tuple[float, float]:
        """
        Run a single simulation point.

        Parameters
        ----------
        x1, x2 : float  — normalised process parameters in [0, 1]
        sleep   : bool  — if False, skip the artificial delay (for unit tests)

        Returns
        -------
        (y_obs, sigma_true) : observed output + ground-truth noise std
        """
        t0 = time.perf_counter()
        if sleep:
            time.sleep(self.cost_seconds)

        x1a, x2a = np.asarray(x1), np.asarray(x2)
        mu    = float(self._physics_response(x1a, x2a))
        sigma = float(self._noise_std(x1a, x2a))
        y_obs = mu + self.rng.normal(0.0, sigma)

        elapsed = time.perf_counter() - t0
        self.call_count += 1
        self.total_time  += elapsed
        return y_obs, sigma

    def run_batch(
        self, X: np.ndarray, *, sleep: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run a batch of points; returns (y_obs, sigma_true) arrays."""
        ys, ss = zip(*[self.run(x[0], x[1], sleep=sleep) for x in X])
        return np.array(ys), np.array(ss)

    def true_surface(self, X: np.ndarray) -> np.ndarray:
        """Noiseless ground truth for evaluation (no sleep, no counter)."""
        return self._physics_response(X[:, 0], X[:, 1])

    def true_noise(self, X: np.ndarray) -> np.ndarray:
        """Ground-truth noise std for evaluation."""
        return self._noise_std(X[:, 0], X[:, 1])

    def reset(self):
        self.call_count = 0
        self.total_time = 0.0
