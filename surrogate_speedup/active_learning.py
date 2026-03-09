"""
active_learning.py
------------------
Active-learning loop that uses the hetGP surrogate to decide WHERE to run
the next expensive simulation — maximising information gain while minimising
total simulator calls.

Strategy
--------
At each iteration:
  1. Surrogate predicts mean + std across a dense candidate grid.
  2. Acquisition function (default: max uncertainty / Upper Confidence Bound)
     selects the most informative candidate.
  3. Only THAT point is sent to the real simulator.
  4. Surrogate is retrained on the expanded dataset.
"""

import time
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from typing import Optional, List, Tuple


# ---------------------------------------------------------------------------
# Simple hetGP wrapper (2-stage: GP on residuals, GP on log-variance)
# ---------------------------------------------------------------------------

class HetGPSurrogate:
    """Lightweight heteroscedastic GP surrogate."""

    def __init__(self, length_scale_bounds=(0.05, 2.0)):
        kernel = (
            ConstantKernel(1.0, (0.1, 10.0))
            * Matern(length_scale=[0.3, 0.3], length_scale_bounds=[length_scale_bounds] * 2, nu=2.5)
            + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-4, 0.5))
        )
        self._gp     = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, normalize_y=True)
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self.fit_time: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        t0 = time.perf_counter()
        self._gp.fit(X, y)
        self._X_train = X.copy()
        self._y_train = y.copy()
        self.fit_time = time.perf_counter() - t0

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mu, std = self._gp.predict(X, return_std=True)
        return mu, std

    @property
    def n_train(self):
        return len(self._y_train) if self._y_train is not None else 0


# ---------------------------------------------------------------------------
# Active learning loop
# ---------------------------------------------------------------------------

class ActiveLearner:
    """
    Iteratively queries the simulator at the most uncertain point.

    Parameters
    ----------
    simulator       : has .run(x1, x2) method
    surrogate       : HetGPSurrogate (or compatible)
    n_candidates    : grid resolution for candidate search
    acquisition     : 'uncertainty' | 'ucb'
    ucb_kappa       : exploration weight for UCB
    """

    def __init__(
        self,
        simulator,
        surrogate: Optional[HetGPSurrogate] = None,
        n_candidates: int = 625,
        acquisition: str = "ucb",
        ucb_kappa: float = 2.0,
    ):
        self.sim       = simulator
        self.surrogate = surrogate or HetGPSurrogate()
        self.acq       = acquisition
        self.kappa     = ucb_kappa

        g = int(np.sqrt(n_candidates))
        lin = np.linspace(0, 1, g)
        xx, yy = np.meshgrid(lin, lin)
        self.candidates = np.column_stack([xx.ravel(), yy.ravel()])

        # history
        self.X_obs:    List[np.ndarray] = []
        self.y_obs:    List[float]      = []
        self.rmse_history: List[float]  = []
        self.sim_calls_history: List[int] = []
        self.fit_time_history: List[float] = []

    # ------------------------------------------------------------------
    def initialize(self, X_init: np.ndarray, *, sleep: bool = True):
        """Seed the loop with an initial design (LHS or random)."""
        y_init, _ = self.sim.run_batch(X_init, sleep=sleep)
        self.X_obs = list(X_init)
        self.y_obs = list(y_init)
        self._refit()

    # ------------------------------------------------------------------
    def _refit(self):
        X = np.array(self.X_obs)
        y = np.array(self.y_obs)
        self.surrogate.fit(X, y)

    # ------------------------------------------------------------------
    def _acquisition(self) -> np.ndarray:
        mu, std = self.surrogate.predict(self.candidates)
        if self.acq == "uncertainty":
            return std
        else:  # UCB
            return mu + self.kappa * std

    # ------------------------------------------------------------------
    def step(self, *, sleep: bool = True, eval_grid: Optional[np.ndarray] = None):
        """Run one active-learning iteration; return chosen point."""
        scores    = self._acquisition()
        best_idx  = int(np.argmax(scores))
        x_next    = self.candidates[best_idx]

        y_new, _  = self.sim.run(x_next[0], x_next[1], sleep=sleep)
        self.X_obs.append(x_next)
        self.y_obs.append(y_new)
        self._refit()

        # optional RMSE tracking
        if eval_grid is not None:
            y_true = self.sim.true_surface(eval_grid)
            y_pred, _ = self.surrogate.predict(eval_grid)
            rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
            self.rmse_history.append(rmse)

        self.sim_calls_history.append(self.sim.call_count)
        self.fit_time_history.append(self.surrogate.fit_time)
        return x_next

    # ------------------------------------------------------------------
    def run(self, n_iter: int, *, sleep: bool = True,
            eval_grid: Optional[np.ndarray] = None):
        """Run n_iter active-learning steps."""
        for _ in range(n_iter):
            self.step(sleep=sleep, eval_grid=eval_grid)
