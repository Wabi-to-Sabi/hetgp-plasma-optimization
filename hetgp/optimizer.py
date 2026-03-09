"""
Bayesian Optimization loop using hetGP surrogate.

Acquisition function: Expected Improvement (EI)
Demonstrates efficient next-experiment suggestion
for high-cost process optimization.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import differential_evolution
from typing import Callable, Tuple, List


def expected_improvement(
    X: np.ndarray,
    model,
    y_best: float,
    xi: float = 0.01,
) -> np.ndarray:
    """
    Expected Improvement acquisition function.

    EI(x) = E[max(f(x) - y_best, 0)]

    With hetGP, the uncertainty σ(x) is spatially varying,
    so EI correctly reflects higher exploration value in
    high-uncertainty regions — unlike standard GP which uses
    a fixed noise estimate.

    Parameters
    ----------
    X : candidate points (n, d)
    model : fitted GP model with .predict() interface
    y_best : best observed value so far
    xi : exploration-exploitation trade-off (higher = more explore)

    Returns
    -------
    ei : Expected Improvement values (n,)
    """
    mu, sigma = model.predict(X)
    sigma = np.maximum(sigma, 1e-8)  # numerical stability
    Z = (mu - y_best - xi) / sigma
    ei = (mu - y_best - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
    ei[sigma < 1e-8] = 0.0
    return ei


def upper_confidence_bound(
    X: np.ndarray,
    model,
    beta: float = 2.0,
) -> np.ndarray:
    """
    Upper Confidence Bound (UCB) acquisition.
    Alternative to EI — good for exploration-heavy settings.
    """
    mu, sigma = model.predict(X)
    return mu + beta * sigma


def suggest_next_point(
    model,
    y_best: float,
    bounds: List[Tuple[float, float]],
    acquisition: str = "EI",
    n_restarts: int = 10,
    xi: float = 0.01,
    beta: float = 2.0,
) -> np.ndarray:
    """
    Suggest the next experiment point by maximizing the acquisition function.

    Uses differential evolution for global optimization of the
    acquisition surface.

    Returns
    -------
    x_next : (d,) array — suggested next process parameters
    """
    def neg_acquisition(x):
        X = x.reshape(1, -1)
        if acquisition == "EI":
            return -expected_improvement(X, model, y_best, xi=xi)[0]
        elif acquisition == "UCB":
            return -upper_confidence_bound(X, model, beta=beta)[0]
        else:
            raise ValueError(f"Unknown acquisition: {acquisition}")

    result = differential_evolution(
        neg_acquisition,
        bounds=bounds,
        seed=42,
        maxiter=200,
        tol=1e-6,
        popsize=15,
    )
    return result.x


class BayesianOptimizer:
    """
    Full Bayesian Optimization loop with hetGP surrogate.

    Usage
    -----
    >>> from hetgp.model import HetGPSklearn
    >>> from hetgp.synthetic_data import generate_training_data
    >>>
    >>> X_init, y_init, _ = generate_training_data(n_samples=15)
    >>> model = HetGPSklearn()
    >>> optimizer = BayesianOptimizer(model, bounds=[(0,1),(0,1)])
    >>> optimizer.initialize(X_init, y_init)
    >>> for _ in range(5):
    ...     x_next = optimizer.suggest()
    ...     y_next = my_experiment(x_next)   # your actual experiment
    ...     optimizer.update(x_next, y_next)
    >>> print("Best found:", optimizer.best_x, optimizer.best_y)
    """

    def __init__(
        self,
        model,
        bounds: List[Tuple[float, float]],
        acquisition: str = "EI",
        xi: float = 0.01,
    ):
        self.model = model
        self.bounds = bounds
        self.acquisition = acquisition
        self.xi = xi
        self.X_obs: List[np.ndarray] = []
        self.y_obs: List[float] = []

    def initialize(self, X_init: np.ndarray, y_init: np.ndarray) -> None:
        """Seed the optimizer with initial observations."""
        self.X_obs = list(X_init)
        self.y_obs = list(y_init)
        self._refit()

    def suggest(self) -> np.ndarray:
        """Suggest the next point to evaluate."""
        return suggest_next_point(
            self.model,
            y_best=self.best_y,
            bounds=self.bounds,
            acquisition=self.acquisition,
            xi=self.xi,
        )

    def update(self, x_new: np.ndarray, y_new: float) -> None:
        """Add a new observation and refit the surrogate."""
        self.X_obs.append(x_new)
        self.y_obs.append(y_new)
        self._refit()

    def _refit(self) -> None:
        X = np.array(self.X_obs)
        y = np.array(self.y_obs)
        self.model.fit(X, y)

    @property
    def best_y(self) -> float:
        return float(np.max(self.y_obs))

    @property
    def best_x(self) -> np.ndarray:
        idx = int(np.argmax(self.y_obs))
        return np.array(self.X_obs[idx])

    def convergence_history(self) -> np.ndarray:
        """Running maximum — shows optimization progress."""
        return np.maximum.accumulate(self.y_obs)
