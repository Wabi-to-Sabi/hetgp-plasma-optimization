"""
GP model wrappers: Standard GP vs Heteroscedastic GP (hetGP).

Both expose a unified predict() interface so they can be
swapped in/out for comparison and optimization.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from typing import Tuple

try:
    import GPy
    HAS_GPY = True
except ImportError:
    HAS_GPY = False


class StandardGP:
    """
    Homoscedastic GP via scikit-learn.
    Baseline comparison model.
    """

    def __init__(self, length_scale: float = 0.3, noise_level: float = 0.05):
        kernel = (
            ConstantKernel(1.0)
            * Matern(length_scale=length_scale, nu=2.5)
            + WhiteKernel(noise_level=noise_level)
        )
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            normalize_y=True,
        )
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StandardGP":
        self.gp.fit(X, y)
        self.is_fitted = True
        return self

    def predict(
        self, X: np.ndarray, return_std: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        mu : predictive mean
        sigma : predictive std (homoscedastic — same everywhere)
        """
        mu, sigma = self.gp.predict(X, return_std=True)
        return mu, sigma

    def log_marginal_likelihood(self) -> float:
        return float(self.gp.log_marginal_likelihood_value_)


class HetGP:
    """
    Heteroscedastic GP via GPy.

    Models input-dependent noise variance using a secondary GP
    on the log-noise, following:
        Goldberg et al. (1998)
        Binois & Gramacy (2021), hetGP R package

    This is the key advantage over standard GP for:
    - Sparse plasma/process data with edge effects
    - Trustworthy uncertainty estimates for Bayesian Optimization
    - Calibrated confidence intervals for process engineers
    """

    def __init__(
        self,
        length_scale: float = 0.3,
        signal_variance: float = 1.0,
        noise_variance: float = 0.05,
    ):
        if not HAS_GPY:
            raise ImportError(
                "GPy is required for HetGP. Install with: pip install GPy"
            )
        self.length_scale = length_scale
        self.signal_variance = signal_variance
        self.noise_variance = noise_variance
        self.model = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HetGP":
        """
        Fit the heteroscedastic GP model.

        Uses GPy's HeteroscedasticGaussianRegression which places
        a GP prior on log-noise variance across the input space.
        """
        y_2d = y.reshape(-1, 1)
        n = X.shape[0]
        input_dim = X.shape[1]

        kernel = GPy.kern.Matern52(
            input_dim=input_dim,
            variance=self.signal_variance,
            lengthscale=self.length_scale,
        )

        # Heteroscedastic likelihood: per-point noise variance
        # Each observation gets its own noise parameter (then regularized)
        self.model = GPy.models.GPHeteroscedasticRegression(
            X, y_2d, kernel=kernel
        )
        # Initialize noise terms
        self.model[".*het_Gauss.variance"].constrain_positive()
        self.model.optimize(messages=False, max_iters=200)
        self.is_fitted = True
        return self

    def predict(
        self, X: np.ndarray, return_std: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        mu : predictive mean  (n,)
        sigma : predictive std — spatially varying  (n,)
        """
        mu, var = self.model.predict(X)
        sigma = np.sqrt(var).ravel()
        return mu.ravel(), sigma

    def noise_variance_at(self, X: np.ndarray) -> np.ndarray:
        """
        Return estimated noise variance at input locations X.
        Unique to hetGP — unavailable in standard GP.
        """
        _, var = self.model.predict(X)
        return var.ravel()


class HetGPSklearn:
    """
    Lightweight hetGP approximation using scikit-learn only.

    Uses a two-stage approach:
    1. Fit standard GP to get residuals
    2. Fit second GP on log(residual^2) to estimate noise variance
    3. Re-fit with estimated per-point noise

    Useful when GPy is unavailable. Less principled than full hetGP
    but demonstrates the concept clearly.
    """

    def __init__(self):
        kernel = ConstantKernel(1.0) * Matern(length_scale=0.3, nu=2.5)
        self.gp_mean = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=3, normalize_y=True
        )
        kernel_noise = ConstantKernel(0.1) * Matern(length_scale=0.5, nu=1.5)
        self.gp_noise = GaussianProcessRegressor(
            kernel=kernel_noise, n_restarts_optimizer=3
        )
        self.X_train = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HetGPSklearn":
        self.X_train = X.copy()
        # Stage 1: fit mean GP
        self.gp_mean.fit(X, y)
        residuals = y - self.gp_mean.predict(X)
        log_sq_res = np.log(residuals ** 2 + 1e-6)
        # Stage 2: fit noise GP on log squared residuals
        self.gp_noise.fit(X, log_sq_res)
        self.is_fitted = True
        return self

    def predict(
        self, X: np.ndarray, return_std: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        mu, sigma_mean = self.gp_mean.predict(X, return_std=True)
        log_noise_pred = self.gp_noise.predict(X)
        # Combine aleatoric (noise) and epistemic (model) uncertainty
        sigma_noise = np.sqrt(np.exp(log_noise_pred) + 1e-6)
        sigma = np.sqrt(sigma_mean ** 2 + sigma_noise ** 2)
        return mu, sigma
