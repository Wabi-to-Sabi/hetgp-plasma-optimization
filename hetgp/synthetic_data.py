"""
Synthetic process data generator.

Mimics a 2D process window (e.g., RF power x pressure)
with spatially varying noise — inspired by BF3 plasma research.
No proprietary data is used.
"""

import numpy as np
from typing import Tuple


def _response_surface(X: np.ndarray) -> np.ndarray:
    """
    Smooth ground-truth response function.
    Represents e.g. ion density or etch rate as a function of
    (normalized) process parameters.
    """
    x1, x2 = X[:, 0], X[:, 1]
    # Multi-modal surface with a ridge and a valley
    f = (
        1.5 * np.exp(-((x1 - 0.3) ** 2 + (x2 - 0.7) ** 2) / 0.1)
        + np.exp(-((x1 - 0.7) ** 2 + (x2 - 0.3) ** 2) / 0.15)
        - 0.5 * np.exp(-((x1 - 0.5) ** 2 + (x2 - 0.5) ** 2) / 0.3)
        + 0.3 * np.sin(3 * np.pi * x1) * np.cos(2 * np.pi * x2)
    )
    return f


def _noise_variance(X: np.ndarray, noise_scale: float = 0.15) -> np.ndarray:
    """
    Heteroscedastic noise variance as a function of input.

    Noise is higher near boundaries and in one corner region,
    mimicking real plasma measurement uncertainty patterns
    (e.g., edge effects in plasma chambers).
    """
    x1, x2 = X[:, 0], X[:, 1]
    # Boundary effect: higher noise near x=0 or x=1
    boundary_noise = noise_scale * (
        np.exp(-10 * x1) + np.exp(-10 * (1 - x1))
        + np.exp(-10 * x2) + np.exp(-10 * (1 - x2))
    )
    # Regional effect: higher noise in high-power low-pressure corner
    corner_noise = 0.08 * np.exp(-((x1 - 0.9) ** 2 + (x2 - 0.1) ** 2) / 0.05)
    return np.clip(boundary_noise + corner_noise + 0.01, 0.01, 1.0)


def generate_training_data(
    n_samples: int = 30,
    noise_scale: float = 0.15,
    random_seed: int = 42,
    domain: Tuple[float, float] = (0.0, 1.0),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate sparse training data with heteroscedastic noise.

    Parameters
    ----------
    n_samples : int
        Number of experimental observations (typically 20-40 for
        high-cost process experiments).
    noise_scale : float
        Base noise level.
    random_seed : int
        Reproducibility seed.
    domain : tuple
        Input domain (lo, hi) — same for both dimensions.

    Returns
    -------
    X : (n_samples, 2) array — normalized process parameters
    y : (n_samples,) array — noisy observations
    sigma_true : (n_samples,) array — true noise std at each point
    """
    rng = np.random.default_rng(random_seed)
    lo, hi = domain
    X = rng.uniform(lo, hi, size=(n_samples, 2))
    f_true = _response_surface(X)
    sigma_true = np.sqrt(_noise_variance(X, noise_scale))
    noise = rng.normal(0, sigma_true)
    y = f_true + noise
    return X, y, sigma_true


def generate_grid(resolution: int = 50) -> np.ndarray:
    """
    Generate a regular grid for visualization / prediction.

    Returns
    -------
    X_grid : (resolution^2, 2) array
    """
    x1 = np.linspace(0, 1, resolution)
    x2 = np.linspace(0, 1, resolution)
    xx1, xx2 = np.meshgrid(x1, x2)
    return np.column_stack([xx1.ravel(), xx2.ravel()])


def true_response_on_grid(resolution: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ground truth response and noise on a grid (for benchmarking).
    """
    X_grid = generate_grid(resolution)
    f_grid = _response_surface(X_grid)
    sigma_grid = np.sqrt(_noise_variance(X_grid))
    return f_grid, sigma_grid
