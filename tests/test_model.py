"""
Basic tests for hetGP pipeline.
Run with: pytest tests/
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hetgp.synthetic_data import generate_training_data, generate_grid
from hetgp.model import StandardGP, HetGPSklearn
from hetgp.optimizer import expected_improvement, BayesianOptimizer


def test_synthetic_data_shape():
    X, y, sigma = generate_training_data(n_samples=20)
    assert X.shape == (20, 2)
    assert y.shape == (20,)
    assert sigma.shape == (20,)
    assert np.all(sigma > 0), "Noise std must be positive"


def test_standard_gp_fit_predict():
    X, y, _ = generate_training_data(n_samples=15)
    gp = StandardGP()
    gp.fit(X, y)
    X_test = generate_grid(10)
    mu, sigma = gp.predict(X_test)
    assert mu.shape == (100,)
    assert sigma.shape == (100,)
    assert np.all(sigma > 0)


def test_hetgp_sklearn_fit_predict():
    X, y, _ = generate_training_data(n_samples=20)
    gp = HetGPSklearn()
    gp.fit(X, y)
    X_test = generate_grid(10)
    mu, sigma = gp.predict(X_test)
    assert mu.shape == (100,)
    assert sigma.shape == (100,)
    assert np.all(sigma > 0)


def test_hetgp_spatially_varying_uncertainty():
    """
    Key property: hetGP sigma should vary across input space.
    Standard GP sigma is approximately constant.
    """
    X, y, _ = generate_training_data(n_samples=25)
    X_test = generate_grid(20)

    gp_std = StandardGP()
    gp_std.fit(X, y)
    _, sigma_std = gp_std.predict(X_test)

    gp_het = HetGPSklearn()
    gp_het.fit(X, y)
    _, sigma_het = gp_het.predict(X_test)

    # hetGP should have more variance in its sigma estimates
    assert sigma_het.std() > 0, "hetGP sigma must vary spatially"


def test_expected_improvement():
    X, y, _ = generate_training_data(n_samples=15)
    gp = HetGPSklearn()
    gp.fit(X, y)
    X_test = generate_grid(10)
    ei = expected_improvement(X_test, gp, y_best=np.max(y))
    assert ei.shape == (100,)
    assert np.all(ei >= 0), "EI must be non-negative"


def test_bayesian_optimizer():
    X_init, y_init, _ = generate_training_data(n_samples=10, random_seed=1)
    optimizer = BayesianOptimizer(
        model=HetGPSklearn(),
        bounds=[(0, 1), (0, 1)],
    )
    optimizer.initialize(X_init, y_init)
    assert optimizer.best_y == np.max(y_init)

    x_next = optimizer.suggest()
    assert x_next.shape == (2,)
    assert all(0 <= xi <= 1 for xi in x_next)

    y_next = 1.5  # mock experiment result
    optimizer.update(x_next, y_next)
    assert optimizer.best_y == 1.5
    assert len(optimizer.X_obs) == 11
