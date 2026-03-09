"""
benchmark.py
------------
Compares three strategies head-to-head (no real sleeping — uses sleep=False):

  A) Full-grid scan     — run the simulator at every grid point (brute force)
  B) Random sampling    — pick n points at random, fit GP, predict elsewhere
  C) Active learning    — use surrogate to choose where to simulate next

Reports wall-clock equivalents, RMSE, and call counts.
"""

import time
import numpy as np
from .simulator import PICMCCSimulator
from .active_learning import HetGPSurrogate, ActiveLearner


def run_benchmark(
    grid_size: int = 20,
    n_initial: int = 10,
    n_active_iter: int = 20,
    cost_per_sim: float = 120.0,   # seconds per real PIC-MCC run (2 min)
    random_seed: int = 0,
) -> dict:
    """
    Execute all three strategies and return a results dict.

    Parameters
    ----------
    grid_size     : full-grid side length  (grid_size² total runs for strategy A)
    n_initial     : initial random samples for strategies B and C
    n_active_iter : additional AL iterations for strategy C
    cost_per_sim  : assumed wall-clock seconds for ONE real simulation
    random_seed   : reproducibility seed
    """
    rng = np.random.default_rng(random_seed)
    g   = np.linspace(0, 1, grid_size)
    gg  = np.stack(np.meshgrid(g, g), axis=-1).reshape(-1, 2)   # (N, 2)

    sim = PICMCCSimulator(cost_seconds=0.0, random_seed=random_seed)

    # ── evaluation grid (dense, for RMSE) ─────────────────────────────
    g_eval = np.linspace(0, 1, 50)
    eval_grid = np.stack(np.meshgrid(g_eval, g_eval), axis=-1).reshape(-1, 2)
    y_true_eval = sim.true_surface(eval_grid)

    results = {}

    # ── Strategy A: Full-grid scan ─────────────────────────────────────
    sim.reset()
    t0 = time.perf_counter()
    y_full, _ = sim.run_batch(gg, sleep=False)
    surr_a = HetGPSurrogate()
    surr_a.fit(gg, y_full)
    mu_a, _ = surr_a.predict(eval_grid)
    t_a = time.perf_counter() - t0

    results["full_grid"] = {
        "n_calls"      : sim.call_count,
        "rmse"         : float(np.sqrt(np.mean((mu_a - y_true_eval) ** 2))),
        "wall_equiv_h" : sim.call_count * cost_per_sim / 3600,
        "label"        : f"Full-grid ({grid_size}×{grid_size})",
    }

    # ── Strategy B: Random sampling ────────────────────────────────────
    sim.reset()
    idx_rand  = rng.choice(len(gg), size=n_initial + n_active_iter, replace=False)
    X_rand    = gg[idx_rand]
    t0 = time.perf_counter()
    y_rand, _ = sim.run_batch(X_rand, sleep=False)
    surr_b = HetGPSurrogate()
    surr_b.fit(X_rand, y_rand)
    mu_b, _ = surr_b.predict(eval_grid)
    t_b = time.perf_counter() - t0

    results["random"] = {
        "n_calls"      : sim.call_count,
        "rmse"         : float(np.sqrt(np.mean((mu_b - y_true_eval) ** 2))),
        "wall_equiv_h" : sim.call_count * cost_per_sim / 3600,
        "label"        : f"Random ({n_initial + n_active_iter} pts)",
    }

    # ── Strategy C: Active learning ────────────────────────────────────
    sim.reset()
    X_init = rng.uniform(0, 1, size=(n_initial, 2))
    learner = ActiveLearner(sim, acquisition="ucb")
    t0 = time.perf_counter()
    learner.initialize(X_init, sleep=False)
    learner.run(n_active_iter, sleep=False, eval_grid=eval_grid)
    mu_c, _ = learner.surrogate.predict(eval_grid)
    t_c = time.perf_counter() - t0

    results["active"] = {
        "n_calls"      : sim.call_count,
        "rmse"         : float(np.sqrt(np.mean((mu_c - y_true_eval) ** 2))),
        "wall_equiv_h" : sim.call_count * cost_per_sim / 3600,
        "label"        : f"Active Learning ({n_initial}+{n_active_iter})",
        "rmse_curve"   : learner.rmse_history,
    }

    results["_meta"] = {
        "cost_per_sim_s": cost_per_sim,
        "eval_grid"     : eval_grid,
        "y_true_eval"   : y_true_eval,
        "surr_a_pred"   : mu_a,
        "surr_b_pred"   : mu_b,
        "surr_c_pred"   : mu_c,
        "X_random"      : X_rand,
        "X_active"      : np.array(learner.X_obs),
        "gg"            : gg,
        "grid_size"     : grid_size,
        "sim"           : sim,
    }
    return results
