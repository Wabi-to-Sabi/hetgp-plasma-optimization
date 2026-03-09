"""
SHAP-based explanations for process optimization models.

Answers the key question for process engineers:
"Which process parameter matters most — and how?"
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


def compute_shap_values(
    model,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    n_background: int = 50,
) -> np.ndarray:
    """
    Compute SHAP values using KernelExplainer.

    Works with any model that has a .predict()[0] interface,
    including both StandardGP and HetGP wrappers.

    Parameters
    ----------
    model : fitted model with predict(X) -> (mu, sigma)
    X_background : background dataset for SHAP baseline
    X_explain : points to explain
    n_background : number of background samples (kmeans summary)

    Returns
    -------
    shap_values : (n_explain, n_features) array
    """
    if not HAS_SHAP:
        raise ImportError("shap is required. Install with: pip install shap")

    def predict_mean(X):
        mu, _ = model.predict(X)
        return mu

    background = shap.kmeans(X_background, min(n_background, len(X_background)))
    explainer = shap.KernelExplainer(predict_mean, background)
    shap_values = explainer.shap_values(X_explain, nsamples=100, silent=True)
    return np.array(shap_values)


def plot_shap_summary(
    shap_values: np.ndarray,
    X_explain: np.ndarray,
    feature_names: Optional[List[str]] = None,
    title: str = "SHAP Feature Importance",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Summary bar plot of SHAP feature importance.
    Shows mean |SHAP| per feature.
    """
    if feature_names is None:
        feature_names = [f"Param {i+1}" for i in range(X_explain.shape[1])]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ["#2196F3" if v == max(mean_abs_shap) else "#90CAF9" for v in mean_abs_shap]
    bars = ax.barh(feature_names, mean_abs_shap, color=colors)
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_shap_dependence(
    shap_values: np.ndarray,
    X_explain: np.ndarray,
    feature_idx: int = 0,
    feature_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Dependence plot: SHAP value vs feature value for one parameter.
    Reveals non-linear effects and interaction structure.
    """
    if feature_names is None:
        feature_names = [f"Param {i+1}" for i in range(X_explain.shape[1])]

    fname = feature_names[feature_idx]
    x_vals = X_explain[:, feature_idx]
    s_vals = shap_values[:, feature_idx]

    fig, ax = plt.subplots(figsize=(6, 4))
    sc = ax.scatter(x_vals, s_vals, c=s_vals, cmap="RdBu_r", alpha=0.8, s=50)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel(fname, fontsize=11)
    ax.set_ylabel(f"SHAP({fname})", fontsize=11)
    ax.set_title(f"SHAP Dependence: {fname}", fontsize=13, fontweight="bold")
    plt.colorbar(sc, ax=ax, label="SHAP value")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def uncertainty_vs_shap(
    model,
    X: np.ndarray,
    shap_values: np.ndarray,
    feature_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Scatter plot: predictive uncertainty vs total |SHAP|.

    Unique to hetGP workflow — shows where the model is
    uncertain AND which features drive predictions there.
    High uncertainty + high SHAP = priority region for next experiment.
    """
    _, sigma = model.predict(X)
    total_shap = np.abs(shap_values).sum(axis=1)

    fig, ax = plt.subplots(figsize=(6, 4))
    sc = ax.scatter(total_shap, sigma, alpha=0.7, c=sigma, cmap="YlOrRd", s=60)
    ax.set_xlabel("Total |SHAP| (feature influence)", fontsize=11)
    ax.set_ylabel("Predictive uncertainty σ(x)", fontsize=11)
    ax.set_title("Where to experiment next?\n(high σ + high SHAP = priority)", fontsize=12)
    plt.colorbar(sc, ax=ax, label="σ(x)")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
