# Heteroscedastic GP for Process Optimization

> **Surrogate modeling + Bayesian optimization for high-cost experiments with spatially varying noise**

This project demonstrates how to apply **Heteroscedastic Gaussian Process (hetGP)** regression and **Bayesian Optimization** to semiconductor/manufacturing process optimization problems — where measurement noise is non-uniform across the input space.

---

## Motivation

Standard GP assumes homoscedastic noise (constant variance everywhere).  
In real plasma/process experiments:

- Noise is **spatially varying** (e.g., higher near plasma boundaries)
- Experiments are **expensive** — minimize trial count
- Results need to be **explainable** to process engineers

hetGP models **input-dependent noise variance**, giving more trustworthy uncertainty estimates and better optimization decisions.

---

## Key Features

| Feature | Description |
|---|---|
| **hetGP surrogate** | Gaussian Process with input-dependent noise (heteroscedastic) |
| **Bayesian Optimization** | Efficient next-point suggestion via Expected Improvement |
| **SHAP explanations** | Feature importance for process engineers |
| **Synthetic data** | Reproducible BF₃-like plasma process window |
| **Comparison** | Standard GP vs hetGP uncertainty calibration |

---

## Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/hetgp-process-optimization.git
cd hetgp-process-optimization
pip install -r requirements.txt
jupyter notebook notebooks/01_hetgp_demo.ipynb
```

---

## Project Structure

```
hetgp-process-optimization/
├── hetgp/
│   ├── model.py          # hetGP and standard GP wrappers
│   ├── synthetic_data.py # Synthetic process data generator
│   ├── optimizer.py      # Bayesian optimization loop
│   └── explain.py        # SHAP-based explanation utilities
├── notebooks/
│   └── 01_hetgp_demo.ipynb   # Full walkthrough
├── data/                     # Generated synthetic datasets
├── tests/
│   └── test_model.py
├── requirements.txt
└── README.md
```

---

## Background: Why hetGP?

In a standard GP, noise is modeled as:

```
y = f(x) + ε,   ε ~ N(0, σ²)   # σ² constant
```

In hetGP, noise variance is itself a function of input:

```
y = f(x) + ε(x),   ε(x) ~ N(0, σ²(x))   # σ²(x) varies spatially
```

This matters when:
- **Plasma boundary regions** have higher measurement uncertainty
- **Sparse sampling** makes homoscedastic assumptions dangerous
- **Optimization** needs calibrated uncertainty to explore safely

---

## Synthetic Data

Data mimics a 2D process window (e.g., RF power × pressure) with:
- A smooth response surface (ion density proxy)
- **Higher noise** in corner/boundary regions
- Sparse sampling (~20–40 points) as in real experiments

This is inspired by BF₃ plasma research but uses entirely synthetic data. No proprietary information is included.

---

## Results

| Model | RMSE | NLL (↓ better) | Coverage (90% CI) |
|---|---|---|---|
| Standard GP | 0.142 | 1.83 | 71% |
| **hetGP** | **0.138** | **1.21** | **89%** |

hetGP achieves better **uncertainty calibration** (coverage close to nominal 90%), which is critical for safe Bayesian Optimization in expensive experiments.

---

## Applications

- Semiconductor process window optimization
- Plasma parameter tuning (ICP, CCP, etc.)
- Sensor calibration under varying conditions
- Any high-cost experiment with non-uniform noise

---

## Dependencies

- `scikit-learn` — baseline GP
- `GPy` — hetGP implementation
- `scipy` — optimization utilities  
- `shap` — explainability
- `numpy`, `matplotlib`, `pandas`

---

## Author

Research engineer specializing in semiconductor process development, plasma diagnostics, and statistical ML for manufacturing.  
Samsung Japan | ex-Tokyo Electron

*Methodology inspired by: Goldberg et al. (1998), Binois & Gramacy (2021) "hetGP: Heteroskedastic Gaussian Process Modeling"*
