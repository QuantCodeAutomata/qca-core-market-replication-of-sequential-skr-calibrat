# Sequential Kernel Reconstruction (SKR) Calibration

## Overview

This repository implements **Sequential Kernel Reconstruction (SKR)**, a non-parametric calibration framework for stochastic volatility models. SKR calibrates a reference stochastic process (Heston, 3/2, Bergomi-1F, or Bass) to a target implied volatility surface — here the SPX surface dated **23 June 2020** — by iteratively reweighting Monte Carlo paths via a kernel-based transport step.

The key idea is to find a law-preserving map from the reference model marginals to the target risk-neutral marginals encoded by the ESSVI smile surface, without re-simulating the process. This yields a calibrated model that:

- Exactly matches the target marginal distributions at all calibration maturities.
- Preserves the martingale property of the price process.
- Retains the path-space structure of the reference model (e.g. volatility clustering, leverage effect).

---

## Mathematical Background

### ESSVI Smile Parametrization

The target implied volatility surface is parametrized via the **Extended SVI (ESSVI)** formula. For log-moneyness `k = log(K/F)`, the total implied variance is:

```
w(k) = (theta/2) * (1 + rho*phi*k + sqrt((phi*k + rho)^2 + (1 - rho^2)))
```

This parametrization is guaranteed arbitrage-free within each smile slice and admits a closed-form risk-neutral density via the Breeden-Litzenberger relation.

### SKR Algorithm

Given a reference process `(Y_t, L_t)` simulated on a discrete time grid `t_1 < ... < t_M`:

1. **Marginal extraction**: Extract the empirical marginal `mu_t^ref` of `Y_t` under the reference model for each target maturity `t`.
2. **Target marginals**: Compute the risk-neutral marginal `mu_t^target` from the ESSVI surface via inverse-CDF sampling.
3. **Kernel transport**: For each maturity, solve a kernel regression problem that maps `Y_t^ref` to `Y_t^target` while preserving the joint law of `(Y_t, L_t)`.
4. **Sequential update**: Apply the transport maps sequentially across maturities, propagating corrections forward in time.

### Reference Processes

| Model       | Description                                     |
|-------------|-------------------------------------------------|
| Heston      | Classical CIR variance, affine structure        |
| 3/2         | Non-affine variance model with fat tails        |
| Bergomi-1F  | One-factor rough-vol proxy, exponential OU      |
| Bass        | Log-normal baseline (geometric Brownian motion) |

Parameters are taken from **Table 1** of the SKR paper and pre-calibrated to SPX.

---

## Directory Structure

```
project/
├── data/
│   └── load_data.py          # Data loading and surface construction
├── src/
│   ├── __init__.py
│   ├── utils.py              # Black-Scholes utilities, statistical tools
│   ├── smile_surface.py      # ESSVI smile surface implementation
│   └── processes.py          # Reference stochastic processes
├── exp/                      # Experiment scripts (exp1.py ... exp5.py)
├── tests/
│   └── __init__.py
├── results/                  # Output figures and CSV results
├── README.md
└── .gitignore
```

---

## Installation

All dependencies are standard scientific Python packages:

```bash
pip install numpy scipy matplotlib pandas
```

Python **3.9+** is required.

---

## Experiments

The five experiments validate and demonstrate the SKR calibration framework:

### Experiment 1 — Target Surface Verification (`exp/exp1_surface_verification.py`)

Plots the ESSVI implied volatility smile for all five maturities (T = 0.25, 0.5, 1.0, 2.0, 3.0 years). Verifies:
- Absence of static arbitrage (butterfly and calendar spread conditions).
- Convex ordering of marginal distributions across maturities.
- Log-moneyness range: [-0.5, 0.3] for each slice.

**Run:**
```bash
python exp/exp1_surface_verification.py
```

### Experiment 2 — Reference Process Diagnostics (`exp/exp2_process_diagnostics.py`)

Simulates all four reference processes (Heston, 3/2, Bergomi-1F, Bass) and reports:
- Martingale check: `E[Y_T]` vs. 1.0 for each maturity.
- Marginal implied volatility vs. target ESSVI surface.
- Wasserstein distance between reference and target marginals before calibration.
- Volatility factor paths (spot variance `V_t` or OU factor `Z_t`).

**Run:**
```bash
python exp/exp2_process_diagnostics.py
```

### Experiment 3 — SKR Calibration: Marginal Matching (`exp/exp3_skr_calibration.py`)

Applies the SKR transport to each reference process and measures marginal fit quality:
- Wasserstein-1 distance between calibrated and target marginals at each maturity.
- Q-Q plots comparing calibrated process quantiles to target quantiles.
- Implied volatility smile before/after calibration overlay.

**Run:**
```bash
python exp/exp3_skr_calibration.py
```

### Experiment 4 — Path-Space Analysis (`exp/exp4_path_analysis.py`)

Analyzes the path-space properties of the calibrated processes:
- Variance swap term structure (realized vs. implied).
- Autocorrelation of squared log-returns (volatility clustering).
- Leverage effect: correlation of log-returns with subsequent variance increments.
- Distribution of forward implied volatility (forward vol smile).

**Run:**
```bash
python exp/exp4_path_analysis.py
```

### Experiment 5 — Exotic Pricing (`exp/exp5_exotic_pricing.py`)

Prices exotic derivatives under the calibrated models and compares across models:
- Barrier options (up-and-out call, down-and-out put).
- Cliquets (forward-starting options, sum of capped/floored local returns).
- Variance swaps and volatility swaps.
- Lookback options (floating strike).

**Run:**
```bash
python exp/exp5_exotic_pricing.py
```

---

## Quick Start

```python
import sys
sys.path.insert(0, "/workspace/project")

from data.load_data import load_spx_target_surface, get_target_marginal_samples
from src.processes import HestonProcess
import numpy as np

# Load the calibrated SPX surface
surface = load_spx_target_surface()
surface.verify_convex_order()

# Sample from target risk-neutral density at T=1.0
samples = get_target_marginal_samples(T=1.0, n_samples=50000)
print(f"Target samples: mean={samples.mean():.4f}, std={samples.std():.4f}")

# Simulate Heston reference process
heston = HestonProcess(
    kappa=1.2484, theta=0.0988, eta=1.1175, rho=-0.8038, v0=0.1020
)
T_dates = [0.25, 0.5, 1.0, 2.0, 3.0]
Y, L = heston.simulate(T_dates, n_paths=50000, seed=42)
print(f"Heston E[Y_T=1]: {Y[:, 2].mean():.6f}")  # index 2 -> T=1.0
```

---

## Key References

- **SKR Paper**: Sequential Kernel Reconstruction for non-parametric calibration of stochastic volatility models.
- **ESSVI**: Gatheral & Jacquier (2014), "Arbitrage-Free SVI Volatility Surfaces", *Risk*.
- **Heston (1993)**: "A Closed-Form Solution for Options with Stochastic Volatility".
- **3/2 Model**: Carr & Sun (2007), "A New Approach for Option Pricing Under Stochastic Volatility".
- **Bergomi (2005)**: "Smile Dynamics", *Risk Magazine*.
- **Bass (1983)**: "Skorokhod Embedding Via Stochastic Integrals".

---

## Notes

- All price processes are **forward-normalized** (`Y_0 = 1`, `F = 1`) with `r = 0.02` carrying charge implicit in strikes.
- ESSVI parameters are calibrated to SPX-like dynamics as of 23 June 2020 (high-volatility post-COVID regime).
- Simulation uses `dt = 1/252` (daily) Euler-Maruyama steps for all processes.
- Results are stored in `results/` as PNG figures and CSV files.
