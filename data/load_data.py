"""
Data loading utilities for the SKR calibration framework.

Provides access to the SPX ESSVI smile surface (23 June 2020) and convenience
functions for sampling from risk-neutral marginals.
"""

import sys
import os

# Ensure project root is on the path when this module is imported directly
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

from src.smile_surface import SPXSmileSurface


def load_spx_target_surface(F: float = 1.0, r: float = 0.02) -> SPXSmileSurface:
    """Create and return the SPX ESSVI smile surface dated 23 June 2020.

    The surface contains five maturity slices at T = 0.25, 0.5, 1.0, 2.0, 3.0
    years, parametrized by ESSVI (Extended SVI) with parameters calibrated to
    SPX-like dynamics in the post-COVID high-volatility regime.

    Parameters
    ----------
    F : float, optional
        Forward price normalisation. Default is 1.0.
    r : float, optional
        Risk-free rate. Default is 0.02.

    Returns
    -------
    SPXSmileSurface
        Fully constructed surface object ready for sampling and IV queries.
    """
    return SPXSmileSurface(F=F, r=r)


def get_target_marginal_samples(
    T: float,
    n_samples: int,
    seed: int = 0,
    F: float = 1.0,
    r: float = 0.02,
) -> np.ndarray:
    """Draw samples from the risk-neutral marginal at maturity T.

    Uses the inverse-CDF (quantile) method applied to the ESSVI risk-neutral
    density derived from the SPX surface dated 23 June 2020.

    Parameters
    ----------
    T : float
        Maturity in years; must be one of {0.25, 0.5, 1.0, 2.0, 3.0}.
    n_samples : int
        Number of Monte Carlo samples to draw.
    seed : int, optional
        Random seed for reproducibility. Default is 0.
    F : float, optional
        Forward price normalisation. Default is 1.0.
    r : float, optional
        Risk-free rate. Default is 0.02.

    Returns
    -------
    np.ndarray
        Shape (n_samples,). Forward-price samples Y_T ~ Q^T (risk-neutral
        marginal at maturity T). Samples are forward-normalised so E[Y_T] ≈ 1.

    Raises
    ------
    KeyError
        If T is not in the set of supported maturities.
    """
    surface = SPXSmileSurface(F=F, r=r)
    return surface.get_marginal_samples(T, n_samples=n_samples, seed=seed)


def verify_marginals(surface: SPXSmileSurface) -> None:
    """Print verification statistics for all five maturity slices.

    For each maturity T, draws 100,000 samples from the risk-neutral density
    and reports:
    - Sample mean (should be ≈ 1.0, martingale check).
    - Sample standard deviation.
    - ATM implied volatility from ESSVI surface.
    - Empirical standard deviation of log-returns (approx. ATM vol * sqrt(T)).
    - Convex order check (variance increasing in T).

    Parameters
    ----------
    surface : SPXSmileSurface
        The ESSVI smile surface to verify.
    """
    print("=" * 65)
    print("SPX ESSVI Surface Verification (23 June 2020)")
    print("=" * 65)
    print(f"{'T':>5}  {'E[Y]':>8}  {'Std[Y]':>8}  {'ATM IV':>8}  {'Emp logVol':>12}")
    print("-" * 65)

    for T in surface.maturities:
        samples = surface.get_marginal_samples(T, n_samples=100_000, seed=0)
        mean_Y = float(np.mean(samples))
        std_Y = float(np.std(samples))

        smile = surface.get_smile(T)
        atm_iv = float(smile.implied_vol(np.array([0.0]))[0])

        log_returns = np.log(samples)
        emp_log_vol = float(np.std(log_returns) / np.sqrt(T))

        print(
            f"{T:>5.2f}  {mean_Y:>8.5f}  {std_Y:>8.5f}  "
            f"{atm_iv:>8.4f}  {emp_log_vol:>12.4f}"
        )

    print("-" * 65)
    print()
    surface.verify_convex_order()
    print("=" * 65)


if __name__ == "__main__":
    surface = load_spx_target_surface()
    verify_marginals(surface)
