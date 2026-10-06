"""
Utility functions for Black-Scholes pricing, implied volatility inversion,
and statistical tools used throughout the SKR calibration framework.
"""

import numpy as np
from scipy import optimize
from scipy.special import erf


def normal_cdf(x: float | np.ndarray) -> float | np.ndarray:
    """Standard normal cumulative distribution function.

    Parameters
    ----------
    x : float or np.ndarray
        Evaluation point(s).

    Returns
    -------
    float or np.ndarray
        Phi(x) = P(Z <= x) for Z ~ N(0,1).
    """
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def normal_pdf(x: float | np.ndarray) -> float | np.ndarray:
    """Standard normal probability density function.

    Parameters
    ----------
    x : float or np.ndarray
        Evaluation point(s).

    Returns
    -------
    float or np.ndarray
        phi(x) = (1/sqrt(2*pi)) * exp(-x^2/2).
    """
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def black_scholes_call(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> float:
    """Black-Scholes price for a European call option.

    Parameters
    ----------
    S : float
        Current spot price.
    K : float
        Strike price.
    T : float
        Time to expiry in years.
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Implied (lognormal) volatility.

    Returns
    -------
    float
        Call option price: S*N(d1) - K*exp(-r*T)*N(d2).

    Raises
    ------
    ValueError
        If sigma <= 0 or T <= 0.
    """
    if sigma <= 0.0 or T <= 0.0:
        raise ValueError(f"sigma and T must be positive; got sigma={sigma}, T={T}")
    F = S * np.exp(r * T)
    sqrt_T = np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return np.exp(-r * T) * (F * normal_cdf(d1) - K * normal_cdf(d2))


def black_scholes_vega(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> float:
    """Black-Scholes vega: derivative of call price with respect to sigma.

    Parameters
    ----------
    S : float
        Current spot price.
    K : float
        Strike price.
    T : float
        Time to expiry in years.
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Implied volatility.

    Returns
    -------
    float
        dC/dsigma = S * sqrt(T) * phi(d1).
    """
    if sigma <= 0.0 or T <= 0.0:
        raise ValueError(f"sigma and T must be positive; got sigma={sigma}, T={T}")
    F = S * np.exp(r * T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    return S * np.sqrt(T) * normal_pdf(d1)


def implied_vol_brent(
    price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    tol: float = 1e-8,
    max_iter: int = 200,
) -> float:
    """Implied volatility via Brent's root-finding method.

    Inverts the Black-Scholes formula to find the implied volatility
    consistent with the observed option price.

    Parameters
    ----------
    price : float
        Observed market price of the option.
    S : float
        Current spot price.
    K : float
        Strike price.
    T : float
        Time to expiry in years.
    r : float
        Continuously compounded risk-free rate.
    option_type : str, optional
        'call' or 'put'. Default is 'call'.
    tol : float, optional
        Convergence tolerance (absolute). Default is 1e-8.
    max_iter : int, optional
        Maximum number of Brent iterations. Default is 200.

    Returns
    -------
    float
        Implied volatility (annualised), or np.nan if no solution found.
    """
    option_type = option_type.lower()
    intrinsic = max(0.0, S * np.exp((r if option_type == "call" else -r) * T) - K) * np.exp(-r * T)
    if price < intrinsic - tol:
        return np.nan

    def objective(sigma: float) -> float:
        call_price = black_scholes_call(S, K, T, r, sigma)
        if option_type == "call":
            return call_price - price
        # Put-call parity
        put_price = call_price - S + K * np.exp(-r * T)
        return put_price - price

    try:
        iv = optimize.brentq(objective, 1e-6, 10.0, xtol=tol, maxiter=max_iter)
    except ValueError:
        return np.nan
    return iv


def empirical_cdf(
    samples: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the empirical CDF of a sample set.

    Parameters
    ----------
    samples : np.ndarray
        1-D array of observations.

    Returns
    -------
    x_sorted : np.ndarray
        Sorted sample values.
    cdf_vals : np.ndarray
        Empirical CDF values at each sorted point: i/N for i=1,...,N.
    """
    x_sorted = np.sort(samples)
    n = len(x_sorted)
    cdf_vals = np.arange(1, n + 1) / n
    return x_sorted, cdf_vals


def empirical_quantile(
    samples: np.ndarray,
    p_values: np.ndarray,
) -> np.ndarray:
    """Empirical quantile function via linear interpolation.

    Parameters
    ----------
    samples : np.ndarray
        1-D array of observations.
    p_values : np.ndarray
        Probability levels in [0, 1] at which to evaluate the quantile.

    Returns
    -------
    np.ndarray
        Quantile values Q(p) = inf{x : F(x) >= p} estimated via
        linear interpolation of the empirical CDF.
    """
    x_sorted, cdf_vals = empirical_cdf(samples)
    return np.interp(p_values, cdf_vals, x_sorted)


def wasserstein_distance_1d(
    samples1: np.ndarray,
    samples2: np.ndarray,
) -> float:
    """1-Wasserstein (Earth Mover's) distance between two empirical distributions.

    Computed as the L1 distance between the two empirical quantile functions,
    discretised on a common uniform grid of probability levels.

    Parameters
    ----------
    samples1 : np.ndarray
        First set of samples.
    samples2 : np.ndarray
        Second set of samples.

    Returns
    -------
    float
        W_1(mu_1, mu_2) = integral_0^1 |Q_1(p) - Q_2(p)| dp.
    """
    n_grid = max(len(samples1), len(samples2), 1000)
    p_vals = np.linspace(0.0, 1.0, n_grid + 2)[1:-1]  # exclude 0 and 1
    q1 = empirical_quantile(samples1, p_vals)
    q2 = empirical_quantile(samples2, p_vals)
    return float(np.mean(np.abs(q1 - q2)))
