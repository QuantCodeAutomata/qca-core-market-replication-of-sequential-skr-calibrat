import pytest
import numpy as np
from src.utils import (
    black_scholes_call, implied_vol_brent, black_scholes_vega,
    normal_cdf, normal_pdf, empirical_cdf, empirical_quantile, wasserstein_distance_1d
)

def test_black_scholes_call_at_the_money():
    # BS call at ATM should be positive
    price = black_scholes_call(S=1.0, K=1.0, T=1.0, r=0.0, sigma=0.2)
    assert price > 0
    assert price < 0.1  # Should be around 0.0797
    assert abs(price - 0.0797) < 0.005

def test_black_scholes_call_deep_itm():
    # Deep ITM call should approach S - K*exp(-rT)
    price = black_scholes_call(S=2.0, K=1.0, T=1.0, r=0.0, sigma=0.01)
    assert abs(price - 1.0) < 0.01

def test_black_scholes_call_deep_otm():
    # Deep OTM call should be close to zero
    price = black_scholes_call(S=1.0, K=3.0, T=1.0, r=0.0, sigma=0.2)
    assert price < 0.001

def test_implied_vol_brent_roundtrip():
    # Test round-trip: price -> IV -> price should match
    S, K, T, r, sigma = 1.0, 1.0, 1.0, 0.02, 0.25
    price = black_scholes_call(S, K, T, r, sigma)
    iv = implied_vol_brent(price, S, K, T, r)
    assert abs(iv - sigma) < 1e-5

def test_implied_vol_brent_otm():
    S, K, T, r, sigma = 1.0, 1.2, 0.5, 0.02, 0.30
    price = black_scholes_call(S, K, T, r, sigma)
    iv = implied_vol_brent(price, S, K, T, r)
    assert abs(iv - sigma) < 1e-4

def test_normal_cdf_properties():
    # CDF should be between 0 and 1
    assert abs(normal_cdf(0) - 0.5) < 1e-10
    assert normal_cdf(-5) < 0.001
    assert normal_cdf(5) > 0.999

def test_normal_pdf_integrates_to_one():
    x = np.linspace(-5, 5, 10000)
    dx = x[1] - x[0]
    pdf_vals = np.array([normal_pdf(xi) for xi in x])
    assert abs(pdf_vals.sum() * dx - 1.0) < 0.001

def test_empirical_quantile_monotone():
    rng = np.random.default_rng(42)
    samples = rng.normal(0, 1, 1000)
    p_vals = np.linspace(0.01, 0.99, 99)
    quantiles = empirical_quantile(samples, p_vals)
    assert np.all(np.diff(quantiles) >= 0)  # Monotone

def test_empirical_quantile_median():
    rng = np.random.default_rng(42)
    samples = rng.normal(0, 1, 10000)
    q50 = empirical_quantile(samples, np.array([0.5]))[0]
    assert abs(q50) < 0.1

def test_wasserstein_distance_same_distribution():
    rng = np.random.default_rng(42)
    samples = rng.normal(0, 1, 1000)
    dist = wasserstein_distance_1d(samples, samples)
    assert dist == 0.0

def test_wasserstein_distance_shifted():
    rng = np.random.default_rng(42)
    s1 = rng.normal(0, 1, 1000)
    s2 = rng.normal(1, 1, 1000)  # Shifted by 1
    dist = wasserstein_distance_1d(s1, s2)
    assert abs(dist - 1.0) < 0.1  # Should be approximately 1
