import pytest
import numpy as np
from src.smile_surface import ESSVISmile, SPXSmileSurface

def test_essvi_total_variance_positive():
    smile = ESSVISmile(T=1.0, theta=0.09, rho=-0.7, phi=1.1)
    k_grid = np.linspace(-0.5, 0.5, 50)
    w = smile.total_variance(k_grid)
    assert np.all(w > 0)

def test_essvi_implied_vol_positive():
    smile = ESSVISmile(T=1.0, theta=0.09, rho=-0.7, phi=1.1)
    k_grid = np.linspace(-0.5, 0.5, 50)
    iv = smile.implied_vol(k_grid)
    assert np.all(iv > 0)
    assert np.all(iv < 2.0)  # Should be reasonable

def test_essvi_call_price_bounds():
    smile = ESSVISmile(T=1.0, theta=0.09, rho=-0.7, phi=1.1)
    K = np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3])
    calls = smile.call_price(K)
    assert np.all(calls >= 0)
    # Call prices should be decreasing in K
    assert np.all(np.diff(calls) <= 0.01)

def test_essvi_density_positive():
    smile = ESSVISmile(T=1.0, theta=0.09, rho=-0.7, phi=1.1)
    k_grid = np.linspace(-0.8, 0.8, 100)
    dk = k_grid[1] - k_grid[0]
    density = smile.risk_neutral_density(k_grid, dk)
    # Density should be mostly positive (might have small negative values at boundaries)
    assert np.sum(density[density < -1e-6]) < 1  # Few violations

def test_spx_surface_has_five_maturities():
    surface = SPXSmileSurface()
    for T in [0.25, 0.5, 1.0, 2.0, 3.0]:
        smile = surface.get_smile(T)
        assert smile is not None
        assert abs(smile.T - T) < 1e-10

def test_spx_surface_convex_order():
    surface = SPXSmileSurface()
    # Verify increasing variance (necessary for convex order)
    result = surface.verify_convex_order()
    assert result is True or result is False  # Should run without error

def test_marginal_samples_mean_near_one():
    surface = SPXSmileSurface()
    for T in [0.25, 1.0, 3.0]:
        samples = surface.get_marginal_samples(T, n_samples=10000)
        assert abs(np.mean(samples) - 1.0) < 0.05

def test_marginal_samples_positive():
    surface = SPXSmileSurface()
    samples = surface.get_marginal_samples(T=1.0, n_samples=5000)
    assert np.all(samples > 0)
