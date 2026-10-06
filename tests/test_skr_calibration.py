import pytest
import numpy as np
from src.smile_surface import SPXSmileSurface
from src.processes import HestonProcess, BassProcess
from src.skr_calibration import (
    SKRCalibrator, martingale_sinkhorn, build_quantile_map, SKRInterval
)

N_PATHS = 5000  # Small for speed

def test_build_quantile_map_monotone():
    rng = np.random.default_rng(42)
    source = rng.normal(0, 1, 1000)
    target = rng.normal(1, 0.5, 1000)
    g = build_quantile_map(source, target)
    # Test on a range of values
    test_vals = np.linspace(source.min(), source.max(), 100)
    g_vals = g(test_vals)
    # g should be monotone
    assert np.all(np.diff(g_vals) >= -1e-10)

def test_build_quantile_map_preserves_distribution():
    rng = np.random.default_rng(42)
    source = rng.normal(0, 1, 1000)
    target = rng.exponential(1, 1000)
    g = build_quantile_map(source, target)
    # Apply g to source, should approximately match target distribution
    transported = g(source)
    from src.utils import wasserstein_distance_1d
    dist = wasserstein_distance_1d(transported, target)
    assert dist < 0.5  # Should be much smaller than original W1 distance

def test_skr_calibrator_smoke_test():
    # Basic smoke test: calibrate Bass model on small path count
    surface = SPXSmileSurface()
    proc = BassProcess()
    calibrator = SKRCalibrator(
        model_name='bass',
        surface=surface,
        n_paths=N_PATHS,
        max_iter=5,  # Few iterations for speed
        tol=1e-3
    )
    calibrator.calibrate(proc)
    assert calibrator.X_paths.shape == (N_PATHS, 5)
    assert len(calibrator.intervals) == 5

def test_skr_calibrator_martingale_property():
    surface = SPXSmileSurface()
    proc = BassProcess()
    calibrator = SKRCalibrator(
        model_name='bass',
        surface=surface,
        n_paths=N_PATHS,
        max_iter=10,
        tol=1e-3
    )
    calibrator.calibrate(proc)
    # Check E[X_T] ~ 1 at all maturities
    for i in range(5):
        mean_X = np.mean(calibrator.X_paths[:, i])
        assert abs(mean_X - 1.0) < 0.1, f"Mean at maturity {i}: {mean_X}"

def test_skr_calibrator_positive_paths():
    surface = SPXSmileSurface()
    proc = BassProcess()
    calibrator = SKRCalibrator(
        model_name='bass',
        surface=surface,
        n_paths=N_PATHS,
        max_iter=5,
        tol=1e-3
    )
    calibrator.calibrate(proc)
    assert np.all(calibrator.X_paths > 0)

def test_skr_interval_apply_g():
    # Test SKRInterval g application
    rng = np.random.default_rng(42)
    source = rng.normal(0, 1, 1000)
    target = rng.lognormal(0, 0.3, 1000)
    g = build_quantile_map(source, target)
    interval = SKRInterval(T_from=0.0, T_to=0.25, g_func=g, a_values=np.zeros(1000))
    test_z = np.array([0.0, 0.5, -0.5])
    result = interval.apply_g(test_z)
    assert result.shape == (3,)
    assert np.all(result > 0)  # Log-normal target

def test_reprice_calls_positive():
    surface = SPXSmileSurface()
    proc = BassProcess()
    calibrator = SKRCalibrator(
        model_name='bass',
        surface=surface,
        n_paths=N_PATHS,
        max_iter=5,
        tol=1e-3
    )
    calibrator.calibrate(proc)
    K_grid = np.exp(np.linspace(-0.3, 0.3, 10))
    calls = calibrator.reprice_calls(T=1.0, K_grid=K_grid)
    assert np.all(calls >= 0)
    assert calls[0] > calls[-1]  # Decreasing in K
