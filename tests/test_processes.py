import pytest
import numpy as np
from src.processes import HestonProcess, ThreeHalvesProcess, BergomiOneFactor, BassProcess

MATURITIES = [0.25, 0.5, 1.0, 2.0, 3.0]
N_PATHS = 5000

def test_heston_martingale_property():
    proc = HestonProcess()
    Y, L = proc.simulate(MATURITIES, N_PATHS, seed=42)
    # Check E[Y_T] ~ 1 at all maturities
    for i in range(5):
        mean_Y = np.mean(Y[:, i])
        assert abs(mean_Y - 1.0) < 0.05, f"Heston mean at T={MATURITIES[i]}: {mean_Y}"

def test_heston_output_shape():
    proc = HestonProcess()
    Y, L = proc.simulate(MATURITIES, N_PATHS, seed=42)
    assert Y.shape == (N_PATHS, 5)
    assert L.shape == (N_PATHS, 5, 1)  # Variance is 1D factor

def test_heston_positive_paths():
    proc = HestonProcess()
    Y, L = proc.simulate(MATURITIES, N_PATHS, seed=42)
    assert np.all(Y > 0)
    assert np.all(L[:, :, 0] >= 0)  # Variance non-negative

def test_three_halves_martingale_property():
    proc = ThreeHalvesProcess()
    Y, L = proc.simulate(MATURITIES, N_PATHS, seed=42)
    for i in range(5):
        mean_Y = np.mean(Y[:, i])
        assert abs(mean_Y - 1.0) < 0.1, f"3/2 mean at T={MATURITIES[i]}: {mean_Y}"

def test_three_halves_output_shape():
    proc = ThreeHalvesProcess()
    Y, L = proc.simulate(MATURITIES, N_PATHS, seed=42)
    assert Y.shape == (N_PATHS, 5)

def test_bergomi_martingale_property():
    proc = BergomiOneFactor()
    Y, L = proc.simulate(MATURITIES, N_PATHS, seed=42)
    for i in range(5):
        mean_Y = np.mean(Y[:, i])
        assert abs(mean_Y - 1.0) < 0.1, f"Bergomi mean at T={MATURITIES[i]}: {mean_Y}"

def test_bass_martingale_property():
    proc = BassProcess()
    Y, L = proc.simulate(MATURITIES, N_PATHS, seed=42)
    for i in range(5):
        mean_Y = np.mean(Y[:, i])
        assert abs(mean_Y - 1.0) < 0.05

def test_bass_output_shape():
    proc = BassProcess()
    Y, L = proc.simulate(MATURITIES, N_PATHS, seed=42)
    assert Y.shape == (N_PATHS, 5)
    # Bass has no factor state
    assert L.shape[0] == N_PATHS
    assert L.shape[1] == 5

def test_reproducibility_same_seed():
    proc = HestonProcess()
    Y1, _ = proc.simulate(MATURITIES, N_PATHS, seed=123)
    Y2, _ = proc.simulate(MATURITIES, N_PATHS, seed=123)
    np.testing.assert_array_equal(Y1, Y2)

def test_different_seeds_different_paths():
    proc = HestonProcess()
    Y1, _ = proc.simulate(MATURITIES, N_PATHS, seed=1)
    Y2, _ = proc.simulate(MATURITIES, N_PATHS, seed=2)
    assert not np.allclose(Y1, Y2)
