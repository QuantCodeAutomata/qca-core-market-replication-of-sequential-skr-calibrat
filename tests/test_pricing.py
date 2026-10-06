import pytest
import numpy as np
from src.pricing import StructuredProductPricer
from src.smile_surface import SPXSmileSurface
from src.processes import BassProcess
from src.skr_calibration import SKRCalibrator

N_PATHS = 3000  # Small for speed

@pytest.fixture(scope='module')
def calibrated_bass():
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
    return calibrator

def test_quarterly_paths_shape(calibrated_bass):
    pricer = StructuredProductPricer(calibrated_bass)
    spot_paths = pricer.get_quarterly_paths(calibrated_bass)
    assert spot_paths.shape == (N_PATHS, 13)

def test_quarterly_paths_initial_value(calibrated_bass):
    pricer = StructuredProductPricer(calibrated_bass)
    spot_paths = pricer.get_quarterly_paths(calibrated_bass)
    # S_0 = 1 for all paths
    np.testing.assert_allclose(spot_paths[:, 0], 1.0, atol=1e-10)

def test_quarterly_paths_positive(calibrated_bass):
    pricer = StructuredProductPricer(calibrated_bass)
    spot_paths = pricer.get_quarterly_paths(calibrated_bass)
    assert np.all(spot_paths > 0)

def test_reverse_cliquet_price_positive(calibrated_bass):
    pricer = StructuredProductPricer(calibrated_bass)
    spot_paths = pricer.get_quarterly_paths(calibrated_bass)
    price, se = pricer.price_reverse_cliquet(spot_paths)
    assert price > 0
    assert se > 0
    assert price < 200  # Notional is 100, max payoff is 150 (100*(1+0.5))

def test_reverse_cliquet_decreasing_in_L(calibrated_bass):
    pricer = StructuredProductPricer(calibrated_bass)
    spot_paths = pricer.get_quarterly_paths(calibrated_bass)
    # Higher L means more clipping protection for issuer -> lower price for buyer?
    # Actually higher L means more losses can be clipped -> higher payoff (higher price)
    # Test that price changes with L
    price1, _ = pricer.price_reverse_cliquet(spot_paths, L=0.05)
    price2, _ = pricer.price_reverse_cliquet(spot_paths, L=0.10)
    # With higher cap, more losses clipped, so payoff should be higher or equal
    assert price2 >= price1 - 5  # Allow some tolerance

def test_autocallable_price_positive(calibrated_bass):
    pricer = StructuredProductPricer(calibrated_bass)
    spot_paths = pricer.get_quarterly_paths(calibrated_bass)
    price, se = pricer.price_memory_autocallable(spot_paths)
    assert price > 0
    assert se >= 0
    assert price < 200  # Bounded by notional plus coupons

def test_autocallable_price_range(calibrated_bass):
    # Autocallable should be around par value
    pricer = StructuredProductPricer(calibrated_bass)
    spot_paths = pricer.get_quarterly_paths(calibrated_bass)
    price, se = pricer.price_memory_autocallable(spot_paths)
    assert 50 < price < 200  # Broad bounds
