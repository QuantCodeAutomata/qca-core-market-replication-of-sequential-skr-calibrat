"""
Experiment 1: Core market replication of sequential SKR calibration on SPX smile slices.
Reproduces vanilla-smile fit for Bass LV, Heston SKR, 3/2 SKR, and Bergomi-1F SKR.
"""
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.smile_surface import SPXSmileSurface, ESSVISmile
from src.processes import HestonProcess, ThreeHalvesProcess, BergomiOneFactor, BassProcess
from src.skr_calibration import SKRCalibrator
from src.utils import implied_vol_brent
from data.load_data import load_spx_target_surface

MATURITIES = [0.25, 0.5, 1.0, 2.0, 3.0]
MODEL_NAMES = ['bass', 'heston', '3_2', 'bergomi']
MODEL_LABELS = {
    'bass': 'Bass LV',
    'heston': 'Heston SKR',
    '3_2': '3/2 SKR',
    'bergomi': 'Bergomi-1F SKR',
}
MODEL_COLORS = {
    'bass': 'tab:blue',
    'heston': 'tab:orange',
    '3_2': 'tab:green',
    'bergomi': 'tab:red',
}

_BERGOMI_XI0 = {
    (0.0, 0.25): 0.10209961,
    (0.25, 0.50): 0.11963007,
    (0.50, 1.0): 0.09081965,
    (1.0, 2.0): 0.08462666,
    (2.0, 3.0): 0.08383247,
}


def _make_process(model_name):
    """Instantiate the reference process for a given model name."""
    if model_name == 'bass':
        return BassProcess(sigma_avg=0.20)
    elif model_name == 'heston':
        return HestonProcess(kappa=1.2484, theta=0.0988, eta=1.1175, rho=-0.8038, v0=0.1020)
    elif model_name == '3_2':
        return ThreeHalvesProcess(kappa=0.3600, theta=5.0050, eta=7.6870, rho=-1.0000, v0=0.0877)
    elif model_name == 'bergomi':
        return BergomiOneFactor(kappa=2.4971, eta=3.6191, rho=-0.8796, xi0_curve=_BERGOMI_XI0)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_strike_grid(T, n_strikes=30):
    """Generate a strike grid for maturity T in log-moneyness space.

    Parameters
    ----------
    T : float
        Maturity (unused directly; included for interface symmetry).
    n_strikes : int
        Number of strike points.

    Returns
    -------
    K : np.ndarray, shape (n_strikes,)
        Strike grid with log-moneyness k = linspace(-0.5, 0.5, n_strikes), K = exp(k).
    """
    k = np.linspace(-0.5, 0.5, n_strikes)
    return np.exp(k)


def run_calibration_for_model(model_name, surface, n_paths=50000, seed=42):
    """Run SKR calibration for one model and collect repriced implied vols.

    Parameters
    ----------
    model_name : str
        One of 'bass', 'heston', '3_2', 'bergomi'.
    surface : SPXSmileSurface
        Target ESSVI surface.
    n_paths : int
        Monte Carlo path count.
    seed : int
        RNG seed.

    Returns
    -------
    dict with keys:
        'calibrator', 'model_name', 'reprice_ivs', 'target_ivs', 'K_grids', 'timing'
    """
    t0 = time.perf_counter()
    process = _make_process(model_name)
    calibrator = SKRCalibrator(model_name, surface, n_paths=n_paths, seed=seed)
    calibrator.calibrate(process)
    timing = time.perf_counter() - t0

    reprice_ivs = {}
    target_ivs = {}
    K_grids = {}

    for T in MATURITIES:
        K_grid = get_strike_grid(T)
        K_grids[T] = K_grid

        try:
            reprice_ivs[T] = calibrator.compute_implied_vols(T, K_grid)
        except Exception as e:
            print(f"  Warning: IV computation failed for {model_name} T={T}: {e}")
            reprice_ivs[T] = np.full(len(K_grid), np.nan)

        smile = surface.get_smile(T)
        k_grid = np.log(K_grid)  # log-moneyness for ESSVI formula
        target_ivs[T] = smile.implied_vol(k_grid)

    return {
        'calibrator': calibrator,
        'model_name': model_name,
        'reprice_ivs': reprice_ivs,
        'target_ivs': target_ivs,
        'K_grids': K_grids,
        'timing': timing,
    }


def compute_smile_fit_metrics(reprice_ivs, target_ivs):
    """Compute per-maturity smile fit metrics.

    Parameters
    ----------
    reprice_ivs : dict[float, np.ndarray]
        Repriced implied vols from SKR model, keyed by maturity.
    target_ivs : dict[float, np.ndarray]
        ESSVI target implied vols, keyed by maturity.

    Returns
    -------
    dict[float, dict]
        Per-maturity metrics: {'rmse_iv': float, 'max_iv_error': float}.
    """
    metrics = {}
    for T in MATURITIES:
        model_iv = np.asarray(reprice_ivs[T])
        target_iv = np.asarray(target_ivs[T])
        valid = np.isfinite(model_iv) & np.isfinite(target_iv)
        if valid.sum() == 0:
            metrics[T] = {'rmse_iv': np.nan, 'max_iv_error': np.nan}
            continue
        diff = model_iv[valid] - target_iv[valid]
        metrics[T] = {
            'rmse_iv': float(np.sqrt(np.mean(diff ** 2))),
            'max_iv_error': float(np.max(np.abs(diff))),
        }
    return metrics


def plot_smile_fits(results_all_models, save_path):
    """Create a 5-row x 4-col subplot figure of smile fits.

    Each subplot shows target ESSVI (black dashed) vs SKR model (colored solid)
    for one maturity (row) and one model (column).

    Parameters
    ----------
    results_all_models : dict[str, dict]
        Mapping model_name -> result dict from run_calibration_for_model.
    save_path : str
        File path to save the figure.
    """
    n_rows = len(MATURITIES)
    n_cols = len(MODEL_NAMES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))

    for row, T in enumerate(MATURITIES):
        for col, model_name in enumerate(MODEL_NAMES):
            ax = axes[row, col]
            if model_name not in results_all_models:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
                continue

            result = results_all_models[model_name]
            K_grid = result['K_grids'][T]
            target_iv = result['target_ivs'][T]
            reprice_iv = result['reprice_ivs'][T]

            ax.plot(K_grid, target_iv * 100, 'k--', lw=1.5, label='ESSVI target')
            ax.plot(
                K_grid, reprice_iv * 100,
                color=MODEL_COLORS[model_name], lw=1.5,
                label=MODEL_LABELS[model_name],
            )

            if row == 0:
                ax.set_title(MODEL_LABELS[model_name], fontsize=10)
            if col == 0:
                ax.set_ylabel(f'T={T}y\nIV (%)', fontsize=9)
            if row == n_rows - 1:
                ax.set_xlabel('Strike K', fontsize=9)

            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)

    fig.suptitle('SKR Smile Replication: ESSVI Target vs Model', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved smile fit plot to {save_path}")


def main():
    os.makedirs('results', exist_ok=True)

    print("Loading SPX ESSVI surface...")
    surface = load_spx_target_surface()

    results_all_models = {}
    metrics_all = {}

    for model_name in MODEL_NAMES:
        print(f"\n{'=' * 60}")
        print(f"Calibrating {MODEL_LABELS[model_name]}...")
        try:
            result = run_calibration_for_model(model_name, surface, n_paths=2000, seed=42)
            results_all_models[model_name] = result
            metrics_all[model_name] = compute_smile_fit_metrics(
                result['reprice_ivs'], result['target_ivs']
            )
            print(f"  Total calibration time: {result['timing']:.1f}s")
        except Exception as e:
            import traceback
            print(f"  ERROR calibrating {model_name}: {e}")
            traceback.print_exc()

    # Build summary DataFrame: model x maturity, values = IV RMSE
    rows = []
    for model_name in MODEL_NAMES:
        if model_name not in metrics_all:
            continue
        row = {'Model': MODEL_LABELS[model_name]}
        for T in MATURITIES:
            m = metrics_all[model_name].get(T, {})
            row[f'T={T}y RMSE'] = m.get('rmse_iv', np.nan)
            row[f'T={T}y MaxErr'] = m.get('max_iv_error', np.nan)
        rows.append(row)

    df = pd.DataFrame(rows).set_index('Model')
    rmse_cols = [c for c in df.columns if 'RMSE' in c]

    print("\n--- IV RMSE Summary (per maturity) ---")
    print(df[rmse_cols].to_string(float_format=lambda x: f"{x:.4f}" if np.isfinite(x) else 'nan'))

    df.to_csv('results/exp1_smile_fit.csv')
    print("\nSaved results/exp1_smile_fit.csv")

    if results_all_models:
        plot_smile_fits(results_all_models, 'results/exp1_smile_fits.png')

    # Martingale property validation
    print("\n--- Martingale Property Check (|E[X_T] - 1| < 0.05) ---")
    for model_name, result in results_all_models.items():
        calibrator = result['calibrator']
        print(f"  {MODEL_LABELS[model_name]}:")
        for i, T in enumerate(MATURITIES):
            mean_X = float(calibrator.X_paths[:, i].mean())
            deviation = abs(mean_X - 1.0)
            status = 'OK' if deviation < 0.05 else 'FAIL'
            print(f"    T={T:.2f}: E[X_T]={mean_X:.5f}  |dev|={deviation:.5f}  [{status}]")

    return {'results': results_all_models, 'metrics': metrics_all, 'df': df}


if __name__ == '__main__':
    main()
