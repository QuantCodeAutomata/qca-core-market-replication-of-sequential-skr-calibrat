"""
Experiment 4: Algorithmic diagnostics and numerical validation.
Validates SKR algorithm correctness: convergence, martingale property, marginal fit.
"""
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy import stats

from src.processes import HestonProcess, ThreeHalvesProcess, BergomiOneFactor, BassProcess
from src.skr_calibration import SKRCalibrator, martingale_sinkhorn
from src.utils import wasserstein_distance_1d
from data.load_data import load_spx_target_surface

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
MATURITIES = [0.25, 0.5, 1.0, 2.0, 3.0]

_BERGOMI_XI0 = {
    (0.0, 0.25): 0.10209961,
    (0.25, 0.50): 0.11963007,
    (0.50, 1.0): 0.09081965,
    (1.0, 2.0): 0.08462666,
    (2.0, 3.0): 0.08383247,
}


def _make_process(model_name):
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


def run_convergence_diagnostics(model_name, surface, n_paths=50000, seed=42):
    """Run calibration and return per-interval convergence history.

    Parameters
    ----------
    model_name : str
    surface : SPXSmileSurface
    n_paths : int
    seed : int

    Returns
    -------
    dict with keys 'calibrator' and 'interval_histories' (list of history lists).
    """
    process = _make_process(model_name)
    calibrator = SKRCalibrator(model_name, surface, n_paths=n_paths, seed=seed)
    calibrator.calibrate(process)

    interval_histories = []
    for interval in calibrator.intervals:
        interval_histories.append(interval.convergence_history)

    return {
        'calibrator': calibrator,
        'model_name': model_name,
        'interval_histories': interval_histories,
    }


def validate_martingale_property(calibrator):
    """Check E[X_{T_i}] for all five calibration maturities.

    Parameters
    ----------
    calibrator : SKRCalibrator
        Fully calibrated calibrator with X_paths populated.

    Returns
    -------
    pd.DataFrame
        Columns: ['T', 'mean_X', 'std_X', 'deviation_from_1'].
    """
    rows = []
    for i, T in enumerate(MATURITIES):
        X = calibrator.X_paths[:, i]
        mean_X = float(X.mean())
        std_X = float(X.std())
        deviation = abs(mean_X - 1.0)
        rows.append({
            'T': T,
            'mean_X': mean_X,
            'std_X': std_X,
            'deviation_from_1': deviation,
        })
    return pd.DataFrame(rows)


def validate_terminal_marginals(calibrator, surface):
    """Compute distributional distance between calibrated and target marginals.

    For each maturity, computes the 1-Wasserstein distance and KS statistic
    between X_paths[:,i] and samples from the ESSVI risk-neutral density.

    Parameters
    ----------
    calibrator : SKRCalibrator
    surface : SPXSmileSurface

    Returns
    -------
    pd.DataFrame
        Columns: ['T', 'wasserstein_dist', 'ks_statistic', 'ks_pvalue'].
    """
    N = calibrator.n_paths
    rows = []

    for i, T in enumerate(MATURITIES):
        X_cal = calibrator.X_paths[:, i]
        # Draw fresh target samples with a different seed to avoid correlations
        target_samples = surface.get_marginal_samples(T, n_samples=N, seed=999)

        try:
            w1 = wasserstein_distance_1d(X_cal, target_samples)
        except Exception:
            w1 = np.nan

        try:
            ks_stat, ks_p = stats.ks_2samp(X_cal, target_samples)
        except Exception:
            ks_stat, ks_p = np.nan, np.nan

        rows.append({
            'T': T,
            'wasserstein_dist': w1,
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_p,
        })

    return pd.DataFrame(rows)


def test_propagation_correctness(model_name, surface, n_paths=20000):
    """Compare correct SKR (X_from propagation) vs incorrect (Y_from propagation).

    Runs two versions of the first two intervals:
    - Correct: uses calibrated X_{T_i} as the initial condition for next interval.
    - Wrong: uses raw reference Y_{T_i} as the initial condition.

    Compares the resulting marginal fit at T=0.5 (second interval).

    Parameters
    ----------
    model_name : str
    surface : SPXSmileSurface
    n_paths : int

    Returns
    -------
    dict with 'correct_w1', 'wrong_w1' (Wasserstein distances at T=0.5).
    """
    seed = 42
    process = _make_process(model_name)
    calibrator = SKRCalibrator(model_name, surface, n_paths=n_paths, seed=seed)
    calibrator.simulate_reference(process)

    Y_all = calibrator._Y_all   # (N, 6)
    L_all = calibrator._L_all   # (N, 6, d)

    N = n_paths
    T_dates = calibrator.T_dates  # [0.25, 0.5, 1.0, 2.0, 3.0]

    # --- Run two intervals manually ---
    def _run_two_intervals(use_correct_propagation):
        """Run intervals 0→0.25 and 0.25→0.5; return X at T=0.5."""
        X_from = np.ones(N)
        Y_from = Y_all[:, 0]
        L_from = L_all[:, 0, :]

        for i in range(2):
            T_to = T_dates[i]
            Y_to = Y_all[:, i + 1]
            L_to = L_all[:, i + 1, :]
            mu_samples = surface.get_marginal_samples(T_to, n_samples=N)

            g_func, a_values, _ = martingale_sinkhorn(
                Y_from, Y_to, L_from, L_to, X_from, mu_samples,
                n_neighbors=30, max_iter=30, tol=1e-4,
            )
            X_to = g_func(a_values + Y_to)

            if use_correct_propagation:
                X_from = X_to   # correct: use calibrated X
            else:
                X_from = Y_to   # wrong: use raw reference Y

            Y_from = Y_to
            L_from = L_to

        return X_to  # X at T=0.5

    target_samples = surface.get_marginal_samples(0.5, n_samples=N, seed=999)

    print(f"  Running correct propagation for {MODEL_LABELS[model_name]}...")
    X_correct = _run_two_intervals(use_correct_propagation=True)
    correct_w1 = wasserstein_distance_1d(X_correct, target_samples)

    print(f"  Running incorrect propagation for {MODEL_LABELS[model_name]}...")
    X_wrong = _run_two_intervals(use_correct_propagation=False)
    wrong_w1 = wasserstein_distance_1d(X_wrong, target_samples)

    return {
        'model_name': model_name,
        'correct_w1': correct_w1,
        'wrong_w1': wrong_w1,
        'correct_mean_X': float(X_correct.mean()),
        'wrong_mean_X': float(X_wrong.mean()),
    }


def sensitivity_to_path_count(model_name, surface, path_counts=None, seed=42):
    """Run calibration for different path counts; record IV RMSE at T=1.0.

    Parameters
    ----------
    model_name : str
    surface : SPXSmileSurface
    path_counts : list[int]
        Path counts to evaluate.
    seed : int

    Returns
    -------
    pd.DataFrame
        Columns: ['n_paths', 'iv_rmse_T1', 'calibration_time'].
    """
    if path_counts is None:
        path_counts = [500, 1000, 2000]

    from src.smile_surface import ESSVISmile
    T_eval = 1.0
    k_grid = np.linspace(-0.5, 0.5, 30)
    K_grid = np.exp(k_grid)
    smile = surface.get_smile(T_eval)
    target_iv = smile.implied_vol(k_grid)

    rows = []
    for n_paths in path_counts:
        print(f"  n_paths={n_paths}...")
        t0 = time.perf_counter()
        try:
            process = _make_process(model_name)
            calibrator = SKRCalibrator(model_name, surface, n_paths=n_paths, seed=seed)
            calibrator.calibrate(process)
            t_cal = time.perf_counter() - t0

            model_iv = calibrator.compute_implied_vols(T_eval, K_grid)
            valid = np.isfinite(model_iv) & np.isfinite(target_iv)
            if valid.sum() > 0:
                rmse = float(np.sqrt(np.mean((model_iv[valid] - target_iv[valid]) ** 2)))
            else:
                rmse = np.nan

            rows.append({'n_paths': n_paths, 'iv_rmse_T1': rmse, 'calibration_time': t_cal})
            print(f"    IV RMSE at T=1.0: {rmse:.4f}, time: {t_cal:.1f}s")

        except Exception as e:
            print(f"    ERROR: {e}")
            rows.append({'n_paths': n_paths, 'iv_rmse_T1': np.nan,
                         'calibration_time': time.perf_counter() - t0})

    return pd.DataFrame(rows)


def plot_convergence_histories(diagnostics_all, save_path):
    """Multi-panel plot of residuals vs iteration for each model and interval.

    Parameters
    ----------
    diagnostics_all : dict[str, dict]
        Mapping model_name -> diagnostics dict from run_convergence_diagnostics.
    save_path : str
    """
    n_models = len(diagnostics_all)
    n_intervals = len(MATURITIES)
    fig, axes = plt.subplots(n_models, n_intervals, figsize=(3 * n_intervals, 3 * n_models))

    if n_models == 1:
        axes = axes[np.newaxis, :]

    for row, (model_name, diag) in enumerate(diagnostics_all.items()):
        histories = diag['interval_histories']
        for col, (T, hist) in enumerate(zip(MATURITIES, histories)):
            ax = axes[row, col]
            if hist:
                iters = [h['iter'] for h in hist]
                max_changes = [h.get('max_change', np.nan) for h in hist]
                ax.semilogy(iters, max_changes, color=MODEL_COLORS.get(model_name, 'gray'),
                            marker='.', ms=4, lw=1.2)
            ax.set_title(f'T={T}y', fontsize=9)
            if col == 0:
                ax.set_ylabel(MODEL_LABELS.get(model_name, model_name), fontsize=9)
            if row == n_models - 1:
                ax.set_xlabel('Iteration', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=7)

    fig.suptitle('SKR Convergence: Max Change per Iteration', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved convergence histories plot to {save_path}")


def main():
    os.makedirs('results', exist_ok=True)

    print("Loading SPX ESSVI surface...")
    surface = load_spx_target_surface()

    diagnostics_all = {}
    martingale_tables = {}
    marginal_tables = {}
    propagation_tests = {}
    path_sensitivity_tables = {}

    for model_name in MODEL_NAMES:
        print(f"\n{'=' * 60}")
        print(f"Diagnostics for {MODEL_LABELS[model_name]}...")

        # 1. Convergence diagnostics + martingale + marginal validation
        try:
            diag = run_convergence_diagnostics(model_name, surface, n_paths=2000, seed=42)
            diagnostics_all[model_name] = diag
            calibrator = diag['calibrator']

            mart_df = validate_martingale_property(calibrator)
            martingale_tables[model_name] = mart_df

            marg_df = validate_terminal_marginals(calibrator, surface)
            marginal_tables[model_name] = marg_df

            print(f"  Martingale deviations: {mart_df['deviation_from_1'].values}")
            print(f"  Wasserstein distances: {marg_df['wasserstein_dist'].values}")

        except Exception as e:
            import traceback
            print(f"  ERROR in convergence diagnostics for {model_name}: {e}")
            traceback.print_exc()

        # 2. Propagation correctness test (lighter weight)
        try:
            prop_result = test_propagation_correctness(model_name, surface, n_paths=1000)
            propagation_tests[model_name] = prop_result
            print(
                f"  Propagation test: correct_W1={prop_result['correct_w1']:.4f}, "
                f"wrong_W1={prop_result['wrong_w1']:.4f}"
            )
        except Exception as e:
            print(f"  ERROR in propagation test for {model_name}: {e}")

        # 3. Sensitivity to path count (first model only to save time)
        if model_name == 'heston':
            try:
                path_df = sensitivity_to_path_count(
                    model_name, surface, path_counts=[500, 1000, 2000], seed=42
                )
                path_sensitivity_tables[model_name] = path_df
                print(f"  Path sensitivity:\n{path_df.to_string(index=False)}")
            except Exception as e:
                print(f"  ERROR in path sensitivity for {model_name}: {e}")

    # --- Plot convergence histories ---
    if diagnostics_all:
        plot_convergence_histories(diagnostics_all, 'results/exp4_convergence.png')

    # --- Build and save martingale summary ---
    mart_rows = []
    for model_name, df in martingale_tables.items():
        for _, row in df.iterrows():
            mart_rows.append({
                'Model': MODEL_LABELS[model_name],
                'T': row['T'],
                'mean_X': row['mean_X'],
                'std_X': row['std_X'],
                'deviation_from_1': row['deviation_from_1'],
            })
    mart_summary = pd.DataFrame(mart_rows)
    mart_summary.to_csv('results/exp4_martingale.csv', index=False)
    print("\nSaved results/exp4_martingale.csv")

    print("\n--- Martingale Property Summary ---")
    print(mart_summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

    # --- Build and save diagnostics summary ---
    diag_rows = []
    for model_name, df in marginal_tables.items():
        for _, row in df.iterrows():
            diag_rows.append({
                'Model': MODEL_LABELS[model_name],
                'T': row['T'],
                'wasserstein_dist': row['wasserstein_dist'],
                'ks_statistic': row['ks_statistic'],
                'ks_pvalue': row['ks_pvalue'],
            })
    diag_summary = pd.DataFrame(diag_rows)
    diag_summary.to_csv('results/exp4_diagnostics.csv', index=False)
    print("\nSaved results/exp4_diagnostics.csv")

    print("\n--- Marginal Fit Summary (Wasserstein Distances) ---")
    print(diag_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # --- Print propagation correctness ---
    if propagation_tests:
        print("\n--- Propagation Correctness Test (W1 at T=0.5) ---")
        print(f"{'Model':<22} {'Correct W1':>12} {'Wrong W1':>12}")
        print("-" * 50)
        for model_name, r in propagation_tests.items():
            print(
                f"{MODEL_LABELS[model_name]:<22} "
                f"{r['correct_w1']:>12.5f} {r['wrong_w1']:>12.5f}"
            )

    return {
        'diagnostics_all': diagnostics_all,
        'martingale_tables': martingale_tables,
        'marginal_tables': marginal_tables,
        'propagation_tests': propagation_tests,
        'path_sensitivity_tables': path_sensitivity_tables,
    }


if __name__ == '__main__':
    main()
