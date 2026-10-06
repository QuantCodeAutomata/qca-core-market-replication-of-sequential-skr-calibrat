"""
Experiment 5: Synthetic convergence experiment for SKR to SLV limit.
Tests convergence of SKR to SLV as maturity grid is refined.

Synthetic model setup
---------------------
Reference GBM : dY_t = sigma_ref * Y_t * dW_t  (sigma_ref = 0.20)
SLV target    : dX_t = lambda(t, X_t) * sigma_ref * X_t * dW_t
                lambda(t, x) = 1 + 0.3 * tanh(-2 * log(x))

Both processes share the SAME Brownian motion W_t (same random seed → same Z values),
so paths are coupled path-by-path.  As the observation grid is refined (h → 0) the SKR
calibrated X_skr[i] converges to the SLV path X_slv[i] driven by the same noise.
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

from src.skr_calibration import martingale_sinkhorn, build_quantile_map
from src.utils import wasserstein_distance_1d

# Synthetic model parameters
_SIGMA_REF = 0.20


def _leverage(x):
    """Local leverage function lambda(t, x) = 1 + 0.3 * tanh(-2 * log(x))."""
    log_x = np.log(np.maximum(x, 1e-12))
    return 1.0 + 0.3 * np.tanh(-2.0 * log_x)


def simulate_slv_benchmark(T, n_steps_per_year=252, n_paths=50000, seed=42):
    """Simulate the synthetic SLV (local-vol) process on a fine time grid.

    Dynamics: X_{t+dt} = X_t * exp(-0.5 * lam^2 * sigma_ref^2 * dt + lam * sigma_ref * sqrt(dt) * Z)
    where lam = lambda(t, X_t).

    Parameters
    ----------
    T : float
        Terminal time in years.
    n_steps_per_year : int
        Number of Euler steps per year.
    n_paths : int
        Number of Monte Carlo paths.
    seed : int
        RNG seed.

    Returns
    -------
    t_grid : np.ndarray, shape (n_steps + 1,)
        Time grid including t=0.
    X_paths : np.ndarray, shape (n_paths, n_steps + 1)
        SLV paths at every fine-grid time point.
    """
    n_steps = int(round(T * n_steps_per_year))
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    t_grid = np.linspace(0.0, T, n_steps + 1)

    rng = np.random.default_rng(seed)
    Z_all = rng.standard_normal((n_steps, n_paths))   # all noise drawn upfront

    X = np.ones(n_paths)
    X_paths = np.empty((n_paths, n_steps + 1))
    X_paths[:, 0] = X

    for i in range(n_steps):
        lam = _leverage(X)
        vol = lam * _SIGMA_REF
        X = X * np.exp(-0.5 * vol ** 2 * dt + vol * sqrt_dt * Z_all[i])
        X = np.maximum(X, 1e-12)
        X_paths[:, i + 1] = X

    return t_grid, X_paths


def simulate_reference_for_skr(T, n_steps_per_year=252, n_paths=50000, seed=42):
    """Simulate the reference GBM Y_t with the SAME noise as simulate_slv_benchmark.

    Uses the same seed → same Z_all sequence → paths share the same Brownian increments.

    Dynamics: Y_{t+dt} = Y_t * exp(-0.5 * sigma_ref^2 * dt + sigma_ref * sqrt(dt) * Z)

    Parameters
    ----------
    T : float
    n_steps_per_year : int
    n_paths : int
    seed : int

    Returns
    -------
    t_grid : np.ndarray, shape (n_steps + 1,)
    Y_paths : np.ndarray, shape (n_paths, n_steps + 1)
    """
    n_steps = int(round(T * n_steps_per_year))
    dt = T / n_steps
    sqrt_dt = np.sqrt(dt)
    t_grid = np.linspace(0.0, T, n_steps + 1)

    rng = np.random.default_rng(seed)
    Z_all = rng.standard_normal((n_steps, n_paths))   # SAME call pattern as SLV

    Y = np.ones(n_paths)
    Y_paths = np.empty((n_paths, n_steps + 1))
    Y_paths[:, 0] = Y

    for i in range(n_steps):
        Y = Y * np.exp(-0.5 * _SIGMA_REF ** 2 * dt + _SIGMA_REF * sqrt_dt * Z_all[i])
        Y_paths[:, i + 1] = Y

    return t_grid, Y_paths


def extract_marginals_from_slv(slv_paths, t_grid, observation_dates):
    """Extract marginal samples from SLV paths at each observation date.

    Parameters
    ----------
    slv_paths : np.ndarray, shape (n_paths, n_fine_steps + 1)
        SLV paths at all fine-grid times.
    t_grid : np.ndarray, shape (n_fine_steps + 1,)
        Fine-grid time points.
    observation_dates : list[float]
        Target observation dates (subset of t_grid, approximately).

    Returns
    -------
    dict[float, np.ndarray]
        Mapping observation_date -> marginal samples (shape n_paths,).
    """
    marginals = {}
    for t_obs in observation_dates:
        idx = int(np.argmin(np.abs(t_grid - t_obs)))
        marginals[t_obs] = slv_paths[:, idx].copy()
    return marginals


def run_skr_on_synthetic(observation_dates, marginal_samples, Y_paths_at_dates, seed=42):
    """Run SKR calibration on the synthetic SLV setting.

    For each interval [t_i, t_{i+1}], applies the martingale Sinkhorn algorithm
    using the reference GBM increments and the SLV marginals as targets.

    Parameters
    ----------
    observation_dates : list[float]
        Sorted observation dates (NOT including t=0).
    marginal_samples : dict[float, np.ndarray]
        SLV marginal samples at each observation date.
    Y_paths_at_dates : dict[float, np.ndarray]
        Reference GBM values at {0.0} union observation_dates.
    seed : int
        RNG seed for sinkhorn (unused here; kept for API consistency).

    Returns
    -------
    X_skr : dict[float, np.ndarray]
        Calibrated X values at each observation date.
    intervals : list[dict]
        Per-interval metadata (g_func, a_values, convergence).
    """
    observation_dates = sorted(observation_dates)
    dates_with_zero = [0.0] + list(observation_dates)
    N = len(Y_paths_at_dates[0.0])

    X_from = np.ones(N)
    Y_from = Y_paths_at_dates[0.0]
    L_empty = np.zeros((N, 0))   # no latent factors for simple GBM

    X_skr = {}
    intervals = []

    for t_to in observation_dates:
        Y_to = Y_paths_at_dates[t_to]
        mu_samples = marginal_samples[t_to]

        try:
            g_func, a_values, history = martingale_sinkhorn(
                Y_from, Y_to,
                L_empty, L_empty,
                X_from, mu_samples,
                n_neighbors=30,
                max_iter=30,
                tol=1e-4,
            )
            X_to = g_func(a_values + Y_to)
        except Exception as e:
            # Fallback: quantile transport only (no martingale constraint)
            print(f"  SKR warning at t={t_to:.4f}: {e}; using quantile map fallback")
            g_func = build_quantile_map(Y_to, mu_samples)
            a_values = np.zeros(N)
            X_to = g_func(Y_to)
            history = []

        X_skr[t_to] = X_to
        intervals.append({
            't_to': t_to,
            'g_func': g_func,
            'a_values': a_values,
            'n_iter': len(history),
        })

        X_from = X_to
        Y_from = Y_to

    return X_skr, intervals


def compute_convergence_error(X_skr_at_dates, X_slv_at_dates):
    """Compute path-wise and distributional error metrics at each observation date.

    Parameters
    ----------
    X_skr_at_dates : dict[float, np.ndarray]
        Calibrated SKR values at each date.
    X_slv_at_dates : dict[float, np.ndarray]
        True SLV values at each date (same path ordering / noise coupling).

    Returns
    -------
    dict[float, dict]
        date -> {'mse': float, 'wasserstein': float}.
    """
    errors = {}
    for t in sorted(X_skr_at_dates.keys()):
        x_skr = X_skr_at_dates[t]
        x_slv = X_slv_at_dates[t]
        mse = float(np.mean((x_skr - x_slv) ** 2))
        try:
            w1 = wasserstein_distance_1d(x_skr, x_slv)
        except Exception:
            w1 = np.nan
        errors[t] = {'mse': mse, 'wasserstein': w1}
    return errors


def run_convergence_study(mesh_sizes=None, T=1.0, n_paths=30000):
    """Convergence study: error vs observation grid mesh size h.

    For each mesh size h, generates observation dates [h, 2h, ..., T], runs SKR,
    and computes the error between X_skr and X_slv at the terminal date T.

    Parameters
    ----------
    mesh_sizes : list[float]
        Mesh sizes h to evaluate.  Defaults to [0.25, 0.125, 0.0625, 0.03125].
    T : float
        Terminal time. Default 1.0.
    n_paths : int
        Number of Monte Carlo paths. Default 30000.

    Returns
    -------
    pd.DataFrame
        Columns: ['h', 'n_obs', 'mse_T', 'wasserstein_T', 'time'].
    """
    if mesh_sizes is None:
        mesh_sizes = [0.25, 0.125, 0.0625, 0.03125]

    n_steps_per_year = 252
    seed = 42
    rows = []

    # Simulate both processes ONCE at the finest resolution
    print("  Simulating SLV benchmark (fine grid)...")
    t_grid, X_slv_fine = simulate_slv_benchmark(
        T, n_steps_per_year=n_steps_per_year, n_paths=n_paths, seed=seed
    )

    print("  Simulating reference GBM (fine grid)...")
    _, Y_ref_fine = simulate_reference_for_skr(
        T, n_steps_per_year=n_steps_per_year, n_paths=n_paths, seed=seed
    )

    for h in mesh_sizes:
        n_obs = int(round(T / h))
        obs_dates = [round(j * h, 10) for j in range(1, n_obs + 1)]
        # Ensure terminal date T is exactly included
        obs_dates[-1] = T
        print(f"\n  Mesh size h={h:.5f}  ({n_obs} intervals, obs_dates={obs_dates[:3]}...)")

        t0 = time.perf_counter()

        # Extract Y and X_slv at observation dates (using the fine-grid paths)
        Y_at_dates = {}
        X_slv_at_dates = {}
        Y_at_dates[0.0] = Y_ref_fine[:, 0].copy()
        for t_obs in obs_dates:
            idx = int(np.argmin(np.abs(t_grid - t_obs)))
            Y_at_dates[t_obs] = Y_ref_fine[:, idx].copy()
            X_slv_at_dates[t_obs] = X_slv_fine[:, idx].copy()

        # Extract SLV marginals at observation dates
        marginals = extract_marginals_from_slv(X_slv_fine, t_grid, obs_dates)

        # Run SKR
        try:
            X_skr, _ = run_skr_on_synthetic(obs_dates, marginals, Y_at_dates, seed=seed)
        except Exception as e:
            print(f"    SKR failed: {e}")
            rows.append({'h': h, 'n_obs': n_obs, 'mse_T': np.nan,
                         'wasserstein_T': np.nan, 'time': np.nan})
            continue

        elapsed = time.perf_counter() - t0

        # Compute error at terminal date T
        errors = compute_convergence_error(X_skr, X_slv_at_dates)
        err_T = errors.get(T, {})
        mse_T = err_T.get('mse', np.nan)
        w1_T = err_T.get('wasserstein', np.nan)

        print(f"    MSE at T: {mse_T:.6f}  W1 at T: {w1_T:.6f}  ({elapsed:.1f}s)")
        rows.append({
            'h': h,
            'n_obs': n_obs,
            'mse_T': mse_T,
            'wasserstein_T': w1_T,
            'time': elapsed,
        })

    return pd.DataFrame(rows)


def plot_convergence(convergence_df, save_path):
    """Log-log plot of error vs mesh size h, with reference line h^1.

    Parameters
    ----------
    convergence_df : pd.DataFrame
        Output of run_convergence_study.
    save_path : str
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    h_vals = convergence_df['h'].values
    valid = np.isfinite(h_vals) & (h_vals > 0)

    for ax, col, title, ylabel in [
        (ax1, 'mse_T', 'Path MSE at T: E[|X_skr - X_slv|²]', 'MSE'),
        (ax2, 'wasserstein_T', 'W₁ Distance at T', 'W₁'),
    ]:
        err_vals = convergence_df[col].values
        ok = valid & np.isfinite(err_vals) & (err_vals > 0)

        if ok.sum() >= 2:
            ax.loglog(h_vals[ok], err_vals[ok], 'o-', color='tab:blue',
                      lw=2, ms=7, label='SKR error')

            # Reference line O(h^1)
            h_ref = h_vals[ok]
            scale = err_vals[ok][0] / h_ref[0]
            ax.loglog(h_ref, scale * h_ref, 'k--', lw=1.5, label='O(h¹)')

        ax.set_xlabel('Mesh size h', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, which='both', alpha=0.3)

    fig.suptitle('SKR → SLV Convergence as Observation Grid Refines', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved convergence plot to {save_path}")


def main():
    os.makedirs('results', exist_ok=True)

    print("Running SKR-to-SLV synthetic convergence study...")
    print(f"Sigma_ref={_SIGMA_REF}, SLV leverage: lambda(x)=1+0.3*tanh(-2*log(x))")

    try:
        convergence_df = run_convergence_study(
            mesh_sizes=[0.25, 0.125, 0.0625, 0.03125],
            T=1.0,
            n_paths=2000,
        )

        print("\n--- Convergence Table ---")
        print(convergence_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

        convergence_df.to_csv('results/exp5_convergence.csv', index=False)
        print("\nSaved results/exp5_convergence.csv")

        plot_convergence(convergence_df, 'results/exp5_convergence.png')

        # Estimate convergence rate via log-log linear regression
        h_vals = convergence_df['h'].values
        for col, label in [('mse_T', 'MSE'), ('wasserstein_T', 'W1')]:
            err_vals = convergence_df[col].values
            ok = np.isfinite(h_vals) & (h_vals > 0) & np.isfinite(err_vals) & (err_vals > 0)
            if ok.sum() >= 2:
                log_h = np.log(h_vals[ok])
                log_err = np.log(err_vals[ok])
                coeffs = np.polyfit(log_h, log_err, 1)
                rate = coeffs[0]
                print(f"\n  Estimated convergence rate ({label}): O(h^{rate:.3f})")
            else:
                print(f"\n  Insufficient valid data to estimate {label} rate.")

    except Exception as e:
        import traceback
        print(f"ERROR in convergence study: {e}")
        traceback.print_exc()
        convergence_df = pd.DataFrame()

    return {'convergence_df': convergence_df}


if __name__ == '__main__':
    main()
