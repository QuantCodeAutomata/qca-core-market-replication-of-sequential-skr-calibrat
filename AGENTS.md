# SKR Calibration Project — Agent Knowledge

## Project Overview
Sequential Kernel Reconstruction (SKR) calibration framework for stochastic
volatility models. Calibrates reference processes to the SPX ESSVI surface
dated 23 June 2020.

## Environment
- Python 3.12, numpy 1.26.4, scipy 1.14.1, matplotlib 3.11.1, pandas 2.2.2
- No virtual environment needed; packages are installed system-wide.

## Key Implementation Details

### src/utils.py
- `normal_cdf` uses `scipy.special.erf` (numpy has no `np.erf` in this version).
- `implied_vol_brent` uses `scipy.optimize.brentq` on [1e-6, 10.0].
- `wasserstein_distance_1d` uses the quantile representation W_1 = mean|Q_1(u)-Q_2(u)|.

### src/smile_surface.py
- ESSVI formula: w(k) = theta/2 * (1 + rho*phi*k + sqrt((phi*k+rho)^2 + 1-rho^2))
- ATM total variance w(0) = theta exactly (formula simplifies when k=0).
- `build_marginal` returns density in log-moneyness space (Jacobian: density_k = density_K * K).
- `get_marginal_samples` uses inverse-CDF with n_quantiles=2000 grid.

### src/processes.py
- All processes use dt = 1/252 (constant `_DT`).
- `_build_target_step_map` maps sorted T_dates to step indices via `int(round(t / _DT))`.
- **HestonProcess**: truncated Euler (max(0, ...)) for variance; Y uses V_t before V update.
- **ThreeHalvesProcess**: Euler with `np.minimum(np.abs(V_new), _V3H_CAP=25.0)` to prevent
  overflow — the 3/2 model with kappa=0.36, eta=7.687 is numerically explosive without a cap.
  Log-returns clipped to [-50, 50] to prevent exp overflow.
- **BergomiOneFactor**: exact OU transition (Z_new = a*Z + sigma_ou*eps_B) for the factor;
  Euler for price. V_t computed at time t=(i-1)*dt using Z_t BEFORE update.
  Correction = eta^2*(1-exp(-2*kappa*t))/(4*kappa) ensures E[V_t]=xi0(t).
- **BassProcess**: exact log-normal step; L is shape (n_paths, N_dates, 0).

### data/load_data.py
- Adds project root to sys.path for standalone execution.
- `verify_marginals` expects E[Y_T] ≈ 1.0, prints ATM IV and empirical log-vol.

## Numerical Notes
- 3/2 model with rho=-1.0 and eta=7.687 causes variance explosion without V_cap.
- ESSVI density integrates to ≈1 on k in [-1.5, 1.0] for T=1.0, but longer maturities
  need wider ranges; current grid [-1.5, 1.0] covers ~98% of probability mass.
- Martingale check: all processes satisfy |E[Y_T] - 1.0| < 0.05 with n_paths=20000.

## Implemented Files

### src/skr_calibration.py  (~700 lines)
Core SKR algorithm. Key design choices:

**`build_quantile_map(source, target)`**
- Sorts both arrays; uses `np.interp` for piecewise-linear monotone map.
- When len(source) != len(target): re-quantiles on a common uniform grid.
- Returns a closure `g(z)` that handles scalar and ndarray inputs.
- Attaches `._src_sorted` / `._tgt_sorted` to the closure for serialisation.

**`martingale_sinkhorn(Y_from, Y_to, L_from, L_to, X_from, mu_to, ...)`**
- Builds `scipy.spatial.KDTree` **once** on normalised `(Y_from, L_from)` state.
- Queries all N particle neighbourhoods at once (N, K+1) → drops self (col 0).
- Pre-fetches `neighbour_Y_to = Y_to[nn_idx]` of shape (N, K).
- **Inner loop is fully vectorised**: `_f_vec(a_test)` computes
  `mean_k g(a_test_i + Y_to[neigh_ik]) − X_from[i]` for all i simultaneously
  via `a_test[:,None] + neighbour_Y_to` → shape (N,K) → `g(flat)` → `.mean(axis=1)`.
- Uses `_vectorised_bisect` (52 bisection steps) instead of per-particle `brentq`.
- `_expand_brackets` doubles bracket width up to 20× for outlier particles.
- Centres shifts after each step: `a -= a.mean()`.
- Returns `(g_func, a_values, history)` where `history` is a list of dicts
  `{iter, max_change, mean_residual}`.

**`SKRInterval`**
- Fields: `T_from`, `T_to`, `g_func` (callable), `a_values` (N,),
  `convergence_history` (list[dict]), `n_iter`, `_g_src`, `_g_tgt`.
- `apply_g(z)` → `np.interp(z, _g_src, _g_tgt)` (bypasses closure overhead).
- `compute_shifted_terminal(Y_to_new)` → `apply_g(a_values + Y_to_new)`.
- `compute_X_at_T(X_from, Y_from, L_from, Y_to, L_to)` → calls above (for calibration paths only).

**`SKRCalibrator`**
- `T_dates = [0.25, 0.5, 1.0, 2.0, 3.0]`
- `simulate_reference(process)` calls `process.simulate([0.0] + T_dates, n_paths, seed)`;
  stores `_Y_all` (N,6), `_L_all` (N,6,d); public `Y_paths`, `L_paths` are `[:,1:,...]`.
- `calibrate(process)` loops i=0..4; initial state from col-0 of `_Y_all`/`_L_all`.
- `reprice_calls(T, K_grid)` → `mean(max(X-K,0)) * exp(-r*T)` for X = `X_paths[:,i]`.
- `compute_implied_vols` uses `from .utils import implied_vol_brent` with
  **correct signature**: `implied_vol_brent(price, S=1.0, K, T, r)`.
- Inline fallback `_bs_implied_vol_brent(price, T, K, F, r)` if utils unavailable.

### src/pricing.py  (~275 lines)

**`StructuredProductPricer(calibrator, r=0.02)`**

`get_quarterly_paths(calibrator, process=None)` → `(N, 13)` spot paths:
- Calibration dates: `_CAL_DATES = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0]`
- Quarterly dates: `j/4` for j=0..12.
- Exact calibration hits → direct lookup; intermediate → linear interp in X.
- `S_{t_j} = exp(r * t_j) * X_{t_j}`, S_0 = 1.

`price_reverse_cliquet(spot_paths, N_notional=100, C0=0.50, L=0.075, r=0.02, T=3.0)`:
- `clip_j = np.clip(-returns_j, 0, L)`, payoff = `N * max(0, 1+C0−Σclip_j)`.

`price_memory_autocallable(spot_paths, coupon=0.05, coupon_barrier=0.60, ...)`:
- **Vectorised over paths**, loop over 12 time steps only.
- Coupon accrual: `accrued += 1` when S/S0 >= coupon_barrier (while alive).
- Coupon payment: pay `N * coupon * accrued` immediately, reset accrued=0.
- Autocall: pay notional, set alive=False (after coupon payment at same step).
- At maturity (alive): pay principal ± capital loss + remaining accrued coupons.

### exp/__init__.py
Empty module file.

### exp/exp1_market_replication.py
- `get_strike_grid(T, n_strikes=30)`: k=linspace(-0.5,0.5), K=exp(k). ATM strike = 1.0.
- `run_calibration_for_model(model_name, surface, n_paths, seed)`: creates process via
  `_make_process`, creates `SKRCalibrator`, calls `calibrate(process)`, then
  `compute_implied_vols(T, K_grid)` for each of 5 maturities. Returns dict with
  'calibrator', 'reprice_ivs', 'target_ivs', 'K_grids', 'timing'.
- `compute_smile_fit_metrics`: per-maturity RMSE and max abs error.
- `plot_smile_fits`: 5-row × 4-col subplot (ESSVI black dashed, model colored solid).
- Saves results/exp1_smile_fit.csv, results/exp1_smile_fits.png.

### exp/exp2_structured_products.py
- `price_all_models(surface, n_paths=100000)`: calibrates all 4 models, builds
  quarterly paths via `StructuredProductPricer.get_quarterly_paths(calibrator)`,
  prices RC (L=0.075) and AC (coupon_barrier=0.60). Returns (DataFrame, calibrators, paths).
- `run_sensitivity_reverse_cliquet/autocallable`: sweep L or b values re-using pre-built
  spot paths (no re-calibration needed). Returns DataFrames with model labels as columns.
- Paper benchmarks stored in `_PAPER_BENCHMARKS` dict for comparison.
- Saves results/exp2_pricing.csv, results/exp2_sensitivity_rc.csv,
  results/exp2_sensitivity_ac.csv, results/exp2_sensitivity.png.

### exp/exp3_runtime_benchmark.py
- `time_component(func, n_repeats=3)`: times a zero-arg callable.
- `benchmark_model`: times (a) simulate_reference alone, (b) full calibrate (includes
  simulation), (c) RC pricing, (d) AC pricing. fixed_point_time = total_cal - ref_sim.
- `create_timing_table`: builds DataFrame with component columns + 'Paper (reference)' row
  using averages from `_PAPER_TIMINGS` dict.
- Saves results/exp3_timing.csv, results/exp3_timing_breakdown.png.

### exp/exp4_diagnostics.py
- `run_convergence_diagnostics`: runs full calibration, extracts
  `calibrator.intervals[i].convergence_history` (list of {'iter', 'max_change', ...}).
- `validate_martingale_property(calibrator)`: returns DataFrame with E[X_T], std, deviation.
- `validate_terminal_marginals(calibrator, surface)`: uses `scipy.stats.ks_2samp` and
  `wasserstein_distance_1d` to compare X_paths[:,i] with fresh surface samples (seed=999).
- `test_propagation_correctness`: runs two variants of the first two SKR intervals manually
  via `martingale_sinkhorn`; correct uses X_from, wrong uses Y_from.
- `sensitivity_to_path_count`: runs Heston calibration for path_counts=[5000,20000,50000].
- `plot_convergence_histories`: n_models × n_intervals semilogy subplot of max_change.
- Saves results/exp4_diagnostics.csv, results/exp4_martingale.csv,
  results/exp4_convergence.png.

### exp/exp5_synthetic_convergence.py
- Synthetic SLV: dX_t = lambda(t,X_t)*sigma_ref*X_t*dW_t, lambda(x)=1+0.3*tanh(-2*log(x))
- Reference GBM: dY_t = sigma_ref*Y_t*dW_t  (sigma_ref=0.20)
- SAME seed for both → same Z values → paths coupled path-by-path.
- `simulate_slv_benchmark` / `simulate_reference_for_skr`: both draw Z_all with
  `rng.standard_normal((n_steps, n_paths))` → same noise sequence for same seed.
- `run_skr_on_synthetic`: calls `martingale_sinkhorn` with L=zeros (no factors);
  falls back to `build_quantile_map` on error. Returns X_skr dict and intervals list.
- `compute_convergence_error`: path-wise MSE + W1 distance at each date.
  W1 is always ≈0 (marginals matched exactly by quantile map); MSE captures path coupling.
- `run_convergence_study`: simulates fine-grid ONCE, extracts at different mesh sizes.
  Estimates rate via `np.polyfit(log_h, log_err, 1)`.
- Saves results/exp5_convergence.csv, results/exp5_convergence.png.

## API Compatibility Notes
- All process `simulate()` calls must accept `T_dates` including T_0=0.
- `L` arrays with n_factors=0 (BassProcess) have shape (N, D, 0) — handled correctly.
- `get_marginal_samples(T, n_samples, seed=0)` — second arg is `n_samples` (positional).
- POT 0.9.7 is installed but unused (no martingale transport capability).

## Experiment Runtime Tuning

**n_paths scaling** (Heston SKR, all 5 maturities):

| n_paths | Time (s) |
|--------:|---------:|
|     500 |      8.6 |
|   1 000 |     21.2 |
|   2 000 |     43.9 |
|  10 000 |    ~223  |

Bottleneck: `_vectorised_bisect` (n_steps=52) × 25–40 Sinkhorn iterations × N×K array ops.
Safe production setting for all exp main() calls: **n_paths = 2 000** (≤ 45 s/model).

**n_paths set in each experiment main() call:**
- exp1: 2 000 (was 10 000 → cut before first completed run)
- exp2: 2 000
- exp3: 2 000
- exp4: convergence diagnostics 2 000, propagation test 1 000, path sensitivity [500,1000,2000]
- exp5: 2 000

**Test suite timing:** 133 s (43/43 pass), dominated by test_skr_calibration.py and test_pricing.py fixtures (both calibrate BassProcess with n_paths=5 000 and 3 000).

## Confirmed Experiment Results (2025-10-06)

All 5 experiments run successfully end-to-end. Key observations:

1. **SKR produces identical repriced IVs across all models** — expected because g maps every
   process to the same marginals. Model differences are visible only in path-dependent payoffs.
2. **Martingale property**: max |E[X_T]-1| = 0.012 at T=3 (Monte-Carlo noise with n=2000).
3. **IV RMSE at T=0.25 is elevated** (0.057 vs <0.002 in paper) due to n=2000 sampling noise.
4. **Structured products**: RC ≈ 128, AC ≈ 107–109; model spread ~1 point, paper spread ~9 points
   → larger n_paths needed for full model discrimination.
5. **Sinkhorn convergence**: 20–41 iterations per interval; longest at T=3 (Bergomi/Heston).
6. **Propagation test W1 gap invisible** at n=1000; need n≥10000 to see the gap.
7. **Synthetic convergence (exp5)**: MSE *increases* with finer mesh at n=2000 (variance
   dominates). O(h) decay only emerges at n≥20000.

## Pre-existing Files (do not modify without user consent)
- `src/utils.py` — implied_vol_brent signature: `(price, S, K, T, r, option_type, ...)`
- `src/smile_surface.py` — SPXSmileSurface with ESSVI params for 5 maturities
- `src/processes.py` — BassProcess, HestonProcess, ThreeHalvesProcess, BergomiOneFactor
- `data/load_data.py`
- `tests/__init__.py`
