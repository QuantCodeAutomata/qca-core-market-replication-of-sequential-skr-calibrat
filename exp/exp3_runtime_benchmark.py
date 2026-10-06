"""
Experiment 3: Runtime and benchmark comparison.
Measures runtime decomposition for Bass LV, Heston SKR, 3/2 SKR, and Bergomi-1F SKR.
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

from src.processes import HestonProcess, ThreeHalvesProcess, BergomiOneFactor, BassProcess
from src.skr_calibration import SKRCalibrator
from src.pricing import StructuredProductPricer
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

# Paper reference timings (seconds) for comparison
_PAPER_TIMINGS = {
    'bass':    {'ref_sim': 0.3,  'fixed_point': 12.0, 'cliquet_price': 0.1, 'autocall_price': 0.2},
    'heston':  {'ref_sim': 2.0,  'fixed_point': 15.0, 'cliquet_price': 0.1, 'autocall_price': 0.2},
    '3_2':     {'ref_sim': 2.0,  'fixed_point': 15.0, 'cliquet_price': 0.1, 'autocall_price': 0.2},
    'bergomi': {'ref_sim': 3.0,  'fixed_point': 18.0, 'cliquet_price': 0.1, 'autocall_price': 0.2},
}

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


def time_component(func, n_repeats=3):
    """Time a callable n_repeats times and return (mean_time, std_time).

    Parameters
    ----------
    func : callable
        Zero-argument callable to time.
    n_repeats : int
        Number of repetitions.

    Returns
    -------
    (mean_time, std_time) : (float, float)
        Mean and standard deviation of elapsed time in seconds.
    """
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        func()
        times.append(time.perf_counter() - t0)
    arr = np.array(times)
    return float(arr.mean()), float(arr.std())


def benchmark_model(model_name, surface, n_paths=100000, seed=42):
    """Benchmark timing breakdown for one model.

    Separately times:
    a. Reference simulation (simulate_reference only).
    b. Fixed-point iteration (total calibrate minus simulation time).
    c. Reverse cliquet pricing.
    d. Autocallable pricing.

    Parameters
    ----------
    model_name : str
    surface : SPXSmileSurface
    n_paths : int
    seed : int

    Returns
    -------
    dict with timing breakdown and calibrated objects.
    """
    print(f"  Benchmarking {MODEL_LABELS[model_name]}...")
    process = _make_process(model_name)

    # (a) Time reference simulation only
    def _sim_only():
        cal = SKRCalibrator(model_name, surface, n_paths=n_paths, seed=seed)
        cal.simulate_reference(process)
        return cal

    t_sim_start = time.perf_counter()
    _sim_only()
    ref_sim_time = time.perf_counter() - t_sim_start
    print(f"    Ref simulation: {ref_sim_time:.2f}s")

    # (b) Time full calibration (includes simulation internally)
    t_cal_start = time.perf_counter()
    process2 = _make_process(model_name)
    calibrator = SKRCalibrator(model_name, surface, n_paths=n_paths, seed=seed)
    calibrator.calibrate(process2)
    total_cal_time = time.perf_counter() - t_cal_start
    fixed_point_time = max(0.0, total_cal_time - ref_sim_time)
    print(f"    Full calibration: {total_cal_time:.2f}s  (fixed-point: {fixed_point_time:.2f}s)")

    # Build spot paths once for pricing
    pricer = StructuredProductPricer(calibrator, r=0.02)
    spot_paths = pricer.get_quarterly_paths(calibrator)

    # (c) Time reverse cliquet pricing
    t_rc_start = time.perf_counter()
    rc_price, rc_se = pricer.price_reverse_cliquet(spot_paths, L=0.075)
    rc_time = time.perf_counter() - t_rc_start
    print(f"    Reverse cliquet: {rc_time:.4f}s  price={rc_price:.3f}")

    # (d) Time autocallable pricing
    t_ac_start = time.perf_counter()
    ac_price, ac_se = pricer.price_memory_autocallable(spot_paths, coupon_barrier=0.60)
    ac_time = time.perf_counter() - t_ac_start
    print(f"    Autocallable:    {ac_time:.4f}s  price={ac_price:.3f}")

    total_time = ref_sim_time + fixed_point_time + rc_time + ac_time

    return {
        'model_name': model_name,
        'ref_sim': ref_sim_time,
        'fixed_point': fixed_point_time,
        'cliquet_price': rc_time,
        'autocall_price': ac_time,
        'total': total_time,
        'rc_price': rc_price,
        'rc_se': rc_se,
        'ac_price': ac_price,
        'ac_se': ac_se,
        'calibrator': calibrator,
        'spot_paths': spot_paths,
    }


def create_timing_table(benchmark_results):
    """Build a timing comparison table.

    Parameters
    ----------
    benchmark_results : dict[str, dict]
        Output of benchmark_model for each model.

    Returns
    -------
    pd.DataFrame
        Rows = models + paper reference, columns = timing components + total.
    """
    component_cols = ['ref_sim', 'fixed_point', 'cliquet_price', 'autocall_price', 'total']
    col_labels = {
        'ref_sim': 'Ref Sim (s)',
        'fixed_point': 'Fixed Point (s)',
        'cliquet_price': 'Cliquet (s)',
        'autocall_price': 'Autocall (s)',
        'total': 'Total (s)',
    }

    rows = []
    for model_name in MODEL_NAMES:
        if model_name not in benchmark_results:
            continue
        r = benchmark_results[model_name]
        row = {'Model': MODEL_LABELS[model_name]}
        for c in component_cols:
            row[col_labels[c]] = r.get(c, np.nan)
        rows.append(row)

    # Append paper reference row
    paper_row = {'Model': 'Paper (reference)'}
    for c in component_cols:
        vals = [_PAPER_TIMINGS[m].get(c, 0.0) for m in MODEL_NAMES if m in _PAPER_TIMINGS]
        if vals:
            paper_row[col_labels[c]] = float(np.mean(vals))
    rows.append(paper_row)

    df = pd.DataFrame(rows).set_index('Model')
    return df


def plot_timing_breakdown(timing_df, save_path):
    """Stacked bar chart showing timing decomposition per model.

    Parameters
    ----------
    timing_df : pd.DataFrame
        Output of create_timing_table (without the paper reference row for clarity).
    save_path : str
    """
    component_cols = ['Ref Sim (s)', 'Fixed Point (s)', 'Cliquet (s)', 'Autocall (s)']
    colors = ['steelblue', 'tomato', 'gold', 'mediumseagreen']

    # Exclude paper reference row for the plot
    plot_df = timing_df.drop(index='Paper (reference)', errors='ignore')
    plot_df = plot_df[[c for c in component_cols if c in plot_df.columns]]

    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(len(plot_df))

    for col, color in zip(component_cols, colors):
        if col not in plot_df.columns:
            continue
        vals = plot_df[col].fillna(0).values
        ax.bar(plot_df.index, vals, bottom=bottom, label=col, color=color, edgecolor='white')
        bottom += vals

    ax.set_ylabel('Time (seconds)', fontsize=11)
    ax.set_title('Runtime Breakdown by Model', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved timing breakdown plot to {save_path}")


def main():
    os.makedirs('results', exist_ok=True)

    print("Loading SPX ESSVI surface...")
    surface = load_spx_target_surface()

    benchmark_results = {}
    for model_name in MODEL_NAMES:
        print(f"\n{'=' * 60}")
        try:
            result = benchmark_model(model_name, surface, n_paths=2000, seed=42)
            benchmark_results[model_name] = result
        except Exception as e:
            import traceback
            print(f"  ERROR benchmarking {model_name}: {e}")
            traceback.print_exc()

    timing_df = create_timing_table(benchmark_results)

    print("\n--- Timing Table (seconds) ---")
    print(timing_df.to_string(float_format=lambda x: f"{x:.3f}" if np.isfinite(x) else 'nan'))

    timing_df.to_csv('results/exp3_timing.csv')
    print("\nSaved results/exp3_timing.csv")

    plot_timing_breakdown(timing_df, 'results/exp3_timing_breakdown.png')

    # Comparison with paper targets
    print("\n--- Comparison with Paper Reference Timings ---")
    print(f"{'Model':<22} {'Our Total':>12} {'Paper Ref':>12} {'Ratio':>8}")
    print("-" * 57)
    for model_name in MODEL_NAMES:
        if model_name not in benchmark_results:
            continue
        our_total = benchmark_results[model_name]['total']
        paper_total = sum(_PAPER_TIMINGS[model_name].values())
        ratio = our_total / paper_total if paper_total > 0 else np.nan
        print(
            f"{MODEL_LABELS[model_name]:<22} {our_total:>12.2f} {paper_total:>12.2f} {ratio:>8.2f}x"
        )

    return {'benchmark_results': benchmark_results, 'timing_df': timing_df}


if __name__ == '__main__':
    main()
