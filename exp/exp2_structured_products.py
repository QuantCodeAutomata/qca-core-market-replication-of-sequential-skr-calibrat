"""
Experiment 2: Structured-product pricing under calibrated SKR models.
Prices reverse cliquet and memory autocallable under all four models.
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

# Paper benchmark prices for reference comparison
_PAPER_BENCHMARKS = {
    'RC':  {'bass': 113.987, 'heston': 122.0,    '3_2': 116.991, 'bergomi': 120.144},
    'AC':  {'bass': 104.899, 'heston': 106.834,  '3_2': 106.444, 'bergomi': 106.800},
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


def price_all_models(surface, n_paths=100000, seed=42):
    """Calibrate all four models and price both structured products.

    Parameters
    ----------
    surface : SPXSmileSurface
        Target ESSVI surface.
    n_paths : int
        Monte Carlo path count.
    seed : int
        RNG seed.

    Returns
    -------
    pd.DataFrame
        Columns: ['Model', 'RC_price', 'RC_se', 'AC_price', 'AC_se'].
        Also returns (calibrators, quarterly_paths_dict) for reuse in sensitivity.
    """
    rows = []
    calibrators = {}
    quarterly_paths_dict = {}

    for model_name in MODEL_NAMES:
        print(f"\nCalibrating {MODEL_LABELS[model_name]}...")
        try:
            t0 = time.perf_counter()
            process = _make_process(model_name)
            calibrator = SKRCalibrator(model_name, surface, n_paths=n_paths, seed=seed)
            calibrator.calibrate(process)
            t_cal = time.perf_counter() - t0
            print(f"  Calibration done in {t_cal:.1f}s")

            pricer = StructuredProductPricer(calibrator, r=0.02)
            spot_paths = pricer.get_quarterly_paths(calibrator)

            rc_price, rc_se = pricer.price_reverse_cliquet(spot_paths, L=0.075)
            ac_price, ac_se = pricer.price_memory_autocallable(
                spot_paths, coupon_barrier=0.60
            )

            calibrators[model_name] = calibrator
            quarterly_paths_dict[model_name] = spot_paths

            rows.append({
                'Model': MODEL_LABELS[model_name],
                'RC_price': rc_price,
                'RC_se': rc_se,
                'AC_price': ac_price,
                'AC_se': ac_se,
            })
            print(f"  RC={rc_price:.3f} (±{rc_se:.3f}), AC={ac_price:.3f} (±{ac_se:.3f})")

        except Exception as e:
            import traceback
            print(f"  ERROR pricing {model_name}: {e}")
            traceback.print_exc()
            rows.append({
                'Model': MODEL_LABELS[model_name],
                'RC_price': np.nan, 'RC_se': np.nan,
                'AC_price': np.nan, 'AC_se': np.nan,
            })

    df = pd.DataFrame(rows)
    return df, calibrators, quarterly_paths_dict


def run_sensitivity_reverse_cliquet(calibrators, quarterly_paths_dict, L_values):
    """Price the reverse cliquet across a range of local loss caps L.

    Parameters
    ----------
    calibrators : dict[str, SKRCalibrator]
    quarterly_paths_dict : dict[str, np.ndarray]
        Pre-built (N, 13) spot paths per model.
    L_values : list[float]
        Loss cap values to sweep.

    Returns
    -------
    pd.DataFrame
        Index = L_values, columns = model names (labels), values = prices.
    """
    results = {MODEL_LABELS[m]: [] for m in MODEL_NAMES if m in calibrators}

    for L in L_values:
        for model_name in MODEL_NAMES:
            if model_name not in calibrators:
                results[MODEL_LABELS[model_name]].append(np.nan)
                continue
            try:
                calibrator = calibrators[model_name]
                pricer = StructuredProductPricer(calibrator, r=0.02)
                spot_paths = quarterly_paths_dict[model_name]
                price, _ = pricer.price_reverse_cliquet(spot_paths, L=L)
                results[MODEL_LABELS[model_name]].append(price)
            except Exception as e:
                print(f"  RC sensitivity error L={L} model={model_name}: {e}")
                results[MODEL_LABELS[model_name]].append(np.nan)

    df = pd.DataFrame(results, index=L_values)
    df.index.name = 'L'
    return df


def run_sensitivity_autocallable(calibrators, quarterly_paths_dict, b_values):
    """Price the memory autocallable across a range of coupon barriers b.

    Parameters
    ----------
    calibrators : dict[str, SKRCalibrator]
    quarterly_paths_dict : dict[str, np.ndarray]
    b_values : list[float]
        Coupon barrier values to sweep.

    Returns
    -------
    pd.DataFrame
        Index = b_values, columns = model labels, values = prices.
    """
    results = {MODEL_LABELS[m]: [] for m in MODEL_NAMES if m in calibrators}

    for b in b_values:
        for model_name in MODEL_NAMES:
            if model_name not in calibrators:
                results[MODEL_LABELS[model_name]].append(np.nan)
                continue
            try:
                calibrator = calibrators[model_name]
                pricer = StructuredProductPricer(calibrator, r=0.02)
                spot_paths = quarterly_paths_dict[model_name]
                price, _ = pricer.price_memory_autocallable(spot_paths, coupon_barrier=b)
                results[MODEL_LABELS[model_name]].append(price)
            except Exception as e:
                print(f"  AC sensitivity error b={b} model={model_name}: {e}")
                results[MODEL_LABELS[model_name]].append(np.nan)

    df = pd.DataFrame(results, index=b_values)
    df.index.name = 'b'
    return df


def plot_sensitivity_curves(rc_sensitivity, ac_sensitivity, save_path):
    """Two-panel figure: price vs L (reverse cliquet) and price vs b (autocallable).

    Parameters
    ----------
    rc_sensitivity : pd.DataFrame
        Output of run_sensitivity_reverse_cliquet.
    ac_sensitivity : pd.DataFrame
        Output of run_sensitivity_autocallable.
    save_path : str
        File path for the saved figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for model_name in MODEL_NAMES:
        label = MODEL_LABELS[model_name]
        color = MODEL_COLORS[model_name]
        if label in rc_sensitivity.columns:
            ax1.plot(rc_sensitivity.index, rc_sensitivity[label], color=color,
                     marker='o', ms=4, label=label)
        if label in ac_sensitivity.columns:
            ax2.plot(ac_sensitivity.index, ac_sensitivity[label], color=color,
                     marker='s', ms=4, label=label)

    ax1.set_xlabel('Local Loss Cap L', fontsize=11)
    ax1.set_ylabel('Price (per 100 notional)', fontsize=11)
    ax1.set_title('Reverse Cliquet: Sensitivity to L', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel('Coupon Barrier b', fontsize=11)
    ax2.set_ylabel('Price (per 100 notional)', fontsize=11)
    ax2.set_title('Memory Autocallable: Sensitivity to b', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved sensitivity curves to {save_path}")


def main():
    os.makedirs('results', exist_ok=True)

    print("Loading SPX ESSVI surface...")
    surface = load_spx_target_surface()

    # Price all models
    pricing_df, calibrators, quarterly_paths_dict = price_all_models(
        surface, n_paths=2000, seed=42
    )

    # Print pricing table
    print("\n--- Structured Product Prices ---")
    print(pricing_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Compare to paper benchmarks
    print("\n--- Comparison to Paper Benchmarks ---")
    print(f"{'Model':<22} {'RC Price':>10} {'RC Paper':>10} {'AC Price':>10} {'AC Paper':>10}")
    print("-" * 65)
    for _, row in pricing_df.iterrows():
        label = row['Model']
        key = {v: k for k, v in MODEL_LABELS.items()}.get(label, '')
        rc_paper = _PAPER_BENCHMARKS['RC'].get(key, np.nan)
        ac_paper = _PAPER_BENCHMARKS['AC'].get(key, np.nan)
        rc_str = f"{row['RC_price']:.3f}" if np.isfinite(row['RC_price']) else 'nan'
        ac_str = f"{row['AC_price']:.3f}" if np.isfinite(row['AC_price']) else 'nan'
        rc_p_str = f"{rc_paper:.3f}" if np.isfinite(rc_paper) else 'N/A'
        ac_p_str = f"{ac_paper:.3f}" if np.isfinite(ac_paper) else 'N/A'
        print(f"{label:<22} {rc_str:>10} {rc_p_str:>10} {ac_str:>10} {ac_p_str:>10}")

    pricing_df.to_csv('results/exp2_pricing.csv', index=False)
    print("\nSaved results/exp2_pricing.csv")

    # Sensitivity analyses
    L_values = [0.02, 0.04, 0.06, 0.075, 0.09, 0.12, 0.15]
    b_values = [0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80]

    print("\nRunning reverse cliquet sensitivity (L sweep)...")
    rc_sensitivity = run_sensitivity_reverse_cliquet(
        calibrators, quarterly_paths_dict, L_values
    )

    print("\nRunning autocallable sensitivity (b sweep)...")
    ac_sensitivity = run_sensitivity_autocallable(
        calibrators, quarterly_paths_dict, b_values
    )

    rc_sensitivity.to_csv('results/exp2_sensitivity_rc.csv')
    ac_sensitivity.to_csv('results/exp2_sensitivity_ac.csv')
    print("Saved results/exp2_sensitivity_rc.csv and results/exp2_sensitivity_ac.csv")

    plot_sensitivity_curves(rc_sensitivity, ac_sensitivity, 'results/exp2_sensitivity.png')

    print("\n--- RC Sensitivity Table ---")
    print(rc_sensitivity.to_string(float_format=lambda x: f"{x:.3f}"))

    print("\n--- AC Sensitivity Table ---")
    print(ac_sensitivity.to_string(float_format=lambda x: f"{x:.3f}"))

    return {
        'pricing_df': pricing_df,
        'rc_sensitivity': rc_sensitivity,
        'ac_sensitivity': ac_sensitivity,
    }


if __name__ == '__main__':
    main()
