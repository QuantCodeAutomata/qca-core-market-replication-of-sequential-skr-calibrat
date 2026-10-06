"""
Structured product pricing using SKR-calibrated Monte Carlo paths.

Implements two structured products:
  1. Reverse Cliquet   – capped sum of quarterly local losses.
  2. Memory Autocallable – quarterly knock-out note with memory coupon.

All prices assume a continuously compounded risk-free rate ``r`` and an
initial normalised spot S_0 = 1 (i.e. all paths are relative to S_0).
"""

import numpy as np
from .skr_calibration import SKRCalibrator


# ---------------------------------------------------------------------------
# Quarterly date grid (T_0=0, T_1=0.25, …, T_12=3)
# ---------------------------------------------------------------------------
_QUARTERLY_DATES = np.arange(13) / 4.0          # 0, 0.25, 0.50, …, 3.0

# Calibration date grid (including T_0=0)
_CAL_DATES = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 3.0])


def _interp_X_paths(calibrator: SKRCalibrator) -> np.ndarray:
    """Build a (N, 6) array of calibrated X values at all six calibration dates.

    Index 0 = T_0=0  →  X = 1 (by definition).
    Indices 1-5 = T_1 … T_5  →  from ``calibrator.X_paths``.
    """
    N = calibrator.n_paths
    X_cal = np.empty((N, 6))
    X_cal[:, 0] = 1.0
    X_cal[:, 1:] = calibrator.X_paths      # (N, 5)
    return X_cal


class StructuredProductPricer:
    """Price path-dependent structured products using SKR-calibrated paths.

    Parameters
    ----------
    calibrator : SKRCalibrator
        A fully calibrated SKR calibrator (``calibrate`` already called).
    r : float
        Continuously compounded risk-free rate.  Default 0.02.
    """

    def __init__(self, calibrator: SKRCalibrator, r: float = 0.02):
        self.calibrator = calibrator
        self.r = r

    # ------------------------------------------------------------------
    # Path construction
    # ------------------------------------------------------------------

    def get_quarterly_paths(self, calibrator: SKRCalibrator, process=None) -> np.ndarray:
        """Build spot paths S_{t_j} at quarterly dates t_j = j/4, j=0,…,12.

        Strategy
        --------
        * If t_j coincides with a calibration date T_i → use X_paths[:,i-1] directly.
        * Otherwise → linearly interpolate between adjacent calibration X values.
        * Convert to spot: S_{t_j} = exp(r * t_j) * X_{t_j}

        Note
        ----
        The proper approach for intermediate dates would propagate the conditional
        bridge distribution; linear interpolation is an approximation valid when
        calibration dates are close.  The ``process`` argument is accepted for
        forward compatibility but is unused in this implementation.

        Parameters
        ----------
        calibrator : SKRCalibrator – must have ``X_paths`` populated.
        process    : reference process object (reserved for future use).

        Returns
        -------
        spot_paths : ndarray, shape (N, 13)
            Normalised spot paths at the 13 quarterly dates.  S_0 = 1.
        """
        N = calibrator.n_paths
        X_cal = _interp_X_paths(calibrator)   # (N, 6) at _CAL_DATES

        spot_paths = np.empty((N, 13))

        for j, t_j in enumerate(_QUARTERLY_DATES):
            if t_j == 0.0:
                # S_0 = 1 by convention
                spot_paths[:, j] = np.exp(self.r * 0.0) * 1.0
                continue

            # Find the enclosing calibration interval [T_left, T_right]
            cal_idx_right = int(np.searchsorted(_CAL_DATES, t_j, side="left"))

            if _CAL_DATES[cal_idx_right] == t_j:
                # t_j exactly matches calibration date
                X_j = X_cal[:, cal_idx_right]
            else:
                # Linear interpolation between neighbouring calibration dates
                i_lo = cal_idx_right - 1
                i_hi = cal_idx_right
                T_lo = _CAL_DATES[i_lo]
                T_hi = _CAL_DATES[i_hi]
                alpha = (t_j - T_lo) / (T_hi - T_lo)
                X_j = (1.0 - alpha) * X_cal[:, i_lo] + alpha * X_cal[:, i_hi]

            spot_paths[:, j] = np.exp(self.r * t_j) * X_j

        return spot_paths

    # ------------------------------------------------------------------
    # Product 1: Reverse Cliquet
    # ------------------------------------------------------------------

    def price_reverse_cliquet(
        self,
        spot_paths: np.ndarray,
        N_notional: float = 100.0,
        C0: float = 0.50,
        L: float = 0.075,
        r: float = 0.02,
        T: float = 3.0,
    ) -> tuple:
        """Price a reverse cliquet note.

        Payoff
        ------
        For each quarter j = 1, …, 12:
            R_j    = S_{t_j} / S_{t_{j-1}} - 1           (quarterly return)
            clip_j = max(0, -R_j - L)                     (loss in excess of floor L)

        Payoff at T:
            N_notional * max(0, 1 + C0 - sum_j clip_j)

        Parameters
        ----------
        spot_paths  : (N, 13) – normalised spot paths from get_quarterly_paths
        N_notional  : float  – face value
        C0          : float  – initial coupon budget (0.50 → 50 %)
        L           : float  – per-quarter loss floor (0.075 → 7.5 %); only losses
                               exceeding L are deducted from the budget.  A higher L
                               provides more protection → higher price.
        r           : float  – risk-free rate for discounting
        T           : float  – product maturity in years

        Returns
        -------
        (price, se) : (float, float)
            Monte Carlo estimate and its standard error.
        """
        # Quarterly returns: S_{t_j} / S_{t_{j-1}} - 1  for j=1,...,12
        returns = spot_paths[:, 1:] / spot_paths[:, :-1] - 1.0    # (N, 12)

        # Local losses: only the portion of each quarterly loss exceeding floor L
        local_losses = np.maximum(-returns - L, 0.0)               # (N, 12)

        # Total clipped loss over all quarters
        total_loss = local_losses.sum(axis=1)                       # (N,)

        # Payoff
        payoff_raw = N_notional * np.maximum(0.0, 1.0 + C0 - total_loss)  # (N,)
        pv = payoff_raw * np.exp(-r * T)

        price = float(pv.mean())
        se = float(pv.std() / np.sqrt(len(pv)))
        return price, se

    # ------------------------------------------------------------------
    # Product 2: Memory Autocallable
    # ------------------------------------------------------------------

    def price_memory_autocallable(
        self,
        spot_paths: np.ndarray,
        N_notional: float = 100.0,
        coupon: float = 0.05,
        coupon_barrier: float = 0.60,
        autocall_barrier: float = 1.0,
        protection_barrier: float = 0.60,
        r: float = 0.02,
        T: float = 3.0,
    ) -> tuple:
        """Price a memory autocallable note.

        Observation dates: t_j = j/4 for j = 1, …, 12  (quarterly).

        Mechanics (per path, evaluated in order at each t_j)
        ------------------------------------------------------
        1. Coupon accrual & payment (while alive):
           If S_{t_j}/S_0 >= coupon_barrier:
               accrued_coupons += 1
               pv += N_notional * coupon * accrued_coupons * exp(-r*t_j)
               accrued_coupons = 0
        2. Autocall (while alive):
           If S_{t_j}/S_0 >= autocall_barrier:
               pv += N_notional * exp(-r*t_j)
               alive = False
        3. At maturity (still alive, t_12 = T):
           s_T = S_T/S_0
           If s_T < protection_barrier:
               pv += N_notional * s_T * exp(-r*T)          (capital at risk)
           Else:
               pv += N_notional * exp(-r*T)                 (full principal)
           Plus any remaining accrued coupons:
               pv += N_notional * coupon * accrued_coupons * exp(-r*T)

        The "memory" feature means coupons missed when below the barrier are
        carried forward and paid in full on the next observation above the barrier
        (or at maturity).

        Parameters
        ----------
        spot_paths       : (N, 13) – from get_quarterly_paths
        N_notional       : float – face value
        coupon           : float – per-observation coupon rate (e.g. 0.05 = 5%)
        coupon_barrier   : float – coupon / memory trigger (fraction of S_0)
        autocall_barrier : float – autocall trigger (fraction of S_0)
        protection_barrier : float – capital-protection trigger at maturity
        r                : float – risk-free rate
        T                : float – product maturity in years

        Returns
        -------
        (price, se) : (float, float)
            Monte Carlo estimate and standard error.
        """
        N = spot_paths.shape[0]
        S0 = spot_paths[:, 0]                  # S_0 = 1 per convention

        alive = np.ones(N, dtype=bool)
        accrued = np.zeros(N, dtype=np.int32)
        pv = np.zeros(N)

        t_obs = _QUARTERLY_DATES[1:]           # t_1, …, t_12  shape (12,)

        for k in range(12):                    # k = 0 → j=1, …, k=11 → j=12
            t_k = t_obs[k]
            df_k = np.exp(-r * t_k)
            s_ratio = spot_paths[:, k + 1] / S0   # S_{t_{k+1}} / S_0

            # --- Coupon barrier check ---
            coupon_hit = alive & (s_ratio >= coupon_barrier)
            if coupon_hit.any():
                accrued[coupon_hit] += 1
                pv[coupon_hit] += (
                    N_notional * coupon * accrued[coupon_hit] * df_k
                )
                accrued[coupon_hit] = 0

            # --- Autocall check ---
            autocall_hit = alive & (s_ratio >= autocall_barrier)
            if autocall_hit.any():
                pv[autocall_hit] += N_notional * df_k
                alive[autocall_hit] = False

        # --- Maturity redemption for surviving paths ---
        if alive.any():
            s_T = spot_paths[:, 12] / S0
            df_T = np.exp(-r * T)

            # Principal redemption
            capital_loss = alive & (s_T < protection_barrier)
            pv[capital_loss] += N_notional * s_T[capital_loss] * df_T
            pv[alive & ~capital_loss] += N_notional * df_T

            # Pay out any remaining memory coupons at maturity
            has_accrued = alive & (accrued > 0)
            if has_accrued.any():
                pv[has_accrued] += (
                    N_notional * coupon * accrued[has_accrued] * df_T
                )

        price = float(pv.mean())
        se = float(pv.std() / np.sqrt(N))
        return price, se
