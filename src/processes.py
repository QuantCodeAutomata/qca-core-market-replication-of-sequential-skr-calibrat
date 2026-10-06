"""
Reference stochastic processes for the SKR calibration framework.

All processes simulate forward-normalized (Y_0 = 1) martingale price paths
using Euler-Maruyama discretization with dt = 1/252 (daily steps).

Fixed parameters from Table 1 of the SKR paper:
  Heston     : kappa=1.2484, theta=0.0988, eta=1.1175, rho=-0.8038, v0=0.1020
  3/2        : kappa=0.3600, theta=5.0050, eta=7.6870, rho=-1.0000, v0=0.0877
  Bergomi-1F : kappa=2.4971, eta=3.6191, rho=-0.8796,
               xi0 piecewise = {(0,0.25):0.10209961, (0.25,0.5):0.11963007,
                                 (0.5,1.0):0.09081965, (1.0,2.0):0.08462666,
                                 (2.0,3.0):0.08383247}

Output format for all models
------------------------------
  Y : np.ndarray, shape (n_paths, N_dates)
      Forward price process recorded at each target date.
  L : np.ndarray, shape (n_paths, N_dates, n_factors)
      Factor state (variance/OU factor) at each target date.
"""

import numpy as np


_DT = 1.0 / 252.0  # daily time step
_V3H_CAP = 25.0    # hard variance cap for 3/2 process (sigma_cap ≈ 500%)


def _build_target_step_map(T_dates: list[float]) -> dict[int, int]:
    """Map sorted target dates to their nearest Euler step indices."""
    return {int(round(t / _DT)): j for j, t in enumerate(sorted(T_dates))}


class HestonProcess:
    """Heston stochastic volatility model.

    Price and variance dynamics under the risk-neutral measure:

        dY_t = sqrt(V_t) * Y_t * dW_y
        dV_t = kappa * (theta - V_t) * dt + eta * sqrt(V_t) * dW_v
        corr(dW_y, dW_v) = rho

    Discretized via truncated Euler-Maruyama (variance floored at 0).

    Parameters
    ----------
    kappa : float
        Mean-reversion speed for variance.
    theta : float
        Long-run variance level.
    eta : float
        Volatility-of-variance (vol-of-vol).
    rho : float
        Leverage correlation in (-1, 1).
    v0 : float
        Initial variance V_0.
    S0 : float, optional
        Initial forward price. Default is 1.0.
    r : float, optional
        Risk-free rate (used for forward adjustment). Default is 0.02.
    """

    def __init__(
        self,
        kappa: float = 1.2484,
        theta: float = 0.0988,
        eta: float = 1.1175,
        rho: float = -0.8038,
        v0: float = 0.1020,
        S0: float = 1.0,
        r: float = 0.02,
    ) -> None:
        self.kappa = kappa
        self.theta = theta
        self.eta = eta
        self.rho = rho
        self.v0 = v0
        self.S0 = S0
        self.r = r

    def simulate(
        self,
        T_dates: list[float],
        n_paths: int,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate the Heston process.

        Parameters
        ----------
        T_dates : list[float]
            Target observation dates in years (e.g. [0.25, 0.5, 1.0, 2.0, 3.0]).
        n_paths : int
            Number of Monte Carlo paths.
        seed : int, optional
            RNG seed for reproducibility. Default is 42.

        Returns
        -------
        Y : np.ndarray, shape (n_paths, len(T_dates))
            Forward price process at each target date.
        L : np.ndarray, shape (n_paths, len(T_dates), 1)
            Variance process V_t at each target date (L[:,:,0] = V_t).
        """
        T_dates_sorted = sorted(T_dates)
        T_max = T_dates_sorted[-1]
        N_steps = int(round(T_max / _DT))
        N_dates = len(T_dates_sorted)

        Y_out = np.zeros((n_paths, N_dates))
        L_out = np.zeros((n_paths, N_dates, 1))
        target_steps = _build_target_step_map(T_dates_sorted)

        rng = np.random.default_rng(seed)
        rho_perp = np.sqrt(max(0.0, 1.0 - self.rho**2))
        sqrt_dt = np.sqrt(_DT)

        Y = np.full(n_paths, self.S0)
        V = np.full(n_paths, self.v0)

        for i in range(1, N_steps + 1):
            Z1 = rng.standard_normal(n_paths)
            Z2 = rng.standard_normal(n_paths)
            # Cholesky decomposition: Z_y ~ N(0,1), Z_v = rho*Z_y + rho_perp*Z_ind
            Z_y = Z1
            Z_v = self.rho * Z1 + rho_perp * Z2

            sq_V = np.sqrt(np.maximum(V, 0.0))

            # Euler-Maruyama for price (uses V_t, the value at start of interval)
            Y = Y * np.exp(-0.5 * V * _DT + sq_V * sqrt_dt * Z_y)

            # Truncated Euler-Maruyama for variance (uses V_t)
            V = np.maximum(
                0.0,
                V + self.kappa * (self.theta - V) * _DT + self.eta * sq_V * sqrt_dt * Z_v,
            )

            if i in target_steps:
                j = target_steps[i]
                Y_out[:, j] = Y
                L_out[:, j, 0] = V

        return Y_out, L_out


class ThreeHalvesProcess:
    """3/2 stochastic volatility model.

    Price and variance dynamics:

        dY_t = sqrt(V_t) * Y_t * dW_y
        dV_t = kappa * V_t * (theta - V_t) * dt + eta * V_t^(3/2) * dW_v
        corr(dW_y, dW_v) = rho

    Discretized via Euler-Maruyama with a reflecting boundary (|V|) to prevent
    the variance from becoming negative.

    Parameters
    ----------
    kappa : float
        Mean-reversion speed.
    theta : float
        Long-run variance level (in units of variance).
    eta : float
        Volatility-of-variance coefficient.
    rho : float
        Leverage correlation.
    v0 : float
        Initial variance V_0.
    S0 : float, optional
        Initial forward price. Default is 1.0.
    """

    def __init__(
        self,
        kappa: float = 0.3600,
        theta: float = 5.0050,
        eta: float = 7.6870,
        rho: float = -1.0000,
        v0: float = 0.0877,
        S0: float = 1.0,
    ) -> None:
        self.kappa = kappa
        self.theta = theta
        self.eta = eta
        self.rho = rho
        self.v0 = v0
        self.S0 = S0

    def simulate(
        self,
        T_dates: list[float],
        n_paths: int,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate the 3/2 process.

        Parameters
        ----------
        T_dates : list[float]
            Target observation dates in years.
        n_paths : int
            Number of Monte Carlo paths.
        seed : int, optional
            RNG seed for reproducibility. Default is 42.

        Returns
        -------
        Y : np.ndarray, shape (n_paths, len(T_dates))
            Forward price process at each target date.
        L : np.ndarray, shape (n_paths, len(T_dates), 1)
            Variance process V_t at each target date (L[:,:,0] = V_t).
        """
        T_dates_sorted = sorted(T_dates)
        T_max = T_dates_sorted[-1]
        N_steps = int(round(T_max / _DT))
        N_dates = len(T_dates_sorted)

        Y_out = np.zeros((n_paths, N_dates))
        L_out = np.zeros((n_paths, N_dates, 1))
        target_steps = _build_target_step_map(T_dates_sorted)

        rng = np.random.default_rng(seed)
        rho_perp = np.sqrt(max(0.0, 1.0 - self.rho**2))
        sqrt_dt = np.sqrt(_DT)

        Y = np.full(n_paths, self.S0)
        V = np.full(n_paths, self.v0)

        for i in range(1, N_steps + 1):
            Z1 = rng.standard_normal(n_paths)
            Z2 = rng.standard_normal(n_paths)
            Z_y = Z1
            Z_v = self.rho * Z1 + rho_perp * Z2

            sq_V = np.sqrt(np.maximum(V, 0.0))
            V_pos = np.maximum(V, 0.0)

            # Euler-Maruyama for price (log-return clipped to avoid exp overflow)
            log_ret = np.clip(
                -0.5 * V_pos * _DT + sq_V * sqrt_dt * Z_y,
                -50.0,
                50.0,
            )
            Y = Y * np.exp(log_ret)

            # Euler for 3/2 variance: reflecting boundary + hard cap
            V_new = (
                V
                + self.kappa * V * (self.theta - V) * _DT
                + self.eta * V * sq_V * sqrt_dt * Z_v
            )
            V = np.minimum(np.abs(V_new), _V3H_CAP)

            if i in target_steps:
                j = target_steps[i]
                Y_out[:, j] = Y
                L_out[:, j, 0] = V

        return Y_out, L_out


class BergomiOneFactor:
    """One-factor Bergomi stochastic volatility model.

    The instantaneous variance is driven by an Ornstein-Uhlenbeck factor Z_t:

        dZ_t = -kappa * Z_t * dt + dB_t          (OU factor)
        V_t  = xi0(t) * exp(eta*Z_t - eta^2*(1-exp(-2*kappa*t))/(4*kappa))
        dY_t = sqrt(V_t) * Y_t * (rho*dB_t + sqrt(1-rho^2)*dW_perp)

    where xi0(t) is a piecewise-constant initial variance curve.
    The correction term ensures E[V_t] = xi0(t) for all t.

    The OU process is simulated via its exact conditional distribution:
        Z_{t+dt} = Z_t * exp(-kappa*dt)
                   + sqrt((1-exp(-2*kappa*dt))/(2*kappa)) * eps_B

    Parameters
    ----------
    kappa : float
        Mean-reversion speed of the OU factor.
    eta : float
        Volatility-of-variance coefficient.
    rho : float
        Leverage correlation between price and OU factor.
    xi0_curve : dict[tuple[float, float], float]
        Piecewise-constant initial variance curve.
        Keys are (t_start, t_end) intervals; values are xi0(t) on that interval.
    S0 : float, optional
        Initial forward price. Default is 1.0.
    """

    _DEFAULT_XI0: dict[tuple[float, float], float] = {
        (0.00, 0.25): 0.10209961,
        (0.25, 0.50): 0.11963007,
        (0.50, 1.00): 0.09081965,
        (1.00, 2.00): 0.08462666,
        (2.00, 3.00): 0.08383247,
    }

    def __init__(
        self,
        kappa: float = 2.4971,
        eta: float = 3.6191,
        rho: float = -0.8796,
        xi0_curve: dict[tuple[float, float], float] | None = None,
        S0: float = 1.0,
    ) -> None:
        self.kappa = kappa
        self.eta = eta
        self.rho = rho
        self.xi0_curve = xi0_curve if xi0_curve is not None else self._DEFAULT_XI0
        self.S0 = S0

        # Pre-sort the intervals for fast lookup
        self._intervals = sorted(self.xi0_curve.keys(), key=lambda iv: iv[0])

    def _xi0(self, t: float) -> float:
        """Return xi0(t) from the piecewise-constant curve."""
        for (t_start, t_end), val in sorted(self.xi0_curve.items()):
            if t_start <= t < t_end:
                return val
        # Beyond last interval: return value of last interval
        return list(self.xi0_curve.values())[-1]

    def _correction(self, t: float) -> float:
        """Ito correction: eta^2 * (1 - exp(-2*kappa*t)) / (4*kappa)."""
        if t <= 0.0:
            return 0.0
        return self.eta**2 * (1.0 - np.exp(-2.0 * self.kappa * t)) / (4.0 * self.kappa)

    def simulate(
        self,
        T_dates: list[float],
        n_paths: int,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate the one-factor Bergomi process.

        Parameters
        ----------
        T_dates : list[float]
            Target observation dates in years.
        n_paths : int
            Number of Monte Carlo paths.
        seed : int, optional
            RNG seed for reproducibility. Default is 42.

        Returns
        -------
        Y : np.ndarray, shape (n_paths, len(T_dates))
            Forward price process at each target date.
        L : np.ndarray, shape (n_paths, len(T_dates), 1)
            OU factor Z_t at each target date (L[:,:,0] = Z_t).
        """
        T_dates_sorted = sorted(T_dates)
        T_max = T_dates_sorted[-1]
        N_steps = int(round(T_max / _DT))
        N_dates = len(T_dates_sorted)

        Y_out = np.zeros((n_paths, N_dates))
        L_out = np.zeros((n_paths, N_dates, 1))
        target_steps = _build_target_step_map(T_dates_sorted)

        rng = np.random.default_rng(seed)
        rho_perp = np.sqrt(max(0.0, 1.0 - self.rho**2))
        sqrt_dt = np.sqrt(_DT)

        # Exact OU transition: Z_{t+dt} = a * Z_t + sigma_ou * eps
        a = np.exp(-self.kappa * _DT)
        sigma_ou = np.sqrt((1.0 - a**2) / (2.0 * self.kappa))

        Y = np.full(n_paths, self.S0)
        Z = np.zeros(n_paths)  # OU factor starts at 0

        for i in range(1, N_steps + 1):
            t_current = (i - 1) * _DT  # time at start of this interval

            # Compute instantaneous variance at t_current
            xi = self._xi0(t_current)
            corr = self._correction(t_current)
            V = xi * np.exp(self.eta * Z - corr)

            sq_V = np.sqrt(np.maximum(V, 0.0))

            # Generate correlated noise
            eps_B = rng.standard_normal(n_paths)      # OU / leverage noise
            eps_perp = rng.standard_normal(n_paths)   # orthogonal price noise

            # Exact OU update
            Z = a * Z + sigma_ou * eps_B

            # Euler-Maruyama price update
            Y = Y * np.exp(
                -0.5 * V * _DT
                + sq_V * sqrt_dt * (self.rho * eps_B + rho_perp * eps_perp)
            )

            if i in target_steps:
                j = target_steps[i]
                Y_out[:, j] = Y
                L_out[:, j, 0] = Z

        return Y_out, L_out


class BassProcess:
    """Bass (log-normal reference) process.

    A simple geometric Brownian motion used as a baseline reference:

        dY_t = sigma_avg * Y_t * dW_t
        Y_t  = exp(-0.5 * sigma_avg^2 * t + sigma_avg * W_t)

    This is simulated exactly at each Euler step:
        Y_{t+dt} = Y_t * exp(-0.5 * sigma_avg^2 * dt + sigma_avg * sqrt(dt) * Z)

    Parameters
    ----------
    sigma_avg : float, optional
        Constant log-normal volatility. Default is 0.20.
    S0 : float, optional
        Initial forward price. Default is 1.0.
    """

    def __init__(self, sigma_avg: float = 0.20, S0: float = 1.0) -> None:
        self.sigma_avg = sigma_avg
        self.S0 = S0

    def simulate(
        self,
        T_dates: list[float],
        n_paths: int,
        seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Simulate the Bass (log-normal) process.

        Parameters
        ----------
        T_dates : list[float]
            Target observation dates in years.
        n_paths : int
            Number of Monte Carlo paths.
        seed : int, optional
            RNG seed for reproducibility. Default is 42.

        Returns
        -------
        Y : np.ndarray, shape (n_paths, len(T_dates))
            Forward price process at each target date.
        L : np.ndarray, shape (n_paths, len(T_dates), 0)
            Empty factor array (no latent factors).
        """
        T_dates_sorted = sorted(T_dates)
        T_max = T_dates_sorted[-1]
        N_steps = int(round(T_max / _DT))
        N_dates = len(T_dates_sorted)

        Y_out = np.zeros((n_paths, N_dates))
        L_out = np.zeros((n_paths, N_dates, 0))
        target_steps = _build_target_step_map(T_dates_sorted)

        rng = np.random.default_rng(seed)
        sigma = self.sigma_avg
        drift = -0.5 * sigma**2 * _DT
        vol = sigma * np.sqrt(_DT)

        Y = np.full(n_paths, self.S0)

        for i in range(1, N_steps + 1):
            Z = rng.standard_normal(n_paths)
            Y = Y * np.exp(drift + vol * Z)

            if i in target_steps:
                j = target_steps[i]
                Y_out[:, j] = Y

        return Y_out, L_out
