"""
ESSVI (Extended SVI) arbitrage-free smile parametrization for the SPX surface
dated 23 June 2020.

The ESSVI total implied variance formula is:

    w(k) = (theta/2) * (1 + rho*phi*k + sqrt((phi*k + rho)^2 + (1 - rho^2)))

where k = log(K/F) is the log-moneyness, and (theta, rho, phi) are
maturity-dependent ESSVI parameters.

Parameters are calibrated to SPX-like dynamics as of 23 June 2020
(post-COVID high-volatility regime).
"""

import numpy as np
from scipy.interpolate import interp1d

from src.utils import normal_cdf, black_scholes_call, implied_vol_brent


# ESSVI parameters per maturity: (theta, rho, phi)
_ESSVI_PARAMS: dict[float, tuple[float, float, float]] = {
    0.25: (0.0408, -0.70, 1.80),
    0.50: (0.0598, -0.72, 1.50),
    1.00: (0.0988, -0.75, 1.10),
    2.00: (0.1693, -0.77, 0.80),
    3.00: (0.2513, -0.78, 0.65),
}


class ESSVISmile:
    """A single-maturity ESSVI implied volatility smile slice.

    Parametrizes the total implied variance as:

        w(k) = (theta/2) * (1 + rho*phi*k + sqrt((phi*k + rho)^2 + 1 - rho^2))

    All prices and densities are in the forward-normalized measure (F = 1).

    Parameters
    ----------
    T : float
        Maturity in years.
    theta : float
        ATM total variance level.
    rho : float
        Correlation / skew parameter in (-1, 1).
    phi : float
        Curvature / wings parameter (> 0).
    F : float, optional
        Forward price. Default is 1.0.
    r : float, optional
        Risk-free rate (for discounting). Default is 0.02.
    """

    def __init__(
        self,
        T: float,
        theta: float,
        rho: float,
        phi: float,
        F: float = 1.0,
        r: float = 0.02,
    ) -> None:
        self.T = T
        self.theta = theta
        self.rho = rho
        self.phi = phi
        self.F = F
        self.r = r

    def total_variance(self, k: np.ndarray) -> np.ndarray:
        """ESSVI total implied variance w(k).

        Parameters
        ----------
        k : np.ndarray
            Log-moneyness k = log(K/F).

        Returns
        -------
        np.ndarray
            w(k) = (theta/2) * (1 + rho*phi*k + sqrt((phi*k + rho)^2 + 1 - rho^2)).
        """
        k = np.asarray(k, dtype=float)
        discriminant = (self.phi * k + self.rho) ** 2 + (1.0 - self.rho**2)
        return 0.5 * self.theta * (
            1.0 + self.rho * self.phi * k + np.sqrt(discriminant)
        )

    def implied_vol(self, k: np.ndarray) -> np.ndarray:
        """ESSVI implied volatility sigma(k).

        Parameters
        ----------
        k : np.ndarray
            Log-moneyness k = log(K/F).

        Returns
        -------
        np.ndarray
            sigma(k) = sqrt(w(k) / T).
        """
        return np.sqrt(self.total_variance(k) / self.T)

    def call_price(self, K_array: np.ndarray) -> np.ndarray:
        """Forward-normalized call prices C(K) = E[max(F - K, 0)] * exp(-rT).

        Uses the Black-Scholes formula with the ESSVI implied volatility,
        with spot S = F*exp(-rT) so that the discounted forward equals F*exp(-rT).

        Parameters
        ----------
        K_array : np.ndarray
            Strike prices (in forward-normalized units, i.e. around F = 1).

        Returns
        -------
        np.ndarray
            Call option prices for each strike.
        """
        K_array = np.asarray(K_array, dtype=float)
        prices = np.empty_like(K_array)
        S = self.F * np.exp(-self.r * self.T)  # spot consistent with F
        for i, K in enumerate(K_array):
            if K <= 0.0:
                prices[i] = max(S - K * np.exp(-self.r * self.T), 0.0)
                continue
            k = np.log(K / self.F)
            sigma = float(self.implied_vol(np.array([k]))[0])
            if sigma <= 0.0 or not np.isfinite(sigma):
                prices[i] = max(S - K * np.exp(-self.r * self.T), 0.0)
            else:
                prices[i] = black_scholes_call(S, K, self.T, self.r, sigma)
        return prices

    def risk_neutral_density(
        self,
        k_grid: np.ndarray,
        dk: float,
    ) -> np.ndarray:
        """Risk-neutral density via Breeden-Litzenberger second derivative.

        Approximates d^2C/dK^2 numerically on the log-moneyness grid.
        The density is expressed in terms of the forward-normalized spot F*exp(k).

        Parameters
        ----------
        k_grid : np.ndarray
            Uniform log-moneyness grid (must have at least 3 points).
        dk : float
            Grid spacing (must match k_grid spacing).

        Returns
        -------
        np.ndarray
            Risk-neutral density p(K) on the K-grid = F * exp(k_grid),
            normalised so that integral p(K) dK ≈ 1.
        """
        K_grid = self.F * np.exp(k_grid)
        prices = self.call_price(K_grid)

        # Second derivative d^2C/dK^2 via central differences in K-space
        d2C_dK2 = np.empty_like(prices)
        dK = np.diff(K_grid)
        # Interior points: central difference
        for i in range(1, len(K_grid) - 1):
            h_fwd = K_grid[i + 1] - K_grid[i]
            h_bwd = K_grid[i] - K_grid[i - 1]
            d2C_dK2[i] = 2.0 * (
                prices[i + 1] / (h_fwd * (h_fwd + h_bwd))
                - prices[i] / (h_fwd * h_bwd)
                + prices[i - 1] / (h_bwd * (h_fwd + h_bwd))
            )
        # Boundary: replicate nearest interior
        d2C_dK2[0] = d2C_dK2[1]
        d2C_dK2[-1] = d2C_dK2[-2]

        # Discount factor converts risk-neutral density under Q
        density = np.exp(self.r * self.T) * d2C_dK2
        density = np.maximum(density, 0.0)

        # Normalise
        total = np.trapz(density, K_grid)
        if total > 0.0:
            density /= total
        return density

    def build_marginal(
        self,
        n_quantiles: int = 2000,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, interp1d]:
        """Build an empirical risk-neutral marginal distribution.

        Computes the risk-neutral density on a fine log-moneyness grid,
        integrates it to get the CDF, and constructs a quantile (inverse-CDF)
        interpolant for Monte Carlo sampling.

        Parameters
        ----------
        n_quantiles : int, optional
            Number of grid points for the density/CDF computation. Default 2000.

        Returns
        -------
        k_grid : np.ndarray
            Log-moneyness grid.
        density : np.ndarray
            Risk-neutral density values on K_grid = F * exp(k_grid).
        cdf : np.ndarray
            Cumulative distribution function values on k_grid.
        quantile_func : scipy.interpolate.interp1d
            Callable Q(p) -> log-moneyness, mapping probabilities to log-moneyness.
        """
        # Wide enough to cover meaningful probability mass
        k_min, k_max = -1.5, 1.0
        k_grid = np.linspace(k_min, k_max, n_quantiles)
        dk = k_grid[1] - k_grid[0]

        density = self.risk_neutral_density(k_grid, dk)

        # Integrate density over dK = F*exp(k)*dk to get CDF in log-moneyness
        K_grid = self.F * np.exp(k_grid)
        # p(K) dK -> convert to p in terms of k: p_k(k) = p(K) * K (Jacobian)
        density_k = density * K_grid  # density w.r.t. log-moneyness

        # Re-normalise in log-moneyness space
        total_k = np.trapz(density_k, k_grid)
        if total_k > 0.0:
            density_k /= total_k

        # Build CDF via cumulative trapezoidal integration
        cdf = np.zeros(n_quantiles)
        for i in range(1, n_quantiles):
            cdf[i] = cdf[i - 1] + 0.5 * (density_k[i - 1] + density_k[i]) * (
                k_grid[i] - k_grid[i - 1]
            )
        cdf /= cdf[-1]  # ensure CDF ends at 1

        # Quantile function: p -> k (remove duplicate cdf values for interp)
        unique_mask = np.concatenate(([True], np.diff(cdf) > 0))
        quantile_func = interp1d(
            cdf[unique_mask],
            k_grid[unique_mask],
            kind="linear",
            bounds_error=False,
            fill_value=(k_grid[0], k_grid[-1]),
        )
        return k_grid, density, cdf, quantile_func


class SPXSmileSurface:
    """SPX implied volatility surface dated 23 June 2020.

    Holds five ESSVI smile slices at maturities T = 0.25, 0.5, 1.0, 2.0, 3.0.

    Attributes
    ----------
    maturities : list[float]
        Sorted list of calibration maturities in years.
    smiles : dict[float, ESSVISmile]
        Dictionary mapping maturity to ESSVISmile instance.
    """

    maturities: list[float] = [0.25, 0.5, 1.0, 2.0, 3.0]

    def __init__(self, F: float = 1.0, r: float = 0.02) -> None:
        self.F = F
        self.r = r
        self.smiles: dict[float, ESSVISmile] = {}
        for T, (theta, rho, phi) in _ESSVI_PARAMS.items():
            self.smiles[T] = ESSVISmile(T, theta, rho, phi, F=F, r=r)

    def get_smile(self, T: float) -> ESSVISmile:
        """Return the ESSVISmile for a given maturity.

        Parameters
        ----------
        T : float
            Maturity in years; must be one of {0.25, 0.5, 1.0, 2.0, 3.0}.

        Returns
        -------
        ESSVISmile
            Calibrated ESSVI smile for maturity T.

        Raises
        ------
        KeyError
            If T is not a supported maturity.
        """
        if T not in self.smiles:
            raise KeyError(
                f"Maturity T={T} not available. Choose from {self.maturities}."
            )
        return self.smiles[T]

    def get_marginal_samples(
        self,
        T: float,
        n_samples: int = 100_000,
        seed: int = 0,
    ) -> np.ndarray:
        """Sample forward prices from the risk-neutral density at maturity T.

        Uses the inverse-CDF (quantile) method: draws U ~ Uniform(0,1)
        and returns F * exp(Q(U)) where Q is the ESSVI quantile function.

        Parameters
        ----------
        T : float
            Maturity in years.
        n_samples : int, optional
            Number of Monte Carlo samples. Default is 100,000.
        seed : int, optional
            Random seed for reproducibility. Default is 0.

        Returns
        -------
        np.ndarray
            Shape (n_samples,). Forward-price samples Y ~ Q^T (risk-neutral
            marginal at maturity T), normalised so E[Y] ≈ 1.
        """
        smile = self.get_smile(T)
        _, _, _, quantile_func = smile.build_marginal(n_quantiles=2000)

        rng = np.random.default_rng(seed)
        u = rng.uniform(0.0, 1.0, size=n_samples)
        k_samples = quantile_func(u)
        S_samples = self.F * np.exp(k_samples)
        return S_samples

    def verify_convex_order(self) -> bool:
        """Check that marginal variances are increasing in T (necessary condition
        for convex ordering).

        For each pair of consecutive maturities (T1 < T2), checks that the
        ATM total implied variance w(0; T1) < w(0; T2), which is a necessary
        condition for the family of marginals {Q^T} to be in convex order.

        Returns
        -------
        bool
            True if the surface satisfies the necessary convex-order condition.
        """
        print("Convex order verification (ATM total variance by maturity):")
        prev_w = -np.inf
        result = True
        for T in self.maturities:
            smile = self.smiles[T]
            w_atm = float(smile.total_variance(np.array([0.0]))[0])
            status = "OK" if w_atm > prev_w else "FAIL"
            if status == "FAIL":
                result = False
            print(f"  T={T:.2f}: w(0)={w_atm:.6f}  [{status}]")
            prev_w = w_atm
        print(f"Convex order: {'SATISFIED' if result else 'VIOLATED'}")
        return result
