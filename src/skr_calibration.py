"""
Sequential Kernel Reconstruction (SKR) calibration algorithm.

Calibrates a reference stochastic process to match target risk-neutral marginals
at a sequence of dates T_1 < T_2 < ... < T_5 using a martingale Sinkhorn iteration.

For each interval [T_i, T_{i+1}] the algorithm finds:
  - a_values: per-particle shift such that the conditional martingale constraint holds
  - g_func:   quantile transport map from the shifted terminal distribution to the
              target marginal mu_{i+1}

Reference:
  "Sequential Kernel Reconstruction for Volatility Model Calibration" (SKR paper)
"""

import warnings
import numpy as np
from scipy.spatial import KDTree
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# Public helper functions
# ---------------------------------------------------------------------------

def build_quantile_map(source_samples: np.ndarray, target_samples: np.ndarray):
    """Build a monotone quantile transport map g: source -> target.

    Sorts both sample arrays and returns a piecewise-linear interpolant so that
    g(source_sorted[i]) == target_sorted[i].  Values outside the source support
    are clamped to the extreme target values.

    Parameters
    ----------
    source_samples : array_like, shape (M,)
    target_samples : array_like, shape (M,) or (N,) – need not be same length

    Returns
    -------
    g : callable
        g(z) maps a scalar or ndarray z into the target distribution.
    """
    src_sorted = np.sort(np.asarray(source_samples, dtype=float).ravel())
    tgt_sorted = np.sort(np.asarray(target_samples, dtype=float).ravel())

    if len(src_sorted) != len(tgt_sorted):
        # Re-quantile: build a common uniform grid
        u = np.linspace(0, 1, max(len(src_sorted), len(tgt_sorted)))
        src_sorted = np.quantile(src_sorted, u)
        tgt_sorted = np.quantile(tgt_sorted, u)

    def g(z):
        z_arr = np.asarray(z, dtype=float)
        return np.interp(z_arr, src_sorted, tgt_sorted)

    # Attach the sorted arrays so SKRInterval can serialise them
    g._src_sorted = src_sorted
    g._tgt_sorted = tgt_sorted
    return g


def conditional_expectation_knn(
    g_func,
    a_candidate: float,
    Y_to_neighbors: np.ndarray,
) -> float:
    """Estimate E[g(a + Y_to) | R_{T_i}] using K nearest-neighbour samples.

    Parameters
    ----------
    g_func : callable
        Quantile map g.
    a_candidate : float
        Candidate shift value.
    Y_to_neighbors : array_like, shape (K,)
        Y_to values at the K nearest neighbours of the current particle.

    Returns
    -------
    float
        Estimated conditional expectation.
    """
    return float(np.mean(g_func(a_candidate + np.asarray(Y_to_neighbors))))


# ---------------------------------------------------------------------------
# Core Martingale Sinkhorn iteration
# ---------------------------------------------------------------------------

def _build_state_matrix(Y: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Concatenate and normalise [Y, L] into the KNN state space.

    Parameters
    ----------
    Y : (N,) array
    L : (N, n_factors) array  – may have n_factors == 0

    Returns
    -------
    state : (N, 1 + n_factors) array, each column normalised by its std.
    """
    cols = [Y.reshape(-1, 1)]
    if L.ndim == 2 and L.shape[1] > 0:
        cols.append(L)
    state = np.concatenate(cols, axis=1).astype(float)
    stds = state.std(axis=0)
    stds[stds < 1e-12] = 1.0
    return state / stds


def _vectorised_bisect(f_vec, lo: np.ndarray, hi: np.ndarray, n_steps: int = 60):
    """Vectorised bisection root finder.

    Finds x in [lo, hi] such that f_vec(x) == 0 for each element independently.
    Assumes f_vec(lo) <= 0 and f_vec(hi) >= 0 element-wise (monotone increasing f).

    Parameters
    ----------
    f_vec : callable (N,) -> (N,)
    lo, hi : (N,) arrays – initial brackets
    n_steps : int – number of bisection steps (60 gives ~1e-18 relative precision)

    Returns
    -------
    roots : (N,) array
    """
    lo = lo.copy().astype(float)
    hi = hi.copy().astype(float)
    f_lo = f_vec(lo)

    for _ in range(n_steps):
        mid = 0.5 * (lo + hi)
        f_mid = f_vec(mid)
        # Root in [lo, mid] when f_lo * f_mid <= 0, else root in [mid, hi]
        move_hi = f_lo * f_mid <= 0
        hi = np.where(move_hi, mid, hi)
        lo = np.where(move_hi, lo, mid)
        f_lo = np.where(move_hi, f_lo, f_mid)

    return 0.5 * (lo + hi)


def _expand_brackets(
    f_vec,
    lo: np.ndarray,
    hi: np.ndarray,
    max_expansions: int = 20,
) -> tuple:
    """Expand brackets for particles where f(lo)*f(hi) > 0."""
    lo, hi = lo.copy(), hi.copy()
    width = hi - lo

    for _ in range(max_expansions):
        f_lo = f_vec(lo)
        f_hi = f_vec(hi)
        not_bracketed = f_lo * f_hi > 0
        if not not_bracketed.any():
            break
        # If both too large (f_lo > 0 too), shift left; if both too small, shift right
        both_pos = not_bracketed & (f_lo > 0)
        both_neg = not_bracketed & (f_lo < 0)
        lo = np.where(both_pos, lo - width, lo)
        hi = np.where(both_neg, hi + width, hi)
        width *= 2.0

    return lo, hi


def martingale_sinkhorn(
    Y_from: np.ndarray,
    Y_to: np.ndarray,
    L_from: np.ndarray,
    L_to: np.ndarray,
    X_from: np.ndarray,
    mu_to_samples: np.ndarray,
    n_neighbors: int = 50,
    max_iter: int = 50,
    tol: float = 1e-5,
) -> tuple:
    """Martingale Sinkhorn algorithm for one calibration interval [T_i, T_{i+1}].

    Finds per-particle shifts *a* and a quantile map *g* such that the
    transported process X_{T_{i+1}} = g(a + Y_to) is a martingale with respect
    to the reference filtration and has the correct marginal mu_{i+1}.

    Parameters
    ----------
    Y_from : (N,)  – reference process values at T_i
    Y_to   : (N,)  – reference process values at T_{i+1}
    L_from : (N, n_factors) – factor states at T_i  (may be (N, 0))
    L_to   : (N, n_factors) – factor states at T_{i+1}
    X_from : (N,)  – calibrated X values at T_i  (defines kappa_i)
    mu_to_samples : (M,) – i.i.d. samples from the target marginal at T_{i+1}
    n_neighbors : int   – K nearest neighbours for conditional expectation
    max_iter    : int   – maximum Sinkhorn iterations
    tol         : float – convergence tolerance on max|a_new - a|

    Returns
    -------
    g_func   : callable  – final quantile map  g: Z-space -> mu_{i+1}-space
    a_values : (N,)      – calibrated per-particle shifts
    history  : list[dict] – convergence log, one dict per iteration
    """
    N = len(Y_from)
    Y_from = np.asarray(Y_from, dtype=float)
    Y_to = np.asarray(Y_to, dtype=float)
    X_from = np.asarray(X_from, dtype=float)
    mu_to_samples = np.asarray(mu_to_samples, dtype=float)

    # Ensure L arrays are 2-D
    if L_from.ndim == 1:
        L_from = L_from.reshape(N, -1)
    if L_to.ndim == 1:
        L_to = L_to.reshape(N, -1)

    # --- Build KDTree on the (Y_from, L_from) state space (done once) ---
    state = _build_state_matrix(Y_from, L_from)
    tree = KDTree(state)
    k_query = min(n_neighbors + 1, N)   # +1 to drop self-match
    _, nn_idx = tree.query(state, k=k_query)
    # Drop the first column (the particle itself, distance 0)
    nn_idx = nn_idx[:, 1:]              # shape (N, n_neighbors)

    # Pre-fetch the Y_to values for every particle's neighbourhood
    # neighbour_Y_to[i, k] = Y_to[nn_idx[i, k]]
    neighbour_Y_to = Y_to[nn_idx]       # shape (N, n_neighbors)

    # Clamp X_from to the interior of mu_to to guarantee bracket existence
    mu_min = float(mu_to_samples.min())
    mu_max = float(mu_to_samples.max())
    X_clamped = np.clip(X_from, mu_min + 1e-10, mu_max - 1e-10)

    y_std = float(np.std(Y_to)) if np.std(Y_to) > 1e-12 else 1.0
    search_width = 5.0 * y_std

    # --- Initialise shifts ---
    a = np.zeros(N)
    history = []
    g_func = None

    for k_iter in range(max_iter):
        # (a) Compute shifted terminal distribution
        Z_k = a + Y_to                              # (N,)

        # (b) Terminal update: build quantile map g_k: Z_k -> mu_to
        g_func = build_quantile_map(Z_k, mu_to_samples)

        # (c) Martingale update via vectorised bisection
        #     For each particle i solve:  E[g_k(a_new_i + Y_to) | R_i] = X_from[i]
        #     Vectorised over all i simultaneously.

        def _f_vec(a_test: np.ndarray) -> np.ndarray:
            """(N,) -> (N,): conditional expectation residual."""
            # shifted[i, k] = a_test[i] + neighbour_Y_to[i, k]
            shifted = a_test[:, np.newaxis] + neighbour_Y_to      # (N, K)
            g_vals = g_func(shifted.ravel()).reshape(N, -1)        # (N, K)
            return g_vals.mean(axis=1) - X_clamped                 # (N,)

        # Initial bracket centred on 0, width ±search_width
        lo_init = np.full(N, -search_width)
        hi_init = np.full(N, +search_width)

        # Expand brackets where needed (should be rare after iteration 1)
        lo_init, hi_init = _expand_brackets(_f_vec, lo_init, hi_init)

        a_new = _vectorised_bisect(_f_vec, lo_init, hi_init, n_steps=52)

        # (d) Centre shifts to remove global drift
        a_new -= a_new.mean()

        # (e) Convergence check
        change = np.abs(a_new - a)
        max_change = float(change.max())
        mean_residual = float(np.abs(_f_vec(a_new)).mean())
        history.append({
            "iter": k_iter,
            "max_change": max_change,
            "mean_residual": mean_residual,
        })

        a = a_new

        if max_change < tol:
            break

    # Rebuild g_func with the final shifts for cleanliness
    Z_final = a + Y_to
    g_func = build_quantile_map(Z_final, mu_to_samples)

    return g_func, a, history


# ---------------------------------------------------------------------------
# SKRInterval – calibration data for one interval
# ---------------------------------------------------------------------------

class SKRInterval:
    """Holds all calibration objects for one interval [T_from, T_to].

    Attributes
    ----------
    T_from, T_to : float – interval endpoints
    g_func       : callable – final quantile map  g: shifted_Z -> mu_{T_to}
    a_values     : (N,) ndarray – per-particle shift
    convergence_history : list[dict] – Sinkhorn convergence log
    n_iter       : int – number of Sinkhorn iterations taken
    """

    def __init__(
        self,
        T_from: float,
        T_to: float,
        g_func,
        a_values: np.ndarray,
        convergence_history: list | None = None,
        n_iter: int = 0,
    ):
        self.T_from = T_from
        self.T_to = T_to
        self.g_func = g_func
        self.a_values = np.asarray(a_values, dtype=float)
        self.convergence_history = convergence_history if convergence_history is not None else []
        self.n_iter = n_iter

        # Cache the sorted arrays for fast interp (bypassing closure overhead)
        self._g_src: np.ndarray = getattr(g_func, "_src_sorted", None)
        self._g_tgt: np.ndarray = getattr(g_func, "_tgt_sorted", None)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def apply_g(self, z_array: np.ndarray) -> np.ndarray:
        """Apply the quantile map g to an array of z-values.

        Parameters
        ----------
        z_array : array_like

        Returns
        -------
        ndarray – same shape as z_array
        """
        z = np.asarray(z_array, dtype=float)
        if self._g_src is not None:
            return np.interp(z, self._g_src, self._g_tgt)
        return self.g_func(z)

    def compute_shifted_terminal(self, Y_to_new: np.ndarray) -> np.ndarray:
        """Compute calibrated X_{T_{i+1}} on a new batch of Y_to samples.

        Uses the stored per-particle shifts a and the calibrated map g:
            X_{T_{i+1}} = g(a + Y_to_new)

        Parameters
        ----------
        Y_to_new : (N,) array – reference process increments at T_{i+1}

        Returns
        -------
        (N,) array – calibrated X values
        """
        return self.apply_g(self.a_values + np.asarray(Y_to_new, dtype=float))

    def compute_X_at_T(
        self,
        X_from: np.ndarray,
        Y_from: np.ndarray,
        L_from: np.ndarray,
        Y_to: np.ndarray,
        L_to: np.ndarray,
    ) -> np.ndarray:
        """Compute X_{T_{i+1}} for the calibration paths using the fitted bridge.

        For the calibration particle set the stored ``a_values`` already encode
        the full solution, so we simply return ``g(a + Y_to)``.

        Parameters
        ----------
        X_from : (N,) – calibrated X at T_from  (unused; kept for API symmetry)
        Y_from : (N,) – reference Y at T_from    (unused; kept for API symmetry)
        L_from : (N, d) – factor states at T_from (unused)
        Y_to   : (N,) – reference Y at T_to
        L_to   : (N, d) – factor states at T_to  (unused)

        Returns
        -------
        (N,) array – calibrated X_{T_{i+1}}
        """
        return self.compute_shifted_terminal(Y_to)

    def __repr__(self) -> str:
        if self.convergence_history:
            conv = (
                f"converged in {self.n_iter} iters, "
                f"final_max_change={self.convergence_history[-1]['max_change']:.2e}"
            )
        else:
            conv = f"n_iter={self.n_iter}"
        return f"SKRInterval([{self.T_from}, {self.T_to}], {conv})"


# ---------------------------------------------------------------------------
# SKRCalibrator – top-level calibration orchestrator
# ---------------------------------------------------------------------------

class SKRCalibrator:
    """Main SKR calibration class.

    Orchestrates the simulation of a reference process and the sequential
    martingale Sinkhorn calibration across all intervals.

    Parameters
    ----------
    model_name : str
        Reference model identifier ('bass', 'heston', '3_2', 'bergomi').
    surface : SPXSmileSurface
        Target implied-volatility surface; must expose
        ``get_marginal_samples(T, N) -> ndarray``.
    n_paths : int
        Monte Carlo path count.
    n_neighbors : int
        KNN neighbourhood size for conditional expectation.
    max_iter : int
        Maximum Sinkhorn iterations per interval.
    tol : float
        Convergence tolerance.
    seed : int
        RNG seed for reproducibility.
    r : float
        Continuously compounded risk-free rate (used for repricing only).

    Attributes
    ----------
    T_dates : list[float]
        Calibration maturities [0.25, 0.5, 1.0, 2.0, 3.0].
    intervals : list[SKRInterval]
        One fitted interval per calibration step (populated after ``calibrate``).
    X_paths : ndarray, shape (n_paths, 5)
        Calibrated discounted-spot paths at the five maturities.
    Y_paths : ndarray, shape (n_paths, 5)
        Reference process values at the five maturities.
    L_paths : ndarray, shape (n_paths, 5, n_factors)
        Factor-state paths at the five maturities.
    """

    T_dates = [0.25, 0.5, 1.0, 2.0, 3.0]

    def __init__(
        self,
        model_name: str,
        surface,
        n_paths: int = 100_000,
        n_neighbors: int = 50,
        max_iter: int = 50,
        tol: float = 1e-5,
        seed: int = 42,
        r: float = 0.0,
    ):
        self.model_name = model_name
        self.surface = surface
        self.n_paths = n_paths
        self.n_neighbors = n_neighbors
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.r = r

        # Populated after simulate_reference / calibrate
        self.Y_paths: np.ndarray = None    # (N, 5)
        self.L_paths: np.ndarray = None    # (N, 5, n_factors)
        self._Y_all: np.ndarray = None     # (N, 6) – includes t=0
        self._L_all: np.ndarray = None     # (N, 6, n_factors) – includes t=0
        self.X_paths: np.ndarray = None    # (N, 5)
        self.intervals: list = []

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate_reference(self, process) -> None:
        """Simulate the reference process and store Y_paths, L_paths.

        Calls ``process.simulate(T_dates_full, n_paths, seed)`` where
        ``T_dates_full = [0.0] + self.T_dates`` (six dates including T_0=0).

        The process must return:
            Y_paths_full : ndarray (N, 6) – spot/price values at all dates
            L_paths_full : ndarray (N, 6, n_factors) – factor states at all dates

        After this call, ``self.Y_paths`` and ``self.L_paths`` hold the values
        at the five calibration maturities (T_1 … T_5).
        """
        T_full = [0.0] + self.T_dates       # 6 dates
        Y_full, L_full = process.simulate(T_full, self.n_paths, self.seed)

        Y_full = np.asarray(Y_full, dtype=float)
        L_full = np.asarray(L_full, dtype=float)

        if Y_full.ndim == 1:
            # Single-path fallback – shouldn't happen in practice
            Y_full = Y_full.reshape(1, -1)
        if L_full.ndim == 2:
            # No latent factors – add empty factor dimension
            L_full = L_full.reshape(*L_full.shape, 0)

        # Validate shapes
        assert Y_full.shape == (self.n_paths, 6), (
            f"Expected Y shape ({self.n_paths}, 6), got {Y_full.shape}"
        )

        self._Y_all = Y_full                 # (N, 6) – columns: T_0…T_5
        self._L_all = L_full                 # (N, 6, d)
        self.Y_paths = Y_full[:, 1:]         # (N, 5)
        self.L_paths = L_full[:, 1:, :]     # (N, 5, d)

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, process) -> None:
        """Full SKR calibration over all five intervals.

        Steps:
        1. Simulate the reference process.
        2. Retrieve target marginal samples from the surface at each maturity.
        3. Run the martingale Sinkhorn algorithm for each interval.
        4. Accumulate calibrated X paths and SKRInterval objects.

        Parameters
        ----------
        process : reference model object
            Must implement ``simulate(T_dates_full, n_paths, seed)``.
        """
        self.simulate_reference(process)

        N = self.n_paths
        n_factors = self._L_all.shape[2]

        self.X_paths = np.zeros((N, 5))
        self.intervals = []

        # Initial state: X_0 = 1  (normalised spot),  Y_0 and L_0 from simulation
        X_from = np.ones(N)
        Y_from = self._Y_all[:, 0]          # (N,) at T_0
        L_from = self._L_all[:, 0, :]       # (N, d) at T_0

        for i in range(5):
            T_from = 0.0 if i == 0 else self.T_dates[i - 1]
            T_to = self.T_dates[i]

            Y_to = self.Y_paths[:, i]        # (N,) at T_{i+1}
            L_to = self.L_paths[:, i, :]    # (N, d) at T_{i+1}

            # Target marginal samples from the calibrated surface
            mu_to_samples = self.surface.get_marginal_samples(T_to, N)

            print(
                f"  [SKR] Calibrating interval [{T_from:.2f}, {T_to:.2f}]  …",
                flush=True,
            )

            g_func, a_values, history = martingale_sinkhorn(
                Y_from,
                Y_to,
                L_from,
                L_to,
                X_from,
                mu_to_samples,
                n_neighbors=self.n_neighbors,
                max_iter=self.max_iter,
                tol=self.tol,
            )

            interval = SKRInterval(
                T_from=T_from,
                T_to=T_to,
                g_func=g_func,
                a_values=a_values,
                convergence_history=history,
                n_iter=len(history),
            )
            self.intervals.append(interval)

            # Propagate calibrated paths forward
            X_to = interval.compute_X_at_T(X_from, Y_from, L_from, Y_to, L_to)
            self.X_paths[:, i] = X_to

            print(
                f"  [SKR]   done  n_iter={interval.n_iter}  "
                f"final_max_change={history[-1]['max_change']:.2e}",
                flush=True,
            )

            # Advance state
            X_from = X_to
            Y_from = Y_to
            L_from = L_to

    # ------------------------------------------------------------------
    # Repricing
    # ------------------------------------------------------------------

    def reprice_calls(self, T: float, K_grid: np.ndarray) -> np.ndarray:
        """Reprice European call options at maturity T over a strike grid.

        Parameters
        ----------
        T : float – maturity; must be one of ``self.T_dates``
        K_grid : (M,) array – relative strikes (K / S_0)

        Returns
        -------
        (M,) array – call prices (discounted expectation)
        """
        idx = self.T_dates.index(T)
        X = self.X_paths[:, idx]            # calibrated discounted spot (N,)
        df = np.exp(-self.r * T)
        K_grid = np.asarray(K_grid, dtype=float)
        return np.array([np.mean(np.maximum(X - K, 0.0)) * df for K in K_grid])

    def compute_implied_vols(self, T: float, K_grid: np.ndarray) -> np.ndarray:
        """Compute Black-Scholes implied vols from repriced calls.

        Uses ``src.utils.implied_vol_brent`` when available; falls back to
        an inline Brent-bisection implementation otherwise.

        Parameters
        ----------
        T : float – maturity (must be in T_dates)
        K_grid : (M,) array – relative strikes (normalised so S_0 = 1)

        Returns
        -------
        (M,) array – implied volatilities  (NaN where inversion fails)
        """
        try:
            from .utils import implied_vol_brent as _iv  # noqa: PLC0415
            # utils signature: implied_vol_brent(price, S, K, T, r, ...)
            def _call_iv(price, K):
                return _iv(price, 1.0, K, T, self.r)
        except ImportError:
            # fallback: inline implementation with (price, T, K, F, r) args
            def _call_iv(price, K):
                return _bs_implied_vol_brent(price, T, K, np.exp(self.r * T), self.r)

        K_grid = np.asarray(K_grid, dtype=float)
        prices = self.reprice_calls(T, K_grid)

        vols = np.empty_like(prices)
        for j, (K, price) in enumerate(zip(K_grid, prices)):
            try:
                vols[j] = _call_iv(price, K)
            except Exception:
                vols[j] = np.nan
        return vols


# ---------------------------------------------------------------------------
# Inline Black-Scholes implied vol fallback
# ---------------------------------------------------------------------------

def _bs_call_price(sigma: float, T: float, K: float, F: float, r: float) -> float:
    """Log-normal Black-Scholes call price (S0 = 1, undiscounted forward F)."""
    from scipy.stats import norm  # noqa: PLC0415

    if sigma <= 0 or T <= 0:
        return max(F * np.exp(-r * T) - K * np.exp(-r * T), 0.0)
    s_sqrt_T = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / s_sqrt_T
    d2 = d1 - s_sqrt_T
    return np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))


def _bs_implied_vol_brent(
    price: float,
    T: float,
    K: float,
    F: float,
    r: float,
    tol: float = 1e-8,
) -> float:
    """Invert Black-Scholes call price to implied vol using Brent's method."""
    from scipy.optimize import brentq  # noqa: PLC0415

    intrinsic = max(np.exp(-r * T) * (F - K), 0.0)
    if price <= intrinsic + 1e-12:
        return np.nan

    def objective(sigma):
        return _bs_call_price(sigma, T, K, F, r) - price

    try:
        return brentq(objective, 1e-6, 10.0, xtol=tol)
    except ValueError:
        return np.nan
