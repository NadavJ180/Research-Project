"""
DVR_Algorithm_1_5.py
=====================================================================
WHAT THIS FILE DOES
---------------------------------------------------------------------
Core 1-D Discrete Variable Representation (DVR) engine for smooth
potentials. Builds the Colbert-Miller sinc-DVR Hamiltonian on a
grid and returns the lowest `num_levels` energy eigenvalues.

CHANGELOG (v1.4 -> v1.5)
---------------------------------------------------------------------
- REVERTED E_ceiling from the user's temporary hack of
  `1000.0 * num_levels` back to the physically correct
  `1.5 * num_levels`. Inflating E_ceiling was wrong for two reasons:
    1. It pushed classical turning points absurdly far (±710 units
       for the double well), making the initial grid span far larger
       than needed.
    2. It made k_max enormous → dx_target microscopic → n_grid
       astronomical (tens of thousands of points) with no benefit.

- ADDED adaptive span expansion loop to `auto_configure_dvr`.
  After the initial grid estimate, the function now iteratively
  expands the span (by `span_growth_factor` per step, default 1.3×)
  and checks convergence by comparing eigenvalues on the current
  grid against the same computation on a wider grid. This correctly
  handles potentials (double well, Morse, quartic) whose wavefunction
  tails extend much further than the HO-calibrated padding heuristic
  would predict. Key design decisions:
    - `dx_target` is FIXED before the expansion loop so that grid
      resolution and span are decoupled (widening the grid adds
      more points, not coarser ones).
    - 1 DVR solve per expansion step (not 2): the "wider" result
      becomes the "current" for the next step, recycling work.
    - If the loop reaches `max_span_iters` without converging, a
      warning is issued and the widest grid found is returned;
      `get_fully_converged_energy_levels` then performs a final
      validation pass as a safety net.
    - Three new optional parameters: `span_tol`, `span_growth_factor`,
      `max_span_iters` — all have sensible defaults and do not
      require changes to any calling code.
=====================================================================
"""

import numpy as np
import scipy.linalg as la
import scipy.optimize as opt
import warnings
import time



# =====================================================================
# Automatic grid configuration (SMOOTH potentials only)
# =====================================================================
def auto_configure_dvr(potential_func, num_levels, mass=1.0, hbar=1.0, x0_guess=1.0,
                        span_tol=1e-5, span_growth_factor=1.3, max_span_iters=10):
    """
    Automatically pick a converged grid window [x_min, x_max] and
    grid point count for a SMOOTH potential well.

    METHOD -- two stages:

    Stage 1 (initial estimate):
        Find the potential minimum, estimate a ceiling energy that
        covers `num_levels` states, locate the classical turning
        points there via root-finding, and add a small padding margin.
        This gives a first-guess (x_min, x_max) and fixes the grid
        spacing dx from the de Broglie / Nyquist criterion at the
        ceiling energy.

    Stage 2 (adaptive span expansion):
        Run a DVR solve on the initial grid, then repeat on a grid
        that is `span_growth_factor` times wider (same dx, proportionally
        more points). If max|ΔE_n| < span_tol the current span is
        proven adequate -- the wavefunction tails have decayed to
        negligible amplitude before the boundary -- and the function
        returns. Otherwise the wider grid becomes the new candidate
        and the loop repeats.

    This two-stage approach correctly handles anharmonic potentials
    (double well, Morse, quartic) whose wavefunction tails extend
    much further than the HO-calibrated fixed-padding heuristic
    would predict, without requiring manual tuning.

    NOTE: dx is fixed in Stage 1 and held constant throughout Stage 2.
    Widening the span adds grid points but does NOT change the spacing.
    This decouples span convergence from resolution convergence; the
    latter is checked separately by `get_fully_converged_energy_levels`.

    Parameters
    ----------
    potential_func : callable
        V(x) -> float or ndarray. Finite everywhere (no hard walls).
    num_levels : int
        Number of energy levels to extract. Determines how high the
        energy ceiling is set and therefore how wide/fine the grid is.
    mass : float, optional
        Particle mass (default 1.0).
    hbar : float, optional
        Reduced Planck constant (default 1.0).
    x0_guess : float, optional
        Starting guess for the potential-minimum search. Must be set
        near a well minimum for multi-well potentials (e.g. 1.0 for
        the symmetric double well x^4/4 - x^2/2).
    span_tol : float, optional
        Maximum allowed max|ΔE_n| between the current-width and
        wider-width DVR solves for the span to be declared converged
        (default 1e-5, matching the 3-pass gate in
        get_fully_converged_energy_levels).
    span_growth_factor : float, optional
        Factor by which the half-span is multiplied at each expansion
        step (default 1.3 = 30% wider per iteration).
    max_span_iters : int, optional
        Maximum number of expansion steps before giving up and issuing
        a warning (default 10). Ten steps at 1.3× gives a maximum
        span of 1.3^10 ≈ 13.8× the initial value, which is more than
        sufficient for any physically reasonable smooth potential.

    Returns
    -------
    x_min, x_max : float
        Span-converged grid boundaries.
    grid_points : int
        Number of grid points between x_min and x_max at dx_target.
    """
    print(f"  [Auto-Scanner] Analyzing smooth potential for {num_levels} states...")

    # ------------------------------------------------------------------
    # Stage 1: initial turning-point estimate and resolution target
    # ------------------------------------------------------------------
    # Find the potential-well bottom
    res = opt.minimize(potential_func, x0=x0_guess)
    x_bottom = res.x[0]
    v_min = res.fun

    # Energy ceiling calibrated for HO-like level spacing (E_n ~ n).
    # The adaptive loop in Stage 2 corrects the span if this
    # underestimates how far the wavefunctions actually extend.
    E_ceiling = v_min + (1.5 * num_levels)
    root_func = lambda x: potential_func(x) - E_ceiling

    try:
        x_right = opt.fsolve(root_func, x0=x_bottom + 5.0)[0]
        x_left  = opt.fsolve(root_func, x0=x_bottom - 5.0)[0]
    except Exception:
        raise RuntimeError("fsolve failed to find classical turning points.")

    # Initial padded span
    span0  = abs(x_right - x_left)
    x_min  = x_left  - (0.15 * span0) - 2.0
    x_max  = x_right + (0.15 * span0) + 2.0

    # Fixed centre used throughout Stage 2 (symmetric expansion)
    x_center = (x_min + x_max) / 2.0

    # Grid resolution from de Broglie / Nyquist criterion -- FIXED.
    # dx_target is not changed during span expansion; only the number
    # of points grows as the span widens.
    k_max     = np.sqrt(2.0 * mass * (E_ceiling - v_min)) / hbar
    dx_target = np.pi / (2.0 * k_max)

    def _n_pts(xlo, xhi):
        """Points to cover [xlo, xhi] at dx_target, with a floor of 4×num_levels."""
        n = int(np.ceil((xhi - xlo) / dx_target)) + 1
        return max(n, int(4.0 * num_levels))

    # ------------------------------------------------------------------
    # Stage 2: iterative span expansion
    # ------------------------------------------------------------------
    n_curr = _n_pts(x_min, x_max)
    E_curr = colbert_miller_dvr_1d(
        potential_func, num_levels, x_min, x_max, n_curr, mass, hbar
    )
    print(f"  [Auto-Scanner] Initial: [{x_min:.3f}, {x_max:.3f}]  "
          f"span={x_max - x_min:.3g}  pts={n_curr}  dx={dx_target:.4g}")

    converged = False
    span_error = np.inf

    for i in range(max_span_iters):
        # Widen span by span_growth_factor (symmetric about x_center)
        half_span_wide = ((x_max - x_min) / 2.0) * span_growth_factor
        x_min_wide     = x_center - half_span_wide
        x_max_wide     = x_center + half_span_wide
        n_wide         = _n_pts(x_min_wide, x_max_wide)

        E_wide     = colbert_miller_dvr_1d(
            potential_func, num_levels, x_min_wide, x_max_wide, n_wide, mass, hbar
        )
        span_error = np.max(np.abs(E_curr - E_wide))

        print(f"  [Auto-Scanner] Span iter {i + 1}/{max_span_iters}: "
              f"span {x_max - x_min:.3g} \u2192 {x_max_wide - x_min_wide:.3g}  "
              f"|ΔE|_max = {span_error:.2e}", end="")

        if span_error < span_tol:
            # Current grid (x_min, x_max) is proven adequate.
            # Return it rather than the wider test grid.
            converged = True
            print("  \u2713 converged")
            break

        # Adopt the wider grid as the new base and continue.
        print("  expanding...")
        x_min, x_max = x_min_wide, x_max_wide
        n_curr = n_wide
        E_curr = E_wide

    if not converged:
        warnings.warn(
            f"[Auto-Scanner] Span did not fully converge within {max_span_iters} "
            f"iterations (last |ΔE| = {span_error:.2e}, tolerance = {span_tol:.1e}). "
            f"Proceeding with the widest grid found. "
            f"get_fully_converged_energy_levels will validate the result -- "
            f"if it raises a DVR CONVERGENCE ERROR, increase max_span_iters "
            f"or decrease span_tol in the auto_configure_dvr call.",
            UserWarning,
        )

    grid_points = _n_pts(x_min, x_max)
    print(f"  [Auto-Scanner] Final:   [{x_min:.3f}, {x_max:.3f}] | "
          f"Grid Points: {grid_points}")
    return x_min, x_max, grid_points


# =====================================================================
# Core sinc-DVR solver (Colbert-Miller method)
# =====================================================================
def colbert_miller_dvr_1d(potential_func, num_levels, x_min, x_max, num_points, mass=1.0, hbar=1.0):
    """
    Diagonalize the 1-D Hamiltonian H = T + V(x) on an evenly spaced
    grid using the Colbert-Miller sinc-DVR kinetic energy matrix, and
    return the lowest `num_levels` eigenvalues.

    This solver assumes a SMOOTH potential: V(x) must be finite at
    every grid point. If you need a hard-wall / infinite-square-well
    potential, this version will reject it (see Raises below) -- use
    a dedicated hard-wall DVR formulation instead.

    Parameters
    ----------
    potential_func : callable
        V(x) -> float or ndarray, evaluated at all grid points at once.
        Must be finite everywhere on [x_min, x_max].
    num_levels : int
        Number of lowest eigenvalues to return.
    x_min, x_max : float
        Grid boundaries.
    num_points : int
        Number of grid points (must exceed num_levels).
    mass : float, optional
        Particle mass (default 1.0).
    hbar : float, optional
        Reduced Planck constant (default 1.0).

    Returns
    -------
    eigenvalues : ndarray, shape (num_levels,)
        The lowest `num_levels` energy eigenvalues, ascending order.

    Raises
    ------
    ValueError
        If num_points <= num_levels, or if the potential evaluates to
        +/-inf anywhere on the grid (i.e. a hard-wall potential was
        passed in, which this smooth-only solver does not support).
    """
    if num_points <= num_levels:
        raise ValueError(f"Number of grid points ({num_points}) must be greater than requested levels ({num_levels}).")

    x = np.linspace(x_min, x_max, num_points)
    dx = x[1] - x[0]
    k_factor = (hbar**2) / (2.0 * mass * dx**2)

    # --- Build the Colbert-Miller kinetic energy matrix ---
    # The kinetic energy matrix is Toeplitz (depends only on |i-j|), so we
    # only need to construct the first row and let scipy expand the full
    # symmetric matrix from it -- much cheaper than an explicit double loop.
    idx = np.arange(num_points, dtype=float)

    # Alternating +1/-1 sign pattern, generated directly (cheaper than (-1)**idx).
    alter_sign = np.ones(num_points)
    alter_sign[1::2] = -1.0

    with np.errstate(divide='ignore', invalid='ignore'):
        first_row = k_factor * 2.0 * alter_sign / (idx**2)

    # The i=j (diagonal) term has a known analytic limit, pi^2/3, which
    # avoids the 0/0 indeterminate form from the general formula above.
    first_row[0] = k_factor * (np.pi**2) / 3.0

    # Expand the Toeplitz structure into the full symmetric kinetic matrix.
    H = la.toeplitz(first_row)

    # --- Add the potential energy along the diagonal ---
    v_diag = potential_func(x)
    if np.any(np.isinf(v_diag)):
        # This is the smooth-potential contract: a hard wall (V -> inf)
        # means the caller needs a different (hard-wall-aware) solver.
        raise ValueError(
            "Potential contains infinite values on the grid. This solver only "
            "supports smooth (everywhere-finite) potentials -- hard-wall / "
            "box-type potentials are not supported in this version. Shift "
            "boundaries inward or use a hard-wall-specific DVR formulation."
        )

    H[np.diag_indices(num_points)] += v_diag

    # --- Diagonalize, keeping only the lowest `num_levels` eigenvalues ---
    # subset_by_index selects the requested band directly via LAPACK's MRRR
    # algorithm, which is much faster than computing the full spectrum.
    eigenvalues = la.eigvalsh(H, overwrite_a=True, subset_by_index=[0, num_levels - 1])

    return eigenvalues


# =====================================================================
# Convergence-checked energy levels (resolution + boundary span checks)
# =====================================================================
def get_fully_converged_energy_levels(potential_func, num_levels, x_min, x_max, num_points,
                                       mass=1.0, hbar=1.0, tolerance=1e-5):
    """
    Compute energy levels with the Colbert-Miller DVR and verify that
    the result is numerically converged with respect to both grid
    resolution and grid boundary placement before returning it.

    Three passes are run:
      1. Base grid, as specified.
      2. A 30% finer grid over the same span (checks resolution error).
      3. A 20%-more-points grid over a 10%-wider span (checks
         boundary/truncation error).
    If either check exceeds `tolerance`, a RuntimeError is raised
    rather than silently returning an under-converged result.

    Parameters
    ----------
    potential_func : callable
        V(x) -> float or ndarray. Must be finite everywhere (smooth potential).
    num_levels : int
        Number of lowest eigenvalues to compute and verify.
    x_min, x_max : float
        Grid boundaries for the base pass.
    num_points : int
        Grid point count for the base pass.
    mass, hbar : float, optional
        Physical constants (default 1.0 each).
    tolerance : float, optional
        Maximum allowed max-abs energy difference between the base
        pass and each verification pass (default 1e-5).

    Returns
    -------
    E_base : ndarray, shape (num_levels,)
        The converged energy eigenvalues from the base-grid pass.

    Raises
    ------
    RuntimeError
        If either the resolution or boundary-span error exceeds tolerance.
    """
    # Pass 1: base grid as specified by caller.
    print(f"  [DVR] Pass 1/3: Base grid ({num_points} pts) ...", end=" ", flush=True)
    t0 = time.time()
    E_base = colbert_miller_dvr_1d(potential_func, num_levels, x_min, x_max, num_points, mass, hbar)
    print(f"done ({time.time()-t0:.1f}s)")

    # Pass 2: 30% more points over the same span — checks resolution (dx) error.
    num_points_finer = int(num_points * 1.3)
    print(f"  [DVR] Pass 2/3: Resolution check ({num_points_finer} pts) ...", end=" ", flush=True)
    t0 = time.time()
    E_res = colbert_miller_dvr_1d(potential_func, num_levels, x_min, x_max, num_points_finer, mass, hbar)
    res_error = np.max(np.abs(E_base - E_res))
    print(f"done ({time.time()-t0:.1f}s)  max|ΔE| = {res_error:.2e}")

    # Pass 3: 20% more points over a 10%-wider span — checks boundary truncation error.
    span = x_max - x_min
    x_min_wider = x_min - 0.1 * span
    x_max_wider = x_max + 0.1 * span
    num_points_wider = int(num_points * 1.2)
    print(f"  [DVR] Pass 3/3: Boundary span check ({num_points_wider} pts, ±10% wider) ...", end=" ", flush=True)
    t0 = time.time()
    E_span = colbert_miller_dvr_1d(potential_func, num_levels, x_min_wider, x_max_wider, num_points_wider, mass, hbar)
    span_error = np.max(np.abs(E_base - E_span))
    print(f"done ({time.time()-t0:.1f}s)  max|ΔE| = {span_error:.2e}")

    if res_error > tolerance or span_error > tolerance:
        raise RuntimeError(f"DVR CONVERGENCE ERROR\nResolution Error: {res_error:.2e}\nSpan Error: {span_error:.2e}")

    return E_base


# =====================================================================
# Quantum heat capacity from a raw energy spectrum (convenience helper)
# =====================================================================
def compute_quantum_heat_capacity(E, beta, k_B=1.0):
    """
    Compute Cv at a single inverse temperature beta directly from a
    set of energy eigenvalues, via the canonical-ensemble variance
    formula Cv = k_B * beta^2 * Var(E).

    Parameters
    ----------
    E : ndarray
        Energy eigenvalues (ascending).
    beta : float
        Inverse temperature, 1 / (k_B * T) in the same energy units as E.
    k_B : float, optional
        Boltzmann constant (default 1.0, dimensionless units).

    Returns
    -------
    Cv : float
        Heat capacity at the given beta.

    Warns
    -----
    UserWarning
        If the highest included energy state still carries non-negligible
        Boltzmann weight, meaning the truncated spectrum may be too short
        for this temperature (the true Cv would need more levels).
    """
    E_shifted = E - E[0]
    weights = np.exp(-beta * E_shifted)
    Z = np.sum(weights)

    highest_state_occupancy = weights[-1] / Z
    if highest_state_occupancy > 1e-4:
        warnings.warn("Thermodynamic Truncation Threat: Heat capacity will collapse toward zero!", UserWarning)

    mean_E = np.sum(E * weights) / Z
    mean_E_sq = np.sum((E**2) * weights) / Z
    variance_E = mean_E_sq - (mean_E**2)

    return k_B * (beta**2) * variance_E