"""
DVR_Algorithm_1_4.py
=====================================================================
WHAT THIS FILE DOES
---------------------------------------------------------------------
Core 1-D Discrete Variable Representation (DVR) engine. Given any
SMOOTH potential V(x) (finite everywhere -- no hard walls / infinite
jumps), it builds the Colbert-Miller sinc-DVR Hamiltonian on a grid,
diagonalizes it, and returns the lowest `num_levels` energy
eigenvalues. Also contains the automatic grid-configuration helper
(turning-point search + grid-density estimate).

This file is system-agnostic: pass in any callable V(x) (Harmonic
Oscillator, double well, etc.) and it will compute energy levels for
that system. Nothing here is specific to the Harmonic Oscillator.

CHANGELOG (v1.3 -> v1.4)
---------------------------------------------------------------------
- REMOVED DisappearingTimer entirely, along with its `sys` and
  `multiprocessing` imports. The multiprocessing-based ticker was
  spawning a child process on every call to
  `get_fully_converged_energy_levels`, adding non-trivial OS
  overhead (process fork + IPC queue + shared Event) and competing
  with LAPACK for CPU time. The DVR file is also called in tight
  loops by DVR_Limit_Finder, where the process-spawn overhead was
  paid on every single grid-density or level-count test step.
- `get_fully_converged_energy_levels` now prints a one-line
  timestamp for each of its three passes using a simple
  `print(..., end=" ", flush=True)` + elapsed-time suffix. This is
  zero-overhead (no threads or processes), survives LAPACK's GIL
  hold without issue (the print happens before and after the heavy
  call, not concurrently with it), and gives the user the same
  "is it still running?" visibility.
- Progress timing for longer pipeline stages is now the
  responsibility of Quantum_HO_Master_1_3.py's `SimpleTimer` (a
  lightweight daemon thread that prints every 10 s), which wraps
  whole pipeline sections rather than individual LAPACK calls.
- No change to the underlying sinc-DVR math.
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
def auto_configure_dvr(potential_func, num_levels, mass=1.0, hbar=1.0, x0_guess=1.0):
    """
    Automatically pick a grid window [x_min, x_max] and a grid point
    count for a SMOOTH potential well, given how many energy levels
    are wanted.

    Method: locate the potential minimum, climb up to an energy
    ceiling roughly proportional to `num_levels` (so the grid covers
    every state we asked for plus headroom), find the classical
    turning points there via root-finding, pad them outward a bit,
    and pick a grid spacing fine enough to resolve the shortest
    wavelength expected at that ceiling energy (via the local
    de Broglie wavelength / Nyquist-style estimate).

    NOTE: This function assumes V(x) is finite and smooth everywhere
    on the search domain (no hard walls). For potentials with hard
    walls / discontinuities, this auto-configuration is not
    applicable -- a hard-wall-aware grid builder would be needed
    instead, which is intentionally out of scope for this module.

    Parameters
    ----------
    potential_func : callable
        V(x) -> float or ndarray. Must be finite everywhere it is
        evaluated (no np.inf).
    num_levels : int
        Number of energy levels the caller intends to extract. Used
        to set how far up in energy (and therefore how wide/fine)
        the grid needs to be.
    mass : float, optional
        Particle mass (default 1.0, matching hbar=1 dimensionless units).
    hbar : float, optional
        Reduced Planck constant (default 1.0, dimensionless units).
    x0_guess : float, optional
        Starting guess for the potential-minimum search. Useful for
        steering the minimizer into a specific well of a multi-well
        potential (kept here for forward-compatibility with future
        non-HO potentials, even though only HO is exercised today).

    Returns
    -------
    x_min, x_max : float
        Recommended grid boundaries.
    grid_points : int
        Recommended number of grid points between x_min and x_max.
    """
    print(f"  [Auto-Scanner] Analyzing smooth potential for {num_levels} states...")

    # Step 1: find the bottom of the well.
    res = opt.minimize(potential_func, x0=x0_guess)
    x_bottom = res.x[0]
    v_min = res.fun

    # Step 2: pick an energy ceiling that comfortably covers num_levels states.
    E_ceiling = v_min + (10.0 * num_levels)         # OG HO -> 2.0 * num_levels
    root_func = lambda x: potential_func(x) - E_ceiling

    try:
        # Search outward from the well bottom in both directions for the
        # classical turning points at the ceiling energy.
        x_right = opt.fsolve(root_func, x0=x_bottom + 5.0)[0]
        x_left = opt.fsolve(root_func, x0=x_bottom - 5.0)[0]
    except Exception:
        raise RuntimeError("fsolve failed to find classical turning points.")

    # Step 3: pad the window outward so the wavefunction has room to decay.
    span = abs(x_right - x_left)
    x_min = x_left - (0.15 * span) - 2.0
    x_max = x_right + (0.15 * span) + 2.0

    # Step 4: estimate the grid spacing needed to resolve the shortest
    # wavelength present at the energy ceiling (de Broglie / Nyquist style).
    k_max = np.sqrt(2.0 * mass * (E_ceiling - v_min)) / hbar
    dx_target = np.pi / (2.0 * k_max)

    grid_points = int(np.ceil((x_max - x_min) / dx_target))
    grid_points = max(grid_points, int(4.0 * num_levels))

    print(f"  [Auto-Scanner] Bounds: [{x_min:.3f}, {x_max:.3f}] | Grid Points: {grid_points}")
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
