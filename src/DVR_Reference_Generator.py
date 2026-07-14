"""
DVR_Reference_Generator_1_0.py
=====================================================================
WHAT THIS FILE DOES
---------------------------------------------------------------------
Generates a HIGH-PRECISION NUMERICAL REFERENCE energy spectrum by
running the DVR solver on a finer and/or wider grid than the "base"
grid used in the main pipeline. This reference spectrum plays the
role that the analytical solution plays in HO_Benchmark -- except it
works for ANY smooth potential, including ones with no closed-form
eigenvalues.

The standard scaling is:
    span  : L  →  span_factor × L    (default: 2× wider)
    dx    : Δx →  Δx / dx_factor     (default: half the spacing)

Both factors can be set freely by the user, either as arguments or
interactively at runtime (set interactive=True). When unsure, the
two-level nesting of L→2L and dx→dx/2 gives a reference grid with
4× more points per unit length over twice the domain -- more than
sufficient to expose any span-truncation or resolution error that
the base grid might carry.

Rationale:
  A DVR eigenvalue E_n converges from above as the grid is refined
  (more points) and widened (larger span).  Comparing base vs
  reference therefore gives a conservative upper bound on the true
  error: if base and reference agree to tolerance ε, the base error
  relative to the exact answer is at most ~ε (often much smaller,
  because the reference overcorrects).

  This is the same idea as Richardson extrapolation, but used here
  purely as a CONVERGENCE CHECK rather than to improve the base
  result itself.

CHANGELOG (NEW FILE, v1.0)
---------------------------------------------------------------------
- New file. Replaces the role of HO_Analytical_1_0.py as the
  ground-truth supplier for systems with no analytic solution.
  DVR_Limit_Finder_1_2 already accepted any reference spectrum; this
  file provides a principled way to generate one without needing
  analytical formulas.
=====================================================================
"""

import time
import numpy as np
from DVR_Algorithm import colbert_miller_dvr_1d


# =====================================================================
# Standard (default) scaling factors
# =====================================================================
DEFAULT_SPAN_FACTOR = 2.0   # multiply span by this (L → 2L)
DEFAULT_DX_FACTOR   = 2.0   # divide dx  by this  (Δx → Δx/2)


# =====================================================================
# Interactive factor selection
# =====================================================================
def ask_scaling_factors(default_span_factor=DEFAULT_SPAN_FACTOR,
                         default_dx_factor=DEFAULT_DX_FACTOR):
    """
    Interactively prompt the user for span and dx scaling factors.

    If the user enters nothing at a prompt, the shown default is used.
    If the user enters a non-numeric value, defaults are used and a
    warning is printed.

    Parameters
    ----------
    default_span_factor : float
        Default value for span scaling (shown in prompt).
    default_dx_factor : float
        Default value for dx division factor (shown in prompt).

    Returns
    -------
    span_factor, dx_factor : float, float
    """
    print("\n  [Reference Generator] Choose reference grid scaling:")
    print(f"  The reference grid will use:")
    print(f"    span  →  span_factor × base_span  (2.0 = double the domain)")
    print(f"    Δx    →  base_Δx / dx_factor       (2.0 = twice the point density)")

    try:
        raw_span = input(f"  span_factor  [default {default_span_factor}]: ").strip()
        span_factor = float(raw_span) if raw_span else default_span_factor

        raw_dx = input(f"  dx_factor    [default {default_dx_factor}]: ").strip()
        dx_factor = float(raw_dx) if raw_dx else default_dx_factor

    except (ValueError, EOFError):
        print("  Invalid input -- using defaults.")
        span_factor, dx_factor = default_span_factor, default_dx_factor

    # Sanity warnings (do not block -- the user may intentionally coarsen)
    if span_factor <= 1.0:
        print(f"  \u26a0  span_factor={span_factor:.3g} \u2264 1.0 produces a NARROWER reference span.")
    if dx_factor <= 1.0:
        print(f"  \u26a0  dx_factor={dx_factor:.3g} \u2264 1.0 produces a COARSER reference grid.")

    return span_factor, dx_factor


# =====================================================================
# Reference grid parameter calculation
# =====================================================================
def compute_reference_grid_params(x_min, x_max, num_points, span_factor, dx_factor):
    """
    Compute the reference grid boundaries and point count from the
    base grid parameters and the two scaling factors.

    The reference span is centred on the same mid-point as the base
    span, so both grids are symmetric about the same axis.

        new_span  = base_span × span_factor
        x_min_ref = x_centre - new_span / 2
        x_max_ref = x_centre + new_span / 2
        dx_ref    = dx_base / dx_factor
        n_pts_ref = round(new_span / dx_ref) + 1

    Parameters
    ----------
    x_min, x_max : float
        Base grid boundaries.
    num_points : int
        Number of points on the BASE grid.
    span_factor : float
        Multiplier for the spatial span (>1 → wider reference).
    dx_factor : float
        Divisor for the grid spacing (>1 → finer reference).

    Returns
    -------
    dict with keys:
        x_min_ref, x_max_ref : float
            Reference grid boundaries.
        num_points_ref : int
            Number of grid points on the reference grid.
        dx_base, dx_ref : float
            Grid spacing on base / reference grids.
        span_base, span_ref : float
            Spatial spans.
    """
    span_base  = x_max - x_min
    dx_base    = span_base / (num_points - 1)
    x_centre   = (x_min + x_max) / 2.0

    span_ref   = span_base * span_factor
    x_min_ref  = x_centre - span_ref / 2.0
    x_max_ref  = x_centre + span_ref / 2.0

    dx_ref         = dx_base / dx_factor
    num_points_ref = int(round(span_ref / dx_ref)) + 1

    return {
        "x_min_ref":      x_min_ref,
        "x_max_ref":      x_max_ref,
        "num_points_ref": num_points_ref,
        "dx_base":        dx_base,
        "dx_ref":         dx_ref,
        "span_base":      span_base,
        "span_ref":       span_ref,
    }


# =====================================================================
# Main entry point
# =====================================================================
def generate_reference_energies(potential_func, num_levels, x_min, x_max, num_points,
                                  span_factor=DEFAULT_SPAN_FACTOR,
                                  dx_factor=DEFAULT_DX_FACTOR,
                                  mass=1.0, hbar=1.0,
                                  interactive=False, verbose=True):
    """
    Compute a high-precision numerical reference energy spectrum by
    running the smooth-potential DVR solver on a finer / wider grid
    than the base grid used in the main pipeline.

    This reference plays the role of the analytical ground truth for
    systems where no closed-form eigenvalues exist. Both the energy-
    level accuracy check (DVR_Limit_Finder, HO_Energy_Level_Error)
    and the Cv benchmark (Cv_Numerical_Benchmark) accept this output
    directly in place of analytic energies.

    Parameters
    ----------
    potential_func : callable
        V(x), smooth (finite everywhere on the reference grid, which
        is wider than the base grid).
    num_levels : int
        Number of reference energy levels to compute. Should be at
        least as large as the number of base levels you intend to
        compare, and ideally larger so the reference partition
        function doesn't truncate before the base one does.
    x_min, x_max : float
        BASE grid boundaries. The reference boundaries are derived
        from these via `span_factor`.
    num_points : int
        BASE grid point count. The reference count is derived via
        both `span_factor` and `dx_factor`.
    span_factor : float, optional
        Multiply the span by this to get the reference span
        (default 2.0 = double the domain, reducing boundary-truncation
        error by moving the walls further from the wavefunction).
    dx_factor : float, optional
        Divide dx by this to get the reference spacing (default 2.0 =
        half the spacing, reducing grid-resolution error).
    mass, hbar : float, optional
        Physical constants (default 1.0, dimensionless units).
    interactive : bool, optional
        If True, prompt the user to enter span_factor and dx_factor
        at runtime. The function-argument values serve as defaults
        in the prompt. Default False (use function-argument values
        silently).
    verbose : bool, optional
        Print a base-vs-reference grid comparison table (default True).

    Returns
    -------
    dict with keys:
        energies : ndarray, shape (num_levels,)
            Reference energy levels, computed on the finer/wider grid.
        grid : dict
            Reference grid parameters from `compute_reference_grid_params`.
        span_factor, dx_factor : float
            The scaling factors actually used (after any interactive
            override), echoed back for logging/plotting labels.
    """
    if interactive:
        span_factor, dx_factor = ask_scaling_factors(span_factor, dx_factor)

    grid = compute_reference_grid_params(x_min, x_max, num_points, span_factor, dx_factor)

    if verbose:
        ratio = grid["num_points_ref"] / num_points
        print(f"\n  [Reference Generator] Numerical reference grid:")
        print(f"  {'':>10}  {'span':>12}  {'dx':>12}  {'num_pts':>9}")
        print(f"  {'Base':>10}  {grid['span_base']:>12.5g}  {grid['dx_base']:>12.5g}  {num_points:>9d}")
        print(f"  {'Reference':>10}  {grid['span_ref']:>12.5g}  {grid['dx_ref']:>12.5g}  {grid['num_points_ref']:>9d}")
        print(f"  Scaling: span \u00d7{span_factor:.2g},  dx \u00f7{dx_factor:.2g}  ({ratio:.1f}\u00d7 more pts total)")
        print(f"  Solving {num_levels} levels on reference grid ...", end=" ", flush=True)

    t0 = time.time()
    energies = colbert_miller_dvr_1d(
        potential_func, num_levels,
        grid["x_min_ref"], grid["x_max_ref"], grid["num_points_ref"],
        mass, hbar,
    )

    if verbose:
        print(f"done ({time.time() - t0:.1f}s)")

    return {
        "energies":    energies,
        "grid":        grid,
        "span_factor": span_factor,
        "dx_factor":   dx_factor,
    }
