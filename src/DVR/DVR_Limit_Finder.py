"""
DVR_Limit_Finder_1_4.py
=====================================================================
WHAT THIS FILE DOES
---------------------------------------------------------------------
General-purpose (system-agnostic) tool for mapping out WHERE the
smooth-potential DVR solver stops being trustworthy, by comparing
its output against a reference spectrum and searching for the
breakdown point in two complementary directions:

    (A) RESOLUTION LIMIT -- fix the number of energy levels n and the
        spatial span, then coarsen the grid (increase dx) until the
        error vs reference exceeds a tolerance. Finds the maximum
        grid spacing dx still trustworthy for n levels.

    (B) LEVEL-COUNT LIMIT -- fix a grid (span + dx, i.e. fixed
        computational cost) then grow the number of levels requested
        from that SAME grid until the error vs reference exceeds
        tolerance. Finds the highest state index that grid can still
        be trusted for, and now continues searching beyond the initial
        reference length when an extension function is provided.

Both searches work with ANY potential and ANY reference spectrum.

CHANGELOG (v1.3 -> v1.4)
---------------------------------------------------------------------
- `find_maximum_levels` gains an optional `extend_reference_func`
  parameter. Previously, when the search reached the end of the
  supplied reference spectrum without finding a breakdown spike, it
  stopped and reported "capped by reference_length" -- leaving the
  plot with a flat floor and no visible cliff.

  With `extend_reference_func` provided, the search instead calls
  that function to compute additional reference levels on demand,
  then continues growing n all the way up to the grid's hard ceiling
  (num_points - 2). This guarantees the plot always shows the full
  error curve including the breakdown cliff, regardless of how many
  reference levels were pre-computed upstream.

  The callable signature is:
      extend_reference_func(n_needed: int) -> array_like
  where n_needed is the total number of reference levels required
  (not a count of extras). The function may compute them fresh each
  call; caching is the caller's responsibility.

  A mutable container (_ref) is used internally so the `evaluate`
  closure always references the latest (possibly extended) reference
  without needing nonlocal.

  If `extend_reference_func` is None (default), behaviour is
  identical to v1.3.

- `plot_level_limit_search` updated to annotate "Reference extended"
  in the plot title when the search used the extension function.
=====================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

from DVR.DVR_Algorithm import colbert_miller_dvr_1d
from error.error_energylevels import compute_energy_level_errors


# =====================================================================
# Shared pass/fail helper used by both searches
# =====================================================================
_METRIC_KEYS = {
    "max_abs": "max_abs_error", "max_rel": "max_rel_error",
    "mean_abs": "mean_abs_error", "mean_rel": "mean_rel_error",
}


def _passes_tolerance(E_test, reference_energies, tolerance, metric):
    """
    Compare a freshly computed DVR spectrum `E_test` against the first
    len(E_test) entries of `reference_energies` and report whether the
    chosen error metric is within `tolerance`.

    Parameters
    ----------
    E_test : ndarray
        DVR-computed energy levels for the configuration being tested.
    reference_energies : array_like
        Ground-truth spectrum (analytic or finer numerical). Must
        have at least len(E_test) entries.
    tolerance : float
        The threshold the chosen metric must not exceed to "pass".
    metric : str
        One of "max_abs", "max_rel", "mean_abs", "mean_rel".
        "max_*" requires EVERY compared level to individually be
        within tolerance (strict, the recommended default); "mean_*"
        only requires the per-spectrum average error to be within
        tolerance (looser -- a single bad level can hide in the
        average).

    Returns
    -------
    passed : bool
    error_value : float
        The scalar value of the chosen metric.
    error_result : dict
        Full output of `compute_energy_level_errors`, for callers
        that need more detail than the single scalar.
    """
    if metric not in _METRIC_KEYS:
        raise ValueError(f"Unknown metric '{metric}'. Choose from {list(_METRIC_KEYS)}.")
    n = len(E_test)
    if len(reference_energies) < n:
        raise ValueError(
            f"reference_energies has only {len(reference_energies)} entries but "
            f"{n} are needed -- supply a longer reference spectrum."
        )
    error_result = compute_energy_level_errors(E_test, np.asarray(reference_energies)[:n])
    error_value = error_result[_METRIC_KEYS[metric]]
    return (error_value <= tolerance), error_value, error_result


def _dx_to_pts(dx, x_min, x_max, hard_floor):
    """
    Convert a desired grid spacing dx into the integer number of grid
    points needed by `colbert_miller_dvr_1d` for a span [x_min, x_max].

    np.linspace(x_min, x_max, num_points) produces a spacing of
    (x_max - x_min) / (num_points - 1), so we invert that:
        num_points = round(span / dx) + 1
    and clip to hard_floor so the DVR solver's num_points > num_levels
    contract is never violated.

    Parameters
    ----------
    dx : float
        Desired grid spacing.
    x_min, x_max : float
        Span boundaries.
    hard_floor : int
        Minimum acceptable num_points (typically num_levels + 2).

    Returns
    -------
    num_points : int
    """
    span = x_max - x_min
    num_points = int(round(span / dx)) + 1
    return max(hard_floor, num_points)


# =====================================================================
# (A) Resolution limit: grow dx at fixed n and fixed span
# =====================================================================
def find_minimum_grid_points(potential_func, num_levels, x_min, x_max, reference_energies,
                              tolerance=1e-6, metric="max_rel", mass=1.0, hbar=1.0,
                              dx_start=None, grow_factor=1.15,
                              max_anchor_pts=200_000, verbose=True):
    """
    Find the RESOLUTION LIMIT of the DVR solver: the maximum grid
    spacing dx (equivalently, the minimum number of grid points) at a
    FIXED spatial span [x_min, x_max] for which all of the lowest
    `num_levels` eigenvalues stay within `tolerance` of the reference.

    WHY dx and not num_points?
    dx has a direct physical meaning: it is the length scale that sets
    the kinetic-energy cutoff of the DVR basis (~pi*hbar/(2*m*dx)^2)
    and determines how well the shortest de Broglie wavelength present
    at the highest requested energy level is sampled. Expressing the
    limit as a maximum dx therefore generalises immediately to new
    spans, new potentials, and new systems without reinterpreting a
    raw point count.

    Method:
        1. Find a fine anchor dx_start that verifiably passes (grow
           it downward automatically if the default does not pass).
        2. Geometrically grow dx by `grow_factor` until the error
           first exceeds `tolerance`.
        3. Binary-search in integer num_points space between the
           last-passing and first-failing point counts to pin the
           boundary to single-point precision, then convert back to dx.

    Parameters
    ----------
    potential_func : callable
        V(x), smooth (finite everywhere), passed through to the solver.
    num_levels : int
        Fixed number of energy levels tested at every grid density.
    x_min, x_max : float
        Fixed spatial span (only grid SPACING is varied here; a
        separate search would be needed to probe span/truncation
        effects independently).
    reference_energies : array_like
        Ground-truth energies, length >= num_levels.  Analytic or a
        finer numerical reference -- the function does not care which.
    tolerance : float, optional
        Maximum allowed error (in units of `metric`) to "pass"
        (default 1e-6).
    metric : str, optional
        One of "max_abs", "max_rel", "mean_abs", "mean_rel"
        (default "max_abs").
    mass, hbar : float, optional
        Physical constants passed through to the DVR solver.
    dx_start : float or None, optional
        Starting (fine, expected-to-pass) grid spacing. Defaults to
        span / max(8*num_levels, 200), which gives a generous starting
        density. Decrease this if the default anchor fails.
    grow_factor : float, optional
        Geometric growth factor applied to dx each step while
        searching for the coarse-grid breakdown point (default 1.15,
        meaning ~15% coarser per step -- fine enough to map the cliff
        well without excessive DVR calls).
    max_anchor_pts : int, optional
        Safety cap: the search will not go below dx = span/max_anchor_pts
        while looking for a passing anchor (default 200,000 points
        equivalent).
    verbose : bool, optional
        Print a one-line summary at the end (default True).

    Returns
    -------
    dict with keys:
        maximum_dx : float
            Coarsest grid spacing found that still passes tolerance.
            This is the primary, physically meaningful result.
        minimum_grid_points : int
            The corresponding number of grid points (derived from
            maximum_dx via the span).
        error_at_limit : float
            The error metric value at maximum_dx.
        capped_by : str
            "tolerance" if a genuine breakdown was found while
            coarsening, or "hard_floor" if the solver remained
            accurate all the way down to the smallest possible grid
            (num_levels + 2 points) without ever failing.
        trace : list of dict
            Every {"dx", "num_points", "error", "passed"} tested, in
            evaluated order. Feed this to `plot_grid_limit_search`.
        tolerance, metric, num_levels : echoed back for convenience.
    """
    span = x_max - x_min
    hard_floor = num_levels + 2  # colbert_miller_dvr_1d requires num_points > num_levels

    if dx_start is None:
        dx_start = span / max(8 * num_levels, 200)

    trace = []

    def evaluate_pts(num_points):
        """Run the DVR at a given num_points, log to trace, return (passed, err)."""
        dx = span / (num_points - 1)
        E_test = colbert_miller_dvr_1d(
            potential_func, num_levels, x_min, x_max, num_points, mass, hbar
        )
        passed, err, _ = _passes_tolerance(E_test, reference_energies, tolerance, metric)
        trace.append({"dx": dx, "num_points": num_points, "error": err, "passed": passed})
        return passed, err

    # --- Step 1: find a known-good fine anchor ---
    # Start at dx_start and shrink dx (more points) until we get a pass.
    current_dx = dx_start
    current_pts = _dx_to_pts(current_dx, x_min, x_max, hard_floor)
    passed, err = evaluate_pts(current_pts)

    min_dx_anchor = span / max_anchor_pts
    while not passed and current_dx > min_dx_anchor:
        # Halve dx (double point count) each attempt until we pass.
        current_dx /= 2.0
        current_pts = _dx_to_pts(current_dx, x_min, x_max, hard_floor)
        passed, err = evaluate_pts(current_pts)

    if not passed:
        raise RuntimeError(
            f"No converged anchor found even with dx as small as {current_dx:.4g} "
            f"(equiv. {current_pts} points). Check potential_func / reference_energies "
            f"agreement, or relax tolerance={tolerance:.1e}."
        )

    good_pts, good_err = current_pts, err

    # --- Step 2: geometrically coarsen dx until tolerance breaks ---
    # We grow dx by grow_factor each step, which means fewer grid points
    # each step -- we are walking from fine to coarse until it fails.
    bad_pts = None
    capped_by = "tolerance"

    while True:
        # Proposed coarser dx (fewer points).
        candidate_dx = span / (good_pts - 1) * grow_factor
        candidate_pts = _dx_to_pts(candidate_dx, x_min, x_max, hard_floor)

        if candidate_pts >= good_pts:
            # Geometric step didn't move us to fewer points (happens near
            # the hard floor) -- force at least one fewer point.
            candidate_pts = good_pts - 1

        if candidate_pts <= hard_floor:
            # Reached the solver's minimum possible grid without failing --
            # the solver holds up all the way to the hard floor.
            capped_by = "hard_floor"
            break

        passed, err = evaluate_pts(candidate_pts)
        if passed:
            good_pts, good_err = candidate_pts, err
        else:
            bad_pts = candidate_pts
            break

    # --- Step 3: bisect in integer num_points space to pinpoint the boundary ---
    # Note: fewer points = coarser dx = more error, so bad_pts < good_pts.
    # We want the largest num_points (= finest dx) that still FAILS is bad_pts,
    # and the smallest num_points (= coarsest dx) that still PASSES is good_pts.
    # Binary search to tighten the gap to ±1 point.
    if bad_pts is not None and capped_by == "tolerance":
        lo, hi = bad_pts, good_pts   # lo: fails, hi: passes
        while hi - lo > 1:
            mid = (lo + hi) // 2
            passed, err = evaluate_pts(mid)
            if passed:
                hi, good_err = mid, err
            else:
                lo = mid
        good_pts = hi   # smallest num_points that passes = coarsest safe grid

    maximum_dx = span / (good_pts - 1)

    if verbose:
        tag = " (held up to hard floor)" if capped_by == "hard_floor" else ""
        print(
            f"  [Resolution Limit] n={num_levels}: "
            f"maximum dx = {maximum_dx:.5g}  "
            f"(= {good_pts} pts, error={good_err:.3e}){tag}"
        )

    return {
        "maximum_dx": maximum_dx,
        "minimum_grid_points": good_pts,
        "error_at_limit": good_err,
        "capped_by": capped_by,
        "trace": trace,
        "tolerance": tolerance,
        "metric": metric,
        "num_levels": num_levels,
        "span": span,
    }


# =====================================================================
# (B) Level-count limit: grow n at a fixed grid (fixed dx)
# =====================================================================
def find_maximum_levels(potential_func, x_min, x_max, num_points, reference_energies,
                         tolerance=1e-6, metric="max_rel", mass=1.0, hbar=1.0,
                         n_start=None, grow_factor=1.3, verbose=True,
                         extend_reference_func=None):
    """
    Find the LEVEL-COUNT LIMIT of the DVR solver: for a FIXED grid
    (x_min, x_max, num_points -- i.e. a fixed dx), the largest number
    of levels n for which all of the lowest n eigenvalues stay within
    `tolerance` of the reference.

    The search grows n geometrically until the error exceeds tolerance,
    then bisects to pin the exact crossing level. If `extend_reference_func`
    is provided and the search exhausts the supplied reference levels
    without finding a spike, it calls that function to obtain more
    reference levels and continues up to the grid's hard ceiling
    (num_points - 2), ensuring the error plot always shows the full
    curve including the breakdown cliff.

    Parameters
    ----------
    potential_func : callable
        V(x), smooth, passed through to the DVR solver.
    x_min, x_max, num_points : float, float, int
        Fixed grid configuration (held constant throughout).
    reference_energies : array_like
        Initial ground-truth spectrum. Should ideally be at least
        num_points - 2 levels long. If shorter, provide
        `extend_reference_func` so the search can obtain more.
    tolerance : float, optional
        Maximum allowed error to "pass" (default 1e-6).
    metric : str, optional
        One of "max_abs", "max_rel", "mean_abs", "mean_rel"
        (default "max_rel").
    mass, hbar : float, optional
        Physical constants passed through to the DVR solver.
    n_start : int or None, optional
        Starting level count (default max(2, num_points // 20)).
    grow_factor : float, optional
        Geometric growth ratio per step (default 1.3).
    verbose : bool, optional
        Print progress lines (default True).
    extend_reference_func : callable or None, optional
        If provided and the search reaches the reference ceiling
        without finding a spike, this function is called as:
            extend_reference_func(n_needed: int) -> array_like
        and must return at least `n_needed` reference energy levels.
        The search then continues up to the grid's hard ceiling.
        If None (default), behaviour is unchanged from v1.3: the
        search stops at min(num_points-2, len(reference_energies))
        and reports "capped by reference_length".

    Returns
    -------
    dict with keys:
        maximum_levels : int or None
            Largest n within tolerance. None if even n_start fails.
        error_at_maximum : float
        dx_fixed : float
        capped_by : str
            "tolerance", "grid_size", or "reference_length".
        reference_extended : bool
            True if extend_reference_func was called during the search.
        trace : list of dict  -- every {"n", "error", "passed"} tested.
        tolerance, metric, num_points : echoed back for convenience.
    """
    span     = x_max - x_min
    dx_fixed = span / (num_points - 1)
    hard_ceiling = num_points - 2  # DVR requires num_points > num_levels

    # Use a single-element mutable list as a "pointer" to the current
    # reference array. The inner evaluate() closure captures _ref itself
    # (not its contents), so reassigning _ref[0] is visible to all
    # subsequent evaluate() calls without needing nonlocal.
    _ref = [np.asarray(reference_energies, dtype=float)]
    ref_ceiling   = len(_ref[0])
    n_cap         = min(hard_ceiling, ref_ceiling)
    reference_extended = False

    if n_cap < 2:
        raise ValueError("Grid/reference too small to test any levels (n_cap < 2).")

    if n_start is None:
        n_start = max(2, num_points // 20)
    n_start = min(n_start, n_cap)

    trace = []

    def evaluate(n):
        """Compute n DVR levels, compare against _ref[0][:n], log to trace."""
        E_test = colbert_miller_dvr_1d(
            potential_func, n, x_min, x_max, num_points, mass, hbar
        )
        passed, err, _ = _passes_tolerance(E_test, _ref[0], tolerance, metric)
        trace.append({"n": n, "error": err, "passed": passed})
        return passed, err

    passed, err = evaluate(n_start)
    if not passed:
        if verbose:
            print(
                f"  [Level Limit] dx={dx_fixed:.4g}: even n_start={n_start} fails "
                f"(error={err:.3e}) -- try a finer fixed grid."
            )
        return {
            "maximum_levels": None, "error_at_maximum": err, "dx_fixed": dx_fixed,
            "capped_by": "tolerance", "reference_extended": False,
            "trace": trace, "tolerance": tolerance, "metric": metric,
            "num_points": num_points,
        }

    good_n, good_err = n_start, err
    bad_n  = None
    capped_by = "tolerance"

    # Outer loop: allows one extension of the reference, then continues.
    while True:
        # --- Geometrically grow n until tolerance breaks or n_cap is reached ---
        while good_n < n_cap:
            candidate = min(int(np.ceil(good_n * grow_factor)), n_cap)
            if candidate <= good_n:
                candidate = good_n + 1
            passed, err = evaluate(candidate)
            if passed:
                good_n, good_err = candidate, err
            else:
                bad_n = candidate
                break

        if bad_n is not None:
            # Found a failure -- bisect and exit.
            capped_by = "tolerance"
            break

        # Reached n_cap without failure -- determine why and whether to extend.
        if good_n >= hard_ceiling:
            capped_by = "grid_size"
            break

        # Reference ran out before the grid ceiling.
        if extend_reference_func is None:
            capped_by = "reference_length"
            break

        # Try to extend the reference up to the grid's hard ceiling.
        if verbose:
            print(
                f"  [Level Limit] Reference exhausted at n={n_cap}. "
                f"Extending to {hard_ceiling} levels ..."
            )
        new_ref = np.asarray(extend_reference_func(hard_ceiling), dtype=float)
        if len(new_ref) <= ref_ceiling:
            # Extension didn't add anything useful.
            if verbose:
                print("  [Level Limit] Extension returned no additional levels; stopping.")
            capped_by = "reference_length"
            break

        # Update the shared reference pointer seen by evaluate().
        _ref[0]  = new_ref
        ref_ceiling = len(_ref[0])
        n_cap    = min(hard_ceiling, ref_ceiling)
        reference_extended = True
        # Loop back to continue growing n from good_n.

    # --- Bisect between the last good and first bad level count ---
    if bad_n is not None:
        lo, hi = good_n, bad_n  # lo passes, hi fails
        while hi - lo > 1:
            mid = (lo + hi) // 2
            passed, err = evaluate(mid)
            if passed:
                lo, good_err = mid, err
            else:
                hi = mid
        good_n    = lo
        capped_by = "tolerance"

    if verbose:
        ext_tag = "  (reference extended)" if reference_extended else ""
        print(
            f"  [Level Limit] dx={dx_fixed:.4g} ({num_points} pts): "
            f"maximum trustworthy n = {good_n}  "
            f"(error={good_err:.3e}, capped by {capped_by}){ext_tag}"
        )

    return {
        "maximum_levels":     good_n,
        "error_at_maximum":   good_err,
        "dx_fixed":           dx_fixed,
        "capped_by":          capped_by,
        "reference_extended": reference_extended,
        "trace":              trace,
        "tolerance":          tolerance,
        "metric":             metric,
        "num_points":         num_points,
    }


# =====================================================================
# Plot: resolution-limit search trace (x-axis = dx)
# =====================================================================
def plot_grid_limit_search(grid_result, system_name="System"):
    """
    Plot relative error vs grid spacing dx from a `find_minimum_grid_points`
    trace, with the tolerance line and the maximum_dx boundary marked.

    The x-axis runs from fine (small dx, left) to coarse (large dx,
    right), matching the natural reading direction: "anything to the
    left of the green line is safe". The y-axis shows the maximum
    relative error max|ΔEₙ/Eₙ| across all tested levels, so the
    safe flat floor is horizontal and easy to read regardless of how
    large the absolute energies are.

    Parameters
    ----------
    grid_result : dict
        Output of `find_minimum_grid_points`.
    system_name : str, optional
        Used in the plot title.

    Returns
    -------
    None (displays the figure).
    """
    BLUE, RED, GREEN = "#1f77b4", "#d62728", "#2ca02c"

    # Sort ascending in dx: fine grid on the left, coarse on the right.
    trace = sorted(grid_result["trace"], key=lambda t: t["dx"])
    dxs  = [t["dx"]  for t in trace]
    errs = [t["error"] for t in trace]
    pts  = [t["num_points"] for t in trace]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        f"{system_name} \u2014 DVR Resolution Limit\n"
        f"(n = {grid_result['num_levels']} levels, fixed span = {grid_result['span']:.3g})",
        fontsize=12, fontweight="bold",
    )

    ax.plot(dxs, errs, "o-", color=BLUE, markersize=5, linewidth=1.3,
            label="Tested grid spacings")

    ax.axhline(
        grid_result["tolerance"], color=RED, linestyle="--", linewidth=1.3,
        label=f"Tolerance = {grid_result['tolerance']:.1e}",
    )
    ax.axvline(
        grid_result["maximum_dx"], color=GREEN, linestyle=":", linewidth=1.8,
        label=(
            f"Maximum dx = {grid_result['maximum_dx']:.4g}"
            f"  ({grid_result['minimum_grid_points']} pts)"
        ),
    )

    ax.set_xlabel(r"Grid spacing  $\Delta x$", fontsize=12)
    ax.set_ylabel(r"Relative error  $\max_n |\Delta E_n / E_n|$", fontsize=11)
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


# =====================================================================
# Plot: level-count-limit search trace (x-axis = n, fixed dx in title)
# =====================================================================
def plot_level_limit_search(level_result, system_name="System"):
    """
    Plot relative error vs number-of-levels-requested from a
    `find_maximum_levels` trace, with the tolerance line and the maximum
    trustworthy n marked. The fixed grid spacing dx is shown in the
    subplot title so these results sit on the same physical footing as
    the resolution-limit plot above.

    The y-axis shows the maximum relative error max|ΔEₙ/Eₙ| across
    the n levels tested at each step. This gives a flat safe floor
    that is easy to compare across different potentials and energy scales.

    Parameters
    ----------
    level_result : dict
        Output of `find_maximum_levels`.
    system_name : str, optional
        Used in the plot title.

    Returns
    -------
    None (displays the figure).
    """
    BLUE, RED, GREEN = "#1f77b4", "#d62728", "#2ca02c"

    trace = sorted(level_result["trace"], key=lambda t: t["n"])
    ns   = [t["n"]     for t in trace]
    errs = [t["error"] for t in trace]

    dx_fixed  = level_result.get("dx_fixed", None)
    num_pts   = level_result.get("num_points", "?")
    dx_label  = f"\u0394x = {dx_fixed:.4g}" if dx_fixed is not None else ""

    fig, ax = plt.subplots(figsize=(8, 5))
    ext_note = "  [reference extended]" if level_result.get("reference_extended") else ""
    fig.suptitle(
        f"{system_name} \u2014 DVR Level-Count Limit\n"
        f"(fixed grid: {num_pts} pts, {dx_label}){ext_note}",
        fontsize=12, fontweight="bold",
    )

    ax.plot(ns, errs, "o-", color=BLUE, markersize=4, linewidth=1.3,
            label="Tested level counts (n)")
    ax.axhline(
        level_result["tolerance"], color=RED, linestyle="--", linewidth=1.3,
        label=f"Tolerance = {level_result['tolerance']:.1e}",
    )
    if level_result["maximum_levels"] is not None:
        ax.axvline(
            level_result["maximum_levels"], color=GREEN, linestyle=":", linewidth=1.8,
            label=f"Maximum trustworthy n = {level_result['maximum_levels']}",
        )
    ax.set_xlabel("Number of levels requested  n", fontsize=11)
    ax.set_ylabel(r"Relative error  $\max_n |\Delta E_n / E_n|$", fontsize=11)
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


# =====================================================================
# Orchestrator: run both searches, plot both, print summary
# =====================================================================
def run_dvr_limit_analysis(potential_func, system_name, reference_energies,
                            num_levels_for_grid_search, x_min, x_max,
                            num_points_for_level_search,
                            tolerance=1e-6, metric="max_rel", mass=1.0, hbar=1.0,
                            grid_search_kwargs=None, level_search_kwargs=None):
    """
    Convenience wrapper: run both `find_minimum_grid_points` (search A,
    x-axis = dx) and `find_maximum_levels` (search B, x-axis = n, fixed
    dx annotated) for the same system/reference/tolerance, produce both
    plots, and print a combined console summary. Call the two finder
    functions directly if you need independent tolerances or metrics.

    Parameters
    ----------
    potential_func : callable
        V(x), smooth, passed through to both searches.
    system_name : str
        Used in plot titles and console output.
    reference_energies : array_like
        Ground-truth spectrum (analytic or finer numerical). Shared by
        both searches. Should have at least
        num_points_for_level_search - 2 entries so search B is not
        artificially capped by reference length.
    num_levels_for_grid_search : int
        Fixed n for search A (the resolution-limit search).
    x_min, x_max : float
        Fixed span for both searches.
    num_points_for_level_search : int
        Fixed grid point count for search B (the level-count limit).
        Commonly the same grid used in the upstream DVR pipeline.
    tolerance : float, optional
        Shared error threshold (default 1e-6).
    metric : str, optional
        Shared error metric (default "max_abs").
    mass, hbar : float, optional
        Physical constants passed through to both searches.
    grid_search_kwargs : dict or None, optional
        Extra keyword overrides for `find_minimum_grid_points`
        (e.g. dx_start, grow_factor).
    level_search_kwargs : dict or None, optional
        Extra keyword overrides for `find_maximum_levels`
        (e.g. n_start, grow_factor).

    Returns
    -------
    dict with keys:
        grid_search : dict
            Full output of `find_minimum_grid_points`.
        level_search : dict
            Full output of `find_maximum_levels`.
    """
    grid_search_kwargs  = grid_search_kwargs  or {}
    level_search_kwargs = level_search_kwargs or {}

    span    = x_max - x_min
    dx_work = span / (num_points_for_level_search - 1)

    print(f"\n{'='*60}")
    print(f"  {system_name} \u2014 DVR Limit Analysis")
    print(f"  Span: [{x_min:.3g}, {x_max:.3g}]  (span = {span:.3g})")
    print(f"  Search A: n={num_levels_for_grid_search} fixed, sweeping dx")
    print(f"  Search B: dx={dx_work:.4g} fixed ({num_points_for_level_search} pts), sweeping n")
    print(f"{'='*60}")

    grid_result = find_minimum_grid_points(
        potential_func, num_levels_for_grid_search, x_min, x_max,
        reference_energies, tolerance=tolerance, metric=metric,
        mass=mass, hbar=hbar, **grid_search_kwargs,
    )
    plot_grid_limit_search(grid_result, system_name=system_name)

    level_result = find_maximum_levels(
        potential_func, x_min, x_max, num_points_for_level_search,
        reference_energies, tolerance=tolerance, metric=metric,
        mass=mass, hbar=hbar, **level_search_kwargs,
    )
    plot_level_limit_search(level_result, system_name=system_name)

    print(f"{'='*60}\n")
    return {"grid_search": grid_result, "level_search": level_result}