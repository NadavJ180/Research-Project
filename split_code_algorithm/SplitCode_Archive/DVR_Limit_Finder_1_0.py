"""
DVR_Limit_Finder_1_0.py
=====================================================================
WHAT THIS FILE DOES
---------------------------------------------------------------------
General-purpose (system-agnostic) tool for mapping out WHERE the
smooth-potential DVR solver (DVR_Algorithm_1_3.py) stops being
trustworthy, by comparing its output against a reference spectrum and
searching for the breakdown point in two complementary directions:

    (A) RESOLUTION LIMIT -- fix the number of energy levels n you
        want and the spatial span, then shrink the grid (fewer
        points => coarser dx) until the error vs reference exceeds a
        tolerance. Finds the minimum grid density still trustworthy
        for n levels.

    (B) LEVEL-COUNT LIMIT -- fix a grid (span + point count) -- i.e.
        fixed computational cost -- then grow the number of levels
        requested from that SAME grid until the error vs reference
        exceeds tolerance. Finds the highest state index that grid
        can still be trusted for.

Both searches work with ANY potential and ANY reference spectrum --
the reference does not have to be analytic. For systems with no
closed-form solution, pass in energies from a separately verified,
much finer/wider numerical DVR run instead; the search logic doesn't
know or care where the reference came from. This is the intended
generalization path: this file is what you reach for to sanity-check
DVR accuracy on a brand-new potential once HO_Analytical_1_0.py-style
ground truth is no longer available.

CHANGELOG (NEW FILE, v1.0)
---------------------------------------------------------------------
- New file. Reuses `compute_energy_level_errors` from
  HO_Energy_Level_Error_1_1.py (that function is generic despite its
  file's HO-specific name -- see that file's v1.1 changelog) rather
  than duplicating the same error-metric math here.
=====================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

from DVR_Algorithm_1_3 import colbert_miller_dvr_1d
from HO_Energy_Level_Error_1_1 import compute_energy_level_errors


# =====================================================================
# Shared pass/fail helper used by both searches
# =====================================================================
_METRIC_KEYS = {
    "max_abs": "max_abs_error", "max_rel": "max_rel_error",
    "mean_abs": "mean_abs_error", "mean_rel": "mean_rel_error",
}


def _passes_tolerance(E_test, reference_energies, tolerance, metric):
    """
    Compare a freshly computed DVR spectrum `E_test` against the
    first len(E_test) entries of `reference_energies`, and report
    whether the chosen error metric is within `tolerance`.

    Parameters
    ----------
    E_test : ndarray
        DVR-computed energy levels for the configuration being tested.
    reference_energies : array_like
        Ground-truth spectrum (analytic or finer numerical), must have
        at least len(E_test) entries.
    tolerance : float
        Threshold the chosen metric must not exceed to "pass".
    metric : str
        One of "max_abs", "max_rel", "mean_abs", "mean_rel".
        "max_*" means EVERY compared level must individually be
        within tolerance (strict, the recommended default); "mean_*"
        only requires the average error to be within tolerance
        (looser -- a single bad level can hide inside the average).

    Returns
    -------
    passed : bool
    error_value : float
        The scalar value of the chosen metric.
    error_result : dict
        Full output of `compute_energy_level_errors`, in case the
        caller wants more detail than the single scalar.
    """
    if metric not in _METRIC_KEYS:
        raise ValueError(f"Unknown metric '{metric}'. Choose from {list(_METRIC_KEYS)}.")
    n = len(E_test)
    if len(reference_energies) < n:
        raise ValueError(
            f"reference_energies only has {len(reference_energies)} entries but "
            f"{n} are needed -- supply a longer reference spectrum."
        )
    error_result = compute_energy_level_errors(E_test, np.asarray(reference_energies)[:n])
    error_value = error_result[_METRIC_KEYS[metric]]
    return (error_value <= tolerance), error_value, error_result


# =====================================================================
# (A) Resolution limit: shrink grid points at fixed n and fixed span
# =====================================================================
def find_minimum_grid_points(potential_func, num_levels, x_min, x_max, reference_energies,
                              tolerance=1e-6, metric="max_abs", mass=1.0, hbar=1.0,
                              num_points_start=None, shrink_factor=0.85,
                              max_anchor_points=200_000, verbose=True):
    """
    Find the RESOLUTION LIMIT of the DVR solver: the minimum number of
    grid points, at a FIXED spatial span [x_min, x_max], for which all
    of the lowest `num_levels` eigenvalues stay within `tolerance` of
    `reference_energies`.

    Method: start from a generously fine grid (verified to pass, growing
    it first if needed), then geometrically shrink num_points (by
    `shrink_factor` < 1 each step) until the error exceeds tolerance,
    then integer-bisect between the last passing and first failing
    point counts to pin the boundary down to single-grid-point precision.

    Parameters
    ----------
    potential_func : callable
        V(x), smooth (finite everywhere), passed straight through to
        the DVR solver.
    num_levels : int
        Fixed number of energy levels to test at every grid density.
    x_min, x_max : float
        Fixed spatial span (only grid DENSITY is varied here; a
        separate search would be needed to probe span/truncation
        effects independently).
    reference_energies : array_like
        Ground-truth energies, length >= num_levels. Analytic or a
        finer numerical reference -- this function doesn't care which.
    tolerance : float, optional
        Maximum allowed error (chosen `metric`'s units) to "pass" (default 1e-6).
    metric : str, optional
        See `_passes_tolerance` (default "max_abs").
    mass, hbar : float, optional
        Physical constants passed through to the DVR solver.
    num_points_start : int or None, optional
        Starting (assumed comfortably converged) grid point count.
        Defaults to 8x num_levels if not given.
    shrink_factor : float, optional
        Geometric shrink ratio applied to num_points each step while
        searching downward (default 0.85).
    max_anchor_points : int, optional
        Safety cap on how far the search will grow the starting point
        looking for a passing anchor before giving up (default 200000).
    verbose : bool, optional
        Print a one-line summary at the end (default True).

    Returns
    -------
    dict with keys:
        minimum_grid_points : int
            Smallest num_points found that still passes tolerance.
        dx_at_minimum : float
            Grid spacing at `minimum_grid_points`.
        error_at_minimum : float
            The error metric value at `minimum_grid_points`.
        capped_by : str
            "tolerance" if a genuine breakdown was found while
            shrinking, or "hard_floor" if the solver remained accurate
            all the way down to the smallest possible grid
            (num_levels + 2 points) without ever failing.
        trace : list of dict
            Every {"num_points", "error", "passed"} tested, in the
            order evaluated -- feed this to `plot_grid_limit_search`.
        tolerance, metric : echoed back for convenience.
    """
    if num_points_start is None:
        num_points_start = max(num_levels + 2, int(8 * num_levels))
    hard_floor = num_levels + 2  # colbert_miller_dvr_1d requires num_points > num_levels

    trace = []

    def evaluate(num_points):
        E_test = colbert_miller_dvr_1d(potential_func, num_levels, x_min, x_max, num_points, mass, hbar)
        passed, err, _ = _passes_tolerance(E_test, reference_energies, tolerance, metric)
        trace.append({"num_points": num_points, "error": err, "passed": passed})
        return passed, err

    # --- Step 1: secure a known-good anchor (grow if the default start fails) ---
    current = num_points_start
    passed, err = evaluate(current)
    while not passed and current < max_anchor_points:
        current = int(current * 1.5)
        passed, err = evaluate(current)
    if not passed:
        raise RuntimeError(
            f"No converged anchor grid found below {max_anchor_points} points for "
            f"n={num_levels}. Check that potential_func/reference_energies match, "
            f"or relax tolerance."
        )
    good_points, good_err = current, err

    # --- Step 2: geometrically shrink the grid until tolerance breaks ---
    bad_points = None
    capped_by = "tolerance"
    while good_points > hard_floor:
        candidate = max(hard_floor, int(good_points * shrink_factor))
        if candidate >= good_points:
            candidate = good_points - 1
        passed, err = evaluate(candidate)
        if passed:
            good_points, good_err = candidate, err
        else:
            bad_points = candidate
            break
    else:
        capped_by = "hard_floor"

    # --- Step 3: bisect between the last good and first bad point counts ---
    if bad_points is not None:
        lo, hi = good_points, bad_points  # lo passes, hi fails
        while hi - lo > 1:
            mid = (lo + hi) // 2
            passed, err = evaluate(mid)
            if passed:
                lo, good_err = mid, err
            else:
                hi = mid
        good_points = lo

    dx_at_minimum = (x_max - x_min) / (good_points - 1)

    if verbose:
        tag = "(solver held up all the way to the hard floor)" if capped_by == "hard_floor" else ""
        print(f"  [Resolution Limit] n={num_levels}: minimum grid points = {good_points} "
              f"(dx={dx_at_minimum:.4g}, error={good_err:.3e}) {tag}")

    return {
        "minimum_grid_points": good_points, "dx_at_minimum": dx_at_minimum,
        "error_at_minimum": good_err, "capped_by": capped_by,
        "trace": trace, "tolerance": tolerance, "metric": metric, "num_levels": num_levels,
    }


# =====================================================================
# (B) Level-count limit: grow n at a fixed grid
# =====================================================================
def find_maximum_levels(potential_func, x_min, x_max, num_points, reference_energies,
                         tolerance=1e-6, metric="max_abs", mass=1.0, hbar=1.0,
                         n_start=None, grow_factor=1.3, verbose=True):
    """
    Find the LEVEL-COUNT LIMIT of the DVR solver: for a FIXED grid
    (x_min, x_max, num_points) -- i.e. fixed computational cost -- the
    largest number of levels n for which all of the lowest n
    eigenvalues stay within `tolerance` of `reference_energies`.

    This answers "for the grid I've already built, how far up the
    spectrum can I trust the result?" The highest extracted
    eigenvalues from any fixed grid are always the least accurate
    (shortest wavelength relative to dx, and the wavefunction probes
    furthest toward the fixed boundary), so this is a genuinely
    different question from the resolution-limit search above.

    Method: start from a small, comfortably accurate n, then
    geometrically grow it (by `grow_factor` each step, respecting the
    solver's num_points > num_levels constraint and the length of
    `reference_energies`) until the error exceeds tolerance, then
    integer-bisect to pin down the exact crossing level.

    Parameters
    ----------
    potential_func : callable
        V(x), smooth, passed straight through to the DVR solver.
    x_min, x_max, num_points : float, float, int
        Fixed grid configuration (held constant throughout).
    reference_energies : array_like
        Ground-truth spectrum. Should ideally be at least
        num_points - 2 levels long so the search is capped by the
        grid itself rather than by running out of reference data.
    tolerance, metric : float, str, optional
        See `find_minimum_grid_points`.
    mass, hbar : float, optional
        Physical constants passed through to the DVR solver.
    n_start : int or None, optional
        Starting (assumed comfortably accurate) level count. Defaults
        to max(2, num_points // 20) if not given.
    grow_factor : float, optional
        Geometric growth ratio applied to n each step (default 1.3).
    verbose : bool, optional
        Print a one-line summary at the end (default True).

    Returns
    -------
    dict with keys:
        maximum_levels : int or None
            Largest n found that still passes tolerance. None only if
            even `n_start` already fails (the grid itself is too coarse).
        error_at_maximum : float
            The error metric value at `maximum_levels`.
        capped_by : str
            "tolerance" if a genuine breakdown was found, "grid_size"
            if growth was stopped by the num_points > num_levels
            constraint before ever failing, or "reference_length" if
            growth was stopped by running out of reference data first.
        trace : list of dict
            Every {"n", "error", "passed"} tested, in evaluated order
            -- feed this to `plot_level_limit_search`.
        tolerance, metric : echoed back for convenience.
    """
    hard_ceiling = num_points - 2  # colbert_miller_dvr_1d requires num_points > num_levels
    ref_ceiling = len(reference_energies)
    n_cap = min(hard_ceiling, ref_ceiling)
    if n_cap < 2:
        raise ValueError("Grid/reference too small to test any levels (n_cap < 2).")

    if n_start is None:
        n_start = max(2, num_points // 20)
    n_start = min(n_start, n_cap)

    trace = []

    def evaluate(n):
        E_test = colbert_miller_dvr_1d(potential_func, n, x_min, x_max, num_points, mass, hbar)
        passed, err, _ = _passes_tolerance(E_test, reference_energies, tolerance, metric)
        trace.append({"n": n, "error": err, "passed": passed})
        return passed, err

    passed, err = evaluate(n_start)
    if not passed:
        if verbose:
            print(f"  [Level Limit] Even n_start={n_start} fails tolerance on this grid "
                  f"(error={err:.3e}) -- the grid itself is too coarse for any analysis "
                  f"here; try a finer fixed grid.")
        return {"maximum_levels": None, "error_at_maximum": err, "capped_by": "tolerance",
                "trace": trace, "tolerance": tolerance, "metric": metric}

    good_n, good_err = n_start, err
    bad_n = None
    capped_by = "tolerance"

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
    else:
        capped_by = "grid_size" if hard_ceiling <= ref_ceiling else "reference_length"

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
        good_n = lo
        capped_by = "tolerance"

    if verbose:
        print(f"  [Level Limit] Grid({num_points} pts): maximum trustworthy n = {good_n} "
              f"(error={good_err:.3e}) -- capped by {capped_by}")

    return {
        "maximum_levels": good_n, "error_at_maximum": good_err, "capped_by": capped_by,
        "trace": trace, "tolerance": tolerance, "metric": metric, "num_points": num_points,
    }


# =====================================================================
# Plot: resolution-limit search trace
# =====================================================================
def plot_grid_limit_search(grid_result, system_name="System"):
    """
    Plot error vs grid point count from a `find_minimum_grid_points`
    trace (log y-axis), with the tolerance line and the found minimum
    grid point count both marked.

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
    trace = sorted(grid_result["trace"], key=lambda t: t["num_points"])
    pts = [t["num_points"] for t in trace]
    errs = [t["error"] for t in trace]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(f"{system_name} \u2014 DVR Resolution Limit  (n = {grid_result['num_levels']} levels, fixed span)", fontsize=12, fontweight="bold")
    ax.plot(pts, errs, "o-", color=BLUE, markersize=4, linewidth=1.3, label="Tested grid point counts")
    ax.axhline(grid_result["tolerance"], color=RED, linestyle="--", linewidth=1.3, label=f"Tolerance = {grid_result['tolerance']:.1e}")
    ax.axvline(grid_result["minimum_grid_points"], color=GREEN, linestyle=":", linewidth=1.5,
               label=f"Minimum viable points = {grid_result['minimum_grid_points']}")
    ax.set_xlabel("Grid points", fontsize=11)
    ax.set_ylabel(f"Error  ({grid_result['metric']})", fontsize=11)
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


# =====================================================================
# Plot: level-count-limit search trace
# =====================================================================
def plot_level_limit_search(level_result, system_name="System"):
    """
    Plot error vs number-of-levels-requested from a `find_maximum_levels`
    trace (log y-axis), with the tolerance line and the found maximum
    trustworthy level count both marked.

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
    ns = [t["n"] for t in trace]
    errs = [t["error"] for t in trace]

    fig, ax = plt.subplots(figsize=(8, 5))
    grid_label = f"  (fixed grid = {level_result['num_points']} points)" if "num_points" in level_result else ""
    fig.suptitle(f"{system_name} \u2014 DVR Level-Count Limit{grid_label}", fontsize=12, fontweight="bold")
    ax.plot(ns, errs, "o-", color=BLUE, markersize=4, linewidth=1.3, label="Tested level counts (n)")
    ax.axhline(level_result["tolerance"], color=RED, linestyle="--", linewidth=1.3, label=f"Tolerance = {level_result['tolerance']:.1e}")
    if level_result["maximum_levels"] is not None:
        ax.axvline(level_result["maximum_levels"], color=GREEN, linestyle=":", linewidth=1.5,
                   label=f"Maximum trustworthy n = {level_result['maximum_levels']}")
    ax.set_xlabel("Number of levels requested  n", fontsize=11)
    ax.set_ylabel(f"Error  ({level_result['metric']})", fontsize=11)
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


# =====================================================================
# Orchestrator: run both searches and produce both plots + a summary
# =====================================================================
def run_dvr_limit_analysis(potential_func, system_name, reference_energies,
                            num_levels_for_grid_search, x_min, x_max, num_points_for_level_search,
                            tolerance=1e-6, metric="max_abs", mass=1.0, hbar=1.0,
                            grid_search_kwargs=None, level_search_kwargs=None):
    """
    Convenience wrapper: run both `find_minimum_grid_points` and
    `find_maximum_levels` for the same system/reference/tolerance,
    plot both traces, and print a combined console summary. Call the
    two finder functions directly instead if you need independent
    tolerances/metrics for the two searches.

    Parameters
    ----------
    potential_func : callable
        V(x), smooth, passed through to both searches.
    system_name : str
        Used in plot titles and console output.
    reference_energies : array_like
        Ground-truth spectrum (analytic or finer numerical), shared by
        both searches. Should be long enough to cover
        num_points_for_level_search - 2 levels for the level search to
        not be artificially capped by reference length.
    num_levels_for_grid_search : int
        Fixed n used by the resolution-limit search.
    x_min, x_max : float
        Fixed span used by the resolution-limit search.
    num_points_for_level_search : int
        Fixed grid point count used by the level-count-limit search
        (commonly the same grid you actually intend to use downstream).
    tolerance, metric : float, str, optional
        Shared by both searches (default 1e-6, "max_abs").
    mass, hbar : float, optional
        Physical constants passed through to both searches.
    grid_search_kwargs, level_search_kwargs : dict or None, optional
        Extra keyword overrides forwarded to `find_minimum_grid_points`
        / `find_maximum_levels` respectively (e.g. shrink_factor,
        grow_factor, num_points_start, n_start).

    Returns
    -------
    dict with keys:
        grid_search : dict
            Output of `find_minimum_grid_points`.
        level_search : dict
            Output of `find_maximum_levels`.
    """
    grid_search_kwargs = grid_search_kwargs or {}
    level_search_kwargs = level_search_kwargs or {}

    print(f"\n{'='*60}\n  {system_name} \u2014 DVR Limit Analysis\n{'='*60}")

    grid_result = find_minimum_grid_points(
        potential_func, num_levels_for_grid_search, x_min, x_max, reference_energies,
        tolerance=tolerance, metric=metric, mass=mass, hbar=hbar, **grid_search_kwargs,
    )
    plot_grid_limit_search(grid_result, system_name=system_name)

    level_result = find_maximum_levels(
        potential_func, x_min, x_max, num_points_for_level_search, reference_energies,
        tolerance=tolerance, metric=metric, mass=mass, hbar=hbar, **level_search_kwargs,
    )
    plot_level_limit_search(level_result, system_name=system_name)

    print(f"{'='*60}\n")
    return {"grid_search": grid_result, "level_search": level_result}
