"""
HO_Energy_Level_Error_1_1.py
=====================================================================
WHAT THIS FILE DOES
---------------------------------------------------------------------
Accuracy check for the smooth-potential DVR solver: compares the
numerical Harmonic Oscillator energy levels produced by
DVR_Algorithm_1_3.py against the exact analytic levels from
HO_Analytical_1_0.py, level by level, and reports/plots how the
error grows with the state index n.

This is the tool you reach for whenever you want to know "how
accurate is my DVR grid, really?" -- which matters a lot once you
move on to systems (double wells, anharmonic potentials, etc.) that
have no analytic answer to compare against. Establishing the
accuracy here, where the truth is known exactly, tells you how much
to trust the same solver elsewhere.

CHANGELOG (v1.0 -> v1.1)
---------------------------------------------------------------------
- No logic change. Clarified the docstring of
  `compute_energy_level_errors` to make explicit that, despite this
  file's HO-specific name, that one function only ever compares two
  plain energy arrays and has no HO-specific assumptions baked in --
  it is reused as-is by DVR_Limit_Finder_1_0.py for general systems,
  where the second array may be a finer numerical reference solution
  instead of an analytic one.
=====================================================================
"""

import numpy as np
import matplotlib.pyplot as plt


# =====================================================================
# Compute absolute/relative error between numerical and analytic levels
# =====================================================================
def compute_energy_level_errors(E_numeric, E_analytic):
    """
    Compute level-by-level absolute and relative error between a
    numerically computed energy spectrum and a ground-truth spectrum
    it should match, plus a few summary statistics.

    NOTE ON GENERALITY: nothing in this function is HO-specific -- it
    only ever compares two plain arrays of the same length. `E_analytic`
    is named for this file's primary use case (an exact closed-form
    spectrum), but it can equally be a finer/independently-verified
    numerical reference solution for systems with no analytic answer.
    DVR_Limit_Finder_1_0.py reuses this exact function for that purpose.

    Parameters
    ----------
    E_numeric : array_like
        Numerically computed energy levels (e.g. from DVR), ascending.
    E_analytic : array_like
        Ground-truth energy levels to compare against (analytic, or a
        finer numerical reference), same length and ordering as
        E_numeric (state n in E_numeric must correspond to state n
        in E_analytic).

    Returns
    -------
    dict with keys:
        abs_error : ndarray
            |E_numeric[n] - E_analytic[n]| for every level n.
        rel_error : ndarray
            abs_error[n] / E_analytic[n] for every level n (relative
            error; meaningful here since E_analytic > 0 for all n).
        max_abs_error, max_abs_idx : float, int
            Largest absolute error and the state index n it occurs at.
        max_rel_error, max_rel_idx : float, int
            Largest relative error and the state index n it occurs at.
        mean_abs_error, mean_rel_error : float
            Spectrum-averaged absolute/relative error.
    """
    E_numeric = np.asarray(E_numeric, dtype=float)
    E_analytic = np.asarray(E_analytic, dtype=float)
    if E_numeric.shape != E_analytic.shape:
        raise ValueError(
            f"Shape mismatch: E_numeric has {E_numeric.shape[0]} levels, "
            f"E_analytic has {E_analytic.shape[0]} levels. Compute both "
            f"with the same num_levels."
        )

    abs_error = np.abs(E_numeric - E_analytic)
    rel_error = abs_error / E_analytic

    max_abs_idx = int(np.argmax(abs_error))
    max_rel_idx = int(np.argmax(rel_error))

    return {
        "abs_error": abs_error,
        "rel_error": rel_error,
        "max_abs_error": float(abs_error[max_abs_idx]),
        "max_abs_idx": max_abs_idx,
        "max_rel_error": float(rel_error[max_rel_idx]),
        "max_rel_idx": max_rel_idx,
        "mean_abs_error": float(np.mean(abs_error)),
        "mean_rel_error": float(np.mean(rel_error)),
    }


# =====================================================================
# Plot: numerical vs analytic energy levels, with a zoom on the worst error
# =====================================================================
def plot_energy_level_comparison(E_numeric, E_analytic, error_result=None, zoom=True, system_name="Harmonic Oscillator"):
    """
    Plot numerical vs analytic energy levels (E_n vs n) on the same
    axes. If `zoom=True` and a zoom window around the largest-error
    state is feasible (i.e. it isn't sitting right at the very first
    or very last level, where there's nothing to zoom into), a second
    subplot zooms into that region so the discrepancy is actually
    visible -- at low n the two curves are normally so close that the
    overall view alone won't show anything.

    Parameters
    ----------
    E_numeric : array_like
        Numerical (DVR) energy levels, ascending.
    E_analytic : array_like
        Exact analytic energy levels, same ordering as E_numeric.
    error_result : dict or None, optional
        Output of `compute_energy_level_errors`. If None, it is
        computed internally so this function can be used standalone.
    zoom : bool, optional
        Whether to attempt a zoomed-in inset/subplot around the
        worst-error region (default True).
    system_name : str, optional
        Used in the plot title.

    Returns
    -------
    None (displays the figure).
    """
    E_numeric = np.asarray(E_numeric, dtype=float)
    E_analytic = np.asarray(E_analytic, dtype=float)
    n = np.arange(len(E_numeric))

    if error_result is None:
        error_result = compute_energy_level_errors(E_numeric, E_analytic)

    worst_idx = error_result["max_abs_idx"]
    N = len(E_numeric)
    # A zoom only makes sense if there's a meaningful neighborhood around
    # the worst state to show, i.e. it isn't pinned to either edge.
    can_zoom = zoom and N >= 6 and 1 <= worst_idx <= N - 2

    BLUE, ORANGE = "#1f77b4", "#d62728"

    if can_zoom:
        fig, (ax_main, ax_zoom) = plt.subplots(1, 2, figsize=(12, 5))
    else:
        fig, ax_main = plt.subplots(figsize=(7, 5))

    fig.suptitle(f"{system_name} \u2014 Numerical (DVR) vs Analytic Energy Levels", fontsize=13, fontweight="bold")

    # --- Main (full-range) comparison ---
    ax_main.plot(n, E_analytic, color=ORANGE, linewidth=2, label="Analytic $E_n$")
    ax_main.plot(n, E_numeric, color=BLUE, linewidth=1.2, linestyle="--", marker=".", markersize=3, label="Numerical (DVR) $E_n$")
    ax_main.set_xlabel("State index  n", fontsize=11)
    ax_main.set_ylabel("$E_n$", fontsize=11)
    ax_main.set_title("Full spectrum", fontsize=11)
    ax_main.legend(fontsize=9)
    ax_main.grid(True, linestyle="--", alpha=0.5)
    if can_zoom:
        # Mark the worst-error state on the main plot for reference.
        ax_main.axvline(worst_idx, color="gray", linestyle=":", linewidth=1)

    # --- Zoomed-in comparison around the largest-error state ---
    if can_zoom:
        half_width = max(3, int(0.03 * N))
        lo = max(0, worst_idx - half_width)
        hi = min(N, worst_idx + half_width + 1)
        ax_zoom.plot(n[lo:hi], E_analytic[lo:hi], color=ORANGE, linewidth=2, marker="o", markersize=4, label="Analytic $E_n$")
        ax_zoom.plot(n[lo:hi], E_numeric[lo:hi], color=BLUE, linewidth=1.5, linestyle="--", marker="x", markersize=5, label="Numerical (DVR) $E_n$")
        ax_zoom.axvline(worst_idx, color="gray", linestyle=":", linewidth=1, label=f"Largest error at n={worst_idx}")
        ax_zoom.set_xlabel("State index  n", fontsize=11)
        ax_zoom.set_ylabel("$E_n$", fontsize=11)
        ax_zoom.set_title(f"Zoom near n={worst_idx} (largest abs. error)", fontsize=11)
        ax_zoom.legend(fontsize=9)
        ax_zoom.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


# =====================================================================
# Plot: error vs state index (the actual accuracy curve)
# =====================================================================
def plot_energy_level_error(error_result, system_name="Harmonic Oscillator"):
    """
    Plot absolute and relative error vs state index n on a log y-axis
    -- the most direct way to see how DVR accuracy degrades for
    higher excited states on a fixed grid.

    Parameters
    ----------
    error_result : dict
        Output of `compute_energy_level_errors`.
    system_name : str, optional
        Used in the plot title.

    Returns
    -------
    None (displays the figure).
    """
    abs_error = error_result["abs_error"]
    rel_error = error_result["rel_error"]
    n = np.arange(len(abs_error))

    BLUE, RED = "#1f77b4", "#d62728"
    fig, ax1 = plt.subplots(figsize=(8, 5))
    fig.suptitle(f"{system_name} \u2014 DVR Energy-Level Error vs State Index", fontsize=13, fontweight="bold")

    ax1.plot(n, abs_error, color=BLUE, linewidth=1.5, label="Absolute error  $|E_n^{num} - E_n^{analytic}|$")
    ax1.set_xlabel("State index  n", fontsize=11)
    ax1.set_ylabel("Absolute error", fontsize=11, color=BLUE)
    ax1.set_yscale("log")
    ax1.tick_params(axis="y", colors=BLUE)
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    ax2.plot(n, rel_error, color=RED, linewidth=1.2, linestyle="--", label="Relative error")
    ax2.set_ylabel("Relative error", fontsize=11, color=RED)
    ax2.set_yscale("log")
    ax2.tick_params(axis="y", colors=RED)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="upper left")

    plt.tight_layout()
    plt.show()


# =====================================================================
# Console summary
# =====================================================================
def print_accuracy_summary(error_result, num_levels, system_name="Harmonic Oscillator"):
    """
    Print a short human-readable accuracy summary to the console.

    Parameters
    ----------
    error_result : dict
        Output of `compute_energy_level_errors`.
    num_levels : int
        Total number of levels that were compared.
    system_name : str, optional
        Used in the printed header.

    Returns
    -------
    None (prints to stdout).
    """
    print(f"\n{'-'*60}")
    print(f"  {system_name}: DVR accuracy vs analytic energy levels")
    print(f"{'-'*60}")
    print(f"  Levels compared:      {num_levels}")
    print(f"  Mean abs error:       {error_result['mean_abs_error']:.3e}")
    print(f"  Mean rel error:       {error_result['mean_rel_error']:.3e}")
    print(f"  Max abs error:        {error_result['max_abs_error']:.3e}  (at n = {error_result['max_abs_idx']})")
    print(f"  Max rel error:        {error_result['max_rel_error']:.3e}  (at n = {error_result['max_rel_idx']})")
    print(f"{'-'*60}\n")
