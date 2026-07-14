"""
Cv_Numerical_Benchmark_1_0.py
=====================================================================
WHAT THIS FILE DOES
---------------------------------------------------------------------
Benchmarks the quantum Cv(T) curve and the numerical classical-limit
Cv(T) curve from the BASE DVR pipeline against the same quantities
recomputed on a HIGH-PRECISION NUMERICAL REFERENCE grid (generated
by DVR_Reference_Generator_1_0.py). Produces two two-panel figures:

    Figure 1 -- Quantum Cv(T):
        top:    base-DVR Cv(T) vs reference Cv(T) on the same axes
        bottom: |Cv_base - Cv_ref| vs T (log y-axis)

    Figure 2 -- Classical limit Cv(T):
        top:    base classical limit vs reference classical limit
        bottom: |classical_base - classical_ref| vs T (log y-axis,
                only where both base AND reference converged)

This file is SYSTEM-AGNOSTIC. It has no knowledge of the HO or any
other specific potential. All it needs are two pre-computed sets of
Cv curves (base and reference) and the shared temperature axis. The
reference Cv is computed internally using the same xi/n-convergence
engine (Classical_Limit_Numerical_1_0, Quantum_Classical_Combined_1_9)
that the base pipeline uses -- the only difference is the input
energy spectrum.

WHY THIS IS USEFUL FOR GENERAL SYSTEMS
---------------------------------------------------------------------
For systems with no analytic solution you cannot compare against a
known formula. Instead you compare the base result against a more
expensive (finer/wider) DVR computation. If the two agree to your
tolerance, the base result is self-consistently converged. This is
the same convergence philosophy used in the DVR limit finder, now
extended from energy levels alone to the thermodynamic observables
(Cv) that are the final physical output.

CHANGELOG (NEW FILE, v1.0)
---------------------------------------------------------------------
- New file. Parallel to HO_Benchmark_1_1.py but uses a numerically
  computed reference instead of an analytical formula. Designed to
  be the drop-in replacement for HO_Benchmark when moving to systems
  without analytic solutions.
=====================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

from Classical_Limit_Numerical_1_0 import sweep_temperature_range
from Quantum_Classical_Combined_1_9 import compute_quantum_heat_capacity_curve


# =====================================================================
# Run the full Cv pipeline (quantum + classical) on an energy spectrum
# =====================================================================
def _run_cv_pipeline(energies, beta_arr,
                     xi_start, tol_xi, min_stable_xi, xi_multiplier, max_xi_steps,
                     tol_cv, min_stable_n, label="", verbose=True):
    """
    Run sweep_temperature_range + compute_quantum_heat_capacity_curve
    for a given energy spectrum. Lightweight wrapper used internally
    by `run_cv_numerical_benchmark` to avoid duplicating sweep logic.

    Parameters
    ----------
    energies : array_like
        Energy spectrum to use (base DVR or reference DVR).
    beta_arr : ndarray
        Inverse-temperature array, shared with the base pipeline.
    xi_start, tol_xi, min_stable_xi, xi_multiplier, max_xi_steps :
        Passed through to sweep_temperature_range (xi-convergence).
    tol_cv, min_stable_n :
        Passed through to sweep_temperature_range (n-convergence).
    label : str, optional
        Short description printed in the tqdm bar ("base" / "reference").
    verbose : bool, optional
        Whether to show the tqdm progress bar (default True).

    Returns
    -------
    dict with keys:
        cv_quantum   : ndarray, shape (len(beta_arr),)
        cv_classical : ndarray, shape (len(beta_arr),) -- NaN where not converged
        n_quantum_used : int
        sweep : dict  (full sweep_temperature_range output)
    """
    if verbose and label:
        print(f"  Sweeping T range [{label}]:")

    sweep = sweep_temperature_range(
        energies, beta_arr,
        xi_start, tol_xi, min_stable_xi, xi_multiplier, max_xi_steps,
        tol_cv, min_stable_n,
        verbose=verbose,
    )

    # Use the maximum converged n found across the temperature sweep
    # to define how many levels the quantum Cv curve uses.
    valid_n = sweep["n_conv"][~np.isnan(sweep["n_conv"])]
    n_quantum = int(np.max(valid_n)) if len(valid_n) > 0 else len(energies)

    cv_quantum = compute_quantum_heat_capacity_curve(
        energies[:n_quantum], beta_arr, xi=1.0
    )

    return {
        "cv_quantum":      cv_quantum,
        "cv_classical":    sweep["cv_classical"],
        "n_quantum_used":  n_quantum,
        "sweep":           sweep,
    }


# =====================================================================
# NaN-safe error between two Cv curves
# =====================================================================
def compute_cv_comparison_error(cv_base, cv_ref):
    """
    Compute the absolute and relative error between a base Cv curve
    and a reference Cv curve, NaN-safe (NaNs in either input propagate
    to NaN in the output rather than crashing or silently becoming 0).

    Parameters
    ----------
    cv_base, cv_ref : array_like
        Cv(T) arrays of the same shape. May contain NaN where
        convergence failed (typical for classical limit at cold T).

    Returns
    -------
    dict with keys:
        abs_error  : ndarray  -- |cv_base - cv_ref|, NaN-safe
        rel_error  : ndarray  -- abs_error / |cv_ref|, NaN-safe
        max_abs    : float    -- nanmax of abs_error
        mean_abs   : float    -- nanmean of abs_error
        max_abs_idx : int     -- index of the maximum absolute error
    """
    cb = np.asarray(cv_base, dtype=float)
    cr = np.asarray(cv_ref,  dtype=float)

    abs_error = np.abs(cb - cr)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_error = abs_error / np.abs(cr)

    if np.all(np.isnan(abs_error)):
        return {"abs_error": abs_error, "rel_error": rel_error,
                "max_abs": np.nan, "mean_abs": np.nan, "max_abs_idx": -1}

    max_abs_idx = int(np.nanargmax(abs_error))
    return {
        "abs_error":   abs_error,
        "rel_error":   rel_error,
        "max_abs":     float(np.nanmax(abs_error)),
        "mean_abs":    float(np.nanmean(abs_error)),
        "max_abs_idx": max_abs_idx,
    }


# =====================================================================
# Figure 1: quantum Cv benchmark (two-panel)
# =====================================================================
def plot_quantum_cv_comparison(T_arr, cv_base, cv_ref, error_result,
                                system_name, reference_label,
                                T_units_label=r"$k_B T / \hbar\omega$"):
    """
    Two-panel comparison of quantum Cv(T): base DVR vs numerical reference.

    Top panel:   both Cv(T) curves on one set of axes (log-T x-axis).
    Bottom panel: |Cv_base - Cv_ref| vs T (log y-axis) with the
                  maximum-error point marked.

    Parameters
    ----------
    T_arr : ndarray
        Temperature axis (= 1/beta_arr).
    cv_base, cv_ref : ndarray
        Quantum Cv(T) from the base DVR and the reference DVR.
    error_result : dict
        Output of `compute_cv_comparison_error` for the quantum Cv.
    system_name : str
        Used in the figure title.
    reference_label : str
        Short description of the reference (e.g. "span×2, dx÷2").
    T_units_label : str, optional
        LaTeX x-axis label.

    Returns
    -------
    None (displays the figure).
    """
    BLUE, ORANGE, RED = "#1f77b4", "#d62728", "#d62728"

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(9, 8), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]},
    )
    fig.suptitle(
        f"{system_name} \u2014 Quantum Cv(T): Base DVR vs Numerical Reference\n"
        f"Reference: {reference_label}",
        fontsize=12, fontweight="bold",
    )

    ax_top.plot(T_arr, cv_base, color=BLUE, linewidth=2.0,
                label="Quantum Cv(T) \u2014 base DVR")
    ax_top.plot(T_arr, cv_ref, color=ORANGE, linewidth=1.6, linestyle="--",
                label=f"Quantum Cv(T) \u2014 numerical reference")
    ax_top.set_ylabel(r"$C_v / k_B$", fontsize=12)
    ax_top.set_xscale("log")
    ax_top.legend(fontsize=10, loc="upper left")
    ax_top.grid(True, linestyle="--", alpha=0.4)

    ax_bot.plot(T_arr, error_result["abs_error"], color=BLUE, linewidth=1.5)
    idx = error_result["max_abs_idx"]
    if idx >= 0 and not np.isnan(error_result["max_abs"]):
        ax_bot.scatter([T_arr[idx]], [error_result["max_abs"]],
                        color=RED, zorder=5, s=60,
                        label=f"Max error = {error_result['max_abs']:.2e}")
        ax_bot.legend(fontsize=9, loc="upper right")
    ax_bot.set_xlabel(T_units_label, fontsize=12)
    ax_bot.set_ylabel(r"$|Cv_{\rm base} - Cv_{\rm ref}|$", fontsize=11)
    ax_bot.set_yscale("log")
    ax_bot.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


# =====================================================================
# Figure 2: classical limit benchmark (two-panel)
# =====================================================================
def plot_classical_limit_comparison(T_arr, cv_classical_base, cv_classical_ref,
                                     error_result, system_name, reference_label,
                                     T_units_label=r"$k_B T / \hbar\omega$"):
    """
    Two-panel comparison of the numerical classical-limit Cv(T):
    base DVR vs numerical reference. Only temperatures where BOTH
    the base and reference converged are shown in the error panel
    (NaN entries from failed xi-convergence are silently masked).

    Parameters
    ----------
    T_arr : ndarray
        Temperature axis.
    cv_classical_base, cv_classical_ref : ndarray
        Classical-limit Cv(T) from the base / reference pipeline.
        Both may contain NaN where xi-convergence failed.
    error_result : dict
        Output of `compute_cv_comparison_error` for the classical limit.
    system_name : str
    reference_label : str
    T_units_label : str, optional

    Returns
    -------
    None (displays the figure).
    """
    GREEN, ORANGE, RED = "#2ca02c", "#d62728", "#d62728"

    # Mask entries where either curve is NaN (no convergence at that T)
    both_valid = ~(np.isnan(cv_classical_base) | np.isnan(cv_classical_ref))

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(9, 8), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1]},
    )
    fig.suptitle(
        f"{system_name} \u2014 Classical Limit Cv(T): Base DVR vs Numerical Reference\n"
        f"Reference: {reference_label}  |  "
        f"Points with both converged: {both_valid.sum()}/{len(T_arr)}",
        fontsize=12, fontweight="bold",
    )

    ax_top.plot(T_arr, cv_classical_base, color=GREEN, linewidth=2.0,
                linestyle="--", label="Classical limit \u2014 base DVR")
    ax_top.plot(T_arr, cv_classical_ref, color=ORANGE, linewidth=1.6,
                linestyle=":", label="Classical limit \u2014 numerical reference")
    ax_top.set_ylabel(r"$C_v / k_B$", fontsize=12)
    ax_top.set_xscale("log")
    ax_top.legend(fontsize=10, loc="upper left")
    ax_top.grid(True, linestyle="--", alpha=0.4)

    if both_valid.any():
        abs_err_masked = np.where(both_valid, error_result["abs_error"], np.nan)
        ax_bot.plot(T_arr[both_valid], abs_err_masked[both_valid],
                    color=GREEN, linewidth=1.5)
        # Mark maximum error among the jointly-valid temperatures
        valid_err = abs_err_masked[both_valid]
        valid_T   = T_arr[both_valid]
        if not np.all(np.isnan(valid_err)):
            peak_idx = int(np.nanargmax(valid_err))
            ax_bot.scatter([valid_T[peak_idx]], [valid_err[peak_idx]],
                            color=RED, zorder=5, s=60,
                            label=f"Max error = {valid_err[peak_idx]:.2e}")
            ax_bot.legend(fontsize=9, loc="upper right")
    else:
        ax_bot.text(0.5, 0.5, "No jointly-converged temperatures",
                    transform=ax_bot.transAxes, ha="center", va="center",
                    fontsize=10, color="gray")

    ax_bot.set_xlabel(T_units_label, fontsize=12)
    ax_bot.set_ylabel(r"$|Cv_{\rm base} - Cv_{\rm ref}|$", fontsize=11)
    ax_bot.set_yscale("log")
    ax_bot.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


# =====================================================================
# Print summary to console
# =====================================================================
def print_cv_benchmark_summary(quantum_err, classical_err, system_name, reference_label):
    """
    Print a concise numerical summary of base vs reference Cv errors.

    Parameters
    ----------
    quantum_err, classical_err : dict
        Outputs of `compute_cv_comparison_error` for quantum Cv and
        classical limit respectively.
    system_name : str
    reference_label : str

    Returns
    -------
    None
    """
    print(f"\n{'-'*60}")
    print(f"  {system_name}: Cv numerical benchmark")
    print(f"  Reference: {reference_label}")
    print(f"{'-'*60}")
    print(f"  Quantum Cv(T):")
    print(f"    mean |ΔCv| = {quantum_err['mean_abs']:.3e}")
    print(f"    max  |ΔCv| = {quantum_err['max_abs']:.3e}")
    print(f"  Classical limit Cv(T):")
    print(f"    mean |ΔCv| = {classical_err['mean_abs']:.3e}")
    print(f"    max  |ΔCv| = {classical_err['max_abs']:.3e}")
    print(f"{'-'*60}\n")


# =====================================================================
# Orchestrator: compute reference Cv, compare, plot, summarise
# =====================================================================
def run_cv_numerical_benchmark(base_cv_results, reference_energies, beta_arr,
                                system_name, reference_label,
                                xi_start, tol_xi, min_stable_xi,
                                xi_multiplier, max_xi_steps,
                                tol_cv, min_stable_n,
                                T_units_label=r"$k_B T / \hbar\omega$"):
    """
    Full numerical Cv benchmark:
        1. Run the quantum Cv + classical limit sweep on `reference_energies`.
        2. Compare both curves against the pre-computed `base_cv_results`.
        3. Print a numerical summary.
        4. Produce Figure 1 (quantum Cv comparison) and Figure 2
           (classical limit comparison).

    Parameters
    ----------
    base_cv_results : dict
        Output of `Quantum_Classical_Combined_1_9.run()` (or the
        equivalent `_run_cv_pipeline` call) for the BASE DVR energies.
        Must contain keys "cv_quantum" and "cv_classical".
    reference_energies : array_like
        High-precision reference energy spectrum (from
        DVR_Reference_Generator_1_0.generate_reference_energies).
    beta_arr : ndarray
        Shared inverse-temperature array (must match the one used to
        produce base_cv_results).
    system_name : str
        Used in figure titles and console output.
    reference_label : str
        Short description of the reference scaling, e.g.
        "span×2.0, dx÷2.0".
    xi_start, tol_xi, min_stable_xi, xi_multiplier, max_xi_steps :
        Xi-convergence parameters -- should match those used in the
        base pipeline so the two sweeps are directly comparable.
    tol_cv, min_stable_n :
        N-convergence parameters -- same as base pipeline.
    T_units_label : str, optional
        LaTeX x-axis label for the Cv plots.

    Returns
    -------
    dict with keys:
        ref_cv_results  : dict  -- from _run_cv_pipeline on reference energies
        quantum_error   : dict  -- from compute_cv_comparison_error
        classical_error : dict  -- from compute_cv_comparison_error
    """
    print(f"\n{'='*60}")
    print(f"  Cv Numerical Benchmark: {system_name}")
    print(f"  Reference: {reference_label}")
    print(f"{'='*60}")

    # Run the same pipeline (quantum Cv + classical limit) on the reference energies.
    # verbose=True keeps the tqdm bar so the user sees progress.
    ref_cv_results = _run_cv_pipeline(
        reference_energies, beta_arr,
        xi_start, tol_xi, min_stable_xi, xi_multiplier, max_xi_steps,
        tol_cv, min_stable_n,
        label="reference", verbose=True,
    )

    # Compute errors (NaN-safe for classical limit where convergence failed)
    quantum_err   = compute_cv_comparison_error(
        base_cv_results["cv_quantum"],
        ref_cv_results["cv_quantum"],
    )
    classical_err = compute_cv_comparison_error(
        base_cv_results["cv_classical"],
        ref_cv_results["cv_classical"],
    )

    T_arr = 1.0 / beta_arr

    print_cv_benchmark_summary(quantum_err, classical_err, system_name, reference_label)

    plot_quantum_cv_comparison(
        T_arr, base_cv_results["cv_quantum"], ref_cv_results["cv_quantum"],
        quantum_err, system_name, reference_label, T_units_label,
    )
    plot_classical_limit_comparison(
        T_arr, base_cv_results["cv_classical"], ref_cv_results["cv_classical"],
        classical_err, system_name, reference_label, T_units_label,
    )

    return {
        "ref_cv_results":  ref_cv_results,
        "quantum_error":   quantum_err,
        "classical_error": classical_err,
    }
