"""
Cv_Numerical_Benchmark_1_2.py
=====================================================================
WHAT THIS FILE DOES
---------------------------------------------------------------------
Benchmarks the quantum Cv(T) curve and the numerical classical-limit
Cv(T) curve from the BASE DVR pipeline against the same quantities
recomputed on a HIGH-PRECISION NUMERICAL REFERENCE grid (generated
by DVR_Reference_Generator_1_0.py). Produces two two-panel figures:

    Figure 1 -- Quantum Cv(T):
        top:    base-DVR Cv(T) vs reference Cv(T) on the same axes
        bottom: |Cv_base - Cv_ref| vs T  (ABSOLUTE error, log y-axis)

    Figure 2 -- Classical limit Cv(T):
        top:    base classical limit vs reference classical limit
        bottom: |Cv_base - Cv_ref| / |Cv_ref| vs T  (RELATIVE error,
                log y-axis, only where both base AND reference converged)

WHICH ERROR METRIC AND WHY
---------------------------------------------------------------------
QUANTUM Cv (Figure 1 bottom) -- ABSOLUTE ERROR:
    Relative error |ΔCv / Cv_ref| diverges to infinity at cold T
    because Cv → 0 exponentially (only the ground state is occupied).
    Both the base and reference give essentially zero, so their ratio
    is numerically ill-defined. Absolute error |ΔCv| remains finite
    and meaningful: it tells you directly how large the discrepancy
    is in the same units as Cv itself (k_B).

CLASSICAL LIMIT Cv (Figure 2 bottom) -- RELATIVE ERROR:
    The classical limit is only reported where the xi-scan converged,
    which happens in the temperature range where Cv is appreciably
    close to k_B (the classical plateau). Cv_ref is never near zero
    in this region, so relative error is well-defined and is the
    better metric: it normalises the comparison to the scale of the
    quantity being measured.

CHANGELOG (v1.1 -> v1.2)
---------------------------------------------------------------------
- QUANTUM Cv ERROR PANEL reverted from relative to ABSOLUTE error.
  Relative error diverges as Cv → 0 at low T (dividing ~0 by ~0),
  producing spurious infinite errors that obscure the actual accuracy
  of the DVR calculation. Absolute error is well-defined everywhere.

- CLASSICAL LIMIT ERROR PANEL keeps relative error from v1.1. The
  classical limit is always reported near k_B ≠ 0 (only where the
  xi-scan converged), so relative error is meaningful there.

- `print_cv_benchmark_summary` now prints absolute error for quantum
  Cv and relative error for classical limit, matching the plots.

- Module docstring updated to explain why each panel uses a
  different metric.
=====================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

from Classical_Limit_Numerical import sweep_temperature_range
from Quantum_Classical_Combined import compute_quantum_heat_capacity_curve


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
    Compute the absolute AND relative error between a base Cv curve
    and a reference Cv curve, NaN-safe (NaNs in either input propagate
    to NaN in all output arrays rather than crashing or becoming 0).

    Both metrics are computed and stored so the caller can choose which
    to use for plotting or thresholding. The plot functions in this file
    use the relative error by default.

    Parameters
    ----------
    cv_base, cv_ref : array_like
        Cv(T) arrays of the same shape. May contain NaN where
        convergence failed (typical for the classical limit at cold T).

    Returns
    -------
    dict with keys:
        abs_error   : ndarray -- |cv_base - cv_ref|, NaN-safe
        rel_error   : ndarray -- |cv_base - cv_ref| / |cv_ref|, NaN-safe
        max_abs     : float   -- nanmax of abs_error
        mean_abs    : float   -- nanmean of abs_error
        max_abs_idx : int     -- index of the maximum absolute error (-1 if all NaN)
        max_rel     : float   -- nanmax of rel_error
        mean_rel    : float   -- nanmean of rel_error
        max_rel_idx : int     -- index of the maximum relative error (-1 if all NaN)
    """
    cb = np.asarray(cv_base, dtype=float)
    cr = np.asarray(cv_ref,  dtype=float)

    abs_error = np.abs(cb - cr)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_error = abs_error / np.abs(cr)

    # Return early if all values are NaN (e.g. classical limit failed everywhere)
    if np.all(np.isnan(abs_error)):
        return {
            "abs_error": abs_error, "rel_error": rel_error,
            "max_abs": np.nan, "mean_abs": np.nan, "max_abs_idx": -1,
            "max_rel": np.nan, "mean_rel": np.nan, "max_rel_idx": -1,
        }

    max_abs_idx = int(np.nanargmax(abs_error))
    max_rel_idx = int(np.nanargmax(rel_error))

    return {
        "abs_error":   abs_error,
        "rel_error":   rel_error,
        "max_abs":     float(np.nanmax(abs_error)),
        "mean_abs":    float(np.nanmean(abs_error)),
        "max_abs_idx": max_abs_idx,
        "max_rel":     float(np.nanmax(rel_error)),
        "mean_rel":    float(np.nanmean(rel_error)),
        "max_rel_idx": max_rel_idx,
    }


# =====================================================================
# Figure 1: quantum Cv benchmark (two-panel)
# =====================================================================
def plot_quantum_cv_comparison(T_arr, cv_base, cv_ref, error_result,
                                system_name, reference_label,
                                T_units_label=r"$k_B T / \hbar\omega$"):
    """
    Two-panel comparison of quantum Cv(T): base DVR vs numerical reference.

    Top panel:    both Cv(T) curves on the same log-T axes.
    Bottom panel: ABSOLUTE error |Cv_base - Cv_ref| vs T (log y-axis).
                  The temperature of maximum absolute error is marked.

    WHY ABSOLUTE ERROR HERE:
    Quantum Cv(T) → 0 exponentially at low T (only the ground state
    is occupied). At these temperatures both the base and reference
    give essentially zero, so their ratio is numerically ill-defined
    (dividing ~0 by ~0 produces spurious large or infinite relative
    errors even when the DVR is perfectly accurate). Absolute error
    |ΔCv| stays finite everywhere and is directly interpretable in
    units of k_B.

    Parameters
    ----------
    T_arr : ndarray
        Temperature axis (= 1/beta_arr).
    cv_base, cv_ref : ndarray
        Quantum Cv(T) from the base DVR and the reference DVR.
    error_result : dict
        Output of `compute_cv_comparison_error`. The bottom panel uses
        the "abs_error", "max_abs", and "max_abs_idx" keys.
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
                label="Quantum Cv(T) \u2014 numerical reference")
    ax_top.set_ylabel(r"$C_v / k_B$", fontsize=12)
    ax_top.set_xscale("log")
    ax_top.legend(fontsize=10, loc="upper left")
    ax_top.grid(True, linestyle="--", alpha=0.4)

    # --- Bottom panel: ABSOLUTE error ---
    ax_bot.plot(T_arr, error_result["abs_error"], color=BLUE, linewidth=1.5)
    idx = error_result["max_abs_idx"]
    if idx >= 0 and not np.isnan(error_result["max_abs"]):
        ax_bot.scatter([T_arr[idx]], [error_result["max_abs"]],
                        color=RED, zorder=5, s=60,
                        label=f"Max abs. error = {error_result['max_abs']:.2e}")
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
    base DVR vs numerical reference.

    Top panel:    both classical-limit curves on the same log-T axes.
    Bottom panel: relative error |Cv_base - Cv_ref| / |Cv_ref| vs T
                  (log y-axis). Only temperatures where BOTH the base
                  and reference xi-scans converged are plotted in the
                  error panel (NaN entries from failed convergence are
                  silently masked). The joint-convergence count is shown
                  in the figure title.

    WHY RELATIVE ERROR: see `plot_quantum_cv_comparison`. The classical-
    limit curve is approximately flat at k_B at high T and falls toward
    zero at cold T (where the classical limit is not physically
    reachable). Relative error normalises the cold-T region correctly
    so the error comparison is fair across the full temperature range.

    Parameters
    ----------
    T_arr : ndarray
        Temperature axis.
    cv_classical_base, cv_classical_ref : ndarray
        Classical-limit Cv(T) from the base / reference pipeline.
        Both may contain NaN where xi-convergence failed.
    error_result : dict
        Output of `compute_cv_comparison_error` for the classical limit.
        The bottom panel uses the "rel_error" key.
    system_name : str
    reference_label : str
    T_units_label : str, optional

    Returns
    -------
    None (displays the figure).
    """
    GREEN, ORANGE, RED = "#2ca02c", "#d62728", "#d62728"

    # Only show error where BOTH base and reference converged
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
    ax_top.set_ylim(0, 1.1)
    ax_top.set_xscale("log")
    ax_top.legend(fontsize=10, loc="upper left")
    ax_top.grid(True, linestyle="--", alpha=0.4)

    if both_valid.any():
        # Mask relative error to jointly-converged temperatures only
        rel_err_masked = np.where(both_valid, error_result["rel_error"], np.nan)
        ax_bot.plot(T_arr[both_valid], rel_err_masked[both_valid],
                    color=GREEN, linewidth=1.5)
        # Mark the temperature of maximum relative error
        valid_err = rel_err_masked[both_valid]
        valid_T   = T_arr[both_valid]
        if not np.all(np.isnan(valid_err)):
            peak_idx = int(np.nanargmax(valid_err))
            ax_bot.scatter([valid_T[peak_idx]], [valid_err[peak_idx]],
                            color=RED, zorder=5, s=60,
                            label=f"Max rel. error = {valid_err[peak_idx]:.2e}")
            ax_bot.legend(fontsize=9, loc="upper right")
    else:
        ax_bot.text(0.5, 0.5, "No jointly-converged temperatures",
                    transform=ax_bot.transAxes, ha="center", va="center",
                    fontsize=10, color="gray")

    ax_bot.set_xlabel(T_units_label, fontsize=12)
    ax_bot.set_ylabel(
        r"$|Cv_{\rm base} - Cv_{\rm ref}|\,/\,|Cv_{\rm ref}|$", fontsize=11
    )
    ax_bot.set_yscale("log")
    ax_bot.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


# =====================================================================
# Print summary to console
# =====================================================================
def print_cv_benchmark_summary(quantum_err, classical_err, system_name, reference_label):
    """
    Print a concise numerical summary of base vs reference Cv errors,
    using the same error metric as the corresponding plot:
        - Quantum Cv:        ABSOLUTE error |ΔCv|  (matches Figure 1)
        - Classical limit:   RELATIVE error |ΔCv/Cv_ref|  (matches Figure 2)

    Parameters
    ----------
    quantum_err, classical_err : dict
        Outputs of `compute_cv_comparison_error`. Must contain
        "max_abs", "mean_abs" (for quantum) and "max_rel", "mean_rel"
        (for classical).
    system_name : str
    reference_label : str

    Returns
    -------
    None (prints to stdout).
    """
    print(f"\n{'-'*60}")
    print(f"  {system_name}: Cv numerical benchmark")
    print(f"  Reference: {reference_label}")
    print(f"{'-'*60}")
    print(f"  Quantum Cv(T)  [absolute error |ΔCv|]:")
    print(f"    mean = {quantum_err['mean_abs']:.3e}")
    print(f"    max  = {quantum_err['max_abs']:.3e}")
    print(f"  Classical limit Cv(T)  [relative error |ΔCv/Cv_ref|]:")
    print(f"    mean = {classical_err['mean_rel']:.3e}")
    print(f"    max  = {classical_err['max_rel']:.3e}")
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
        1. Run the quantum Cv + classical limit sweep on `reference_energies`
           using the same xi/n parameters as the base pipeline.
        2. Compare both curves against the pre-computed `base_cv_results`
           via `compute_cv_comparison_error` (which returns both absolute
           and relative error arrays).
        3. Print a console summary of the relative errors.
        4. Produce Figure 1 (quantum Cv, relative error bottom panel) and
           Figure 2 (classical limit, relative error bottom panel).

    The bottom panels use DIFFERENT metrics for each figure:
        Figure 1 (quantum Cv):     absolute error |Cv_base - Cv_ref|
        Figure 2 (classical limit): relative error |Cv_base - Cv_ref| / |Cv_ref|
    See the module docstring for the rationale.

    Parameters
    ----------
    base_cv_results : dict
        Output of `Quantum_Classical_Combined.run()` (or the equivalent
        `_run_cv_pipeline` call) for the BASE DVR energies. Must contain
        keys "cv_quantum", "cv_classical", and "beta_arr".
    reference_energies : array_like
        High-precision reference energy spectrum (from
        DVR_Reference_Generator.generate_reference_energies).
    beta_arr : ndarray
        Shared inverse-temperature array (must match the one used to
        produce base_cv_results).
    system_name : str
        Used in figure titles and console output.
    reference_label : str
        Short description of the reference scaling, e.g.
        "span×2.0, dx÷2.0". Shown in figure titles and console output.
    xi_start, tol_xi, min_stable_xi, xi_multiplier, max_xi_steps :
        Xi-convergence parameters. Should match those used in the base
        pipeline so the two sweeps are directly comparable.
    tol_cv, min_stable_n :
        N-convergence parameters. Should match the base pipeline.
    T_units_label : str, optional
        LaTeX x-axis label for both Cv plots
        (default r"$k_B T / \\hbar\\omega$").

    Returns
    -------
    dict with keys:
        ref_cv_results  : dict  -- from _run_cv_pipeline on reference energies;
                                   contains "cv_quantum", "cv_classical",
                                   "n_quantum_used", "sweep"
        quantum_error   : dict  -- from compute_cv_comparison_error;
                                   contains abs_error, rel_error, max_abs,
                                   mean_abs, max_abs_idx, max_rel, mean_rel,
                                   max_rel_idx
        classical_error : dict  -- same structure as quantum_error
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