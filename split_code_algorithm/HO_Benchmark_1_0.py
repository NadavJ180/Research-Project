"""
HO_Benchmark_1_0.py
=====================================================================
WHAT THIS FILE DOES
---------------------------------------------------------------------
HO-only benchmark layer: takes the NUMERICAL results already produced
by Quantum_Classical_Combined_1_9.py's `run()` (the true quantum
Cv(T) curve from DVR energies + the numerically found classical
limit Cv(T)) and overlays them against the ANALYTICAL Harmonic
Oscillator solutions from HO_Analytical_1_0.py (the exact Einstein
Cv(T) formula + the exact classical limit, k_B). It then quantifies
the agreement (absolute/relative error vs temperature) and plots
everything together.

This is the piece that is only meaningful for a system with a known
closed-form answer -- which is exactly why it is HO-only. For future
systems with no analytic solution, this file is simply not
applicable; only the general pipeline (Quantum_Classical_Combined)
and DVR engine apply there.

CHANGELOG (NEW FILE, v1.0)
---------------------------------------------------------------------
- New file. Did not exist in any previous version of the project.
  Pulls together Quantum_Classical_Combined_1_9.run()'s numerical
  output with HO_Analytical_1_0.py's exact formulas to produce a
  dedicated benchmark comparison, including a quantitative Cv(T)
  error curve that the general pipeline alone does not produce.
=====================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

from HO_Analytical_1_0 import analytic_cv_HO_quantum, analytic_cv_HO_classical


# =====================================================================
# Quantify agreement between numerical and analytic quantum Cv(T)
# =====================================================================
def compute_cv_benchmark_error(cv_numeric, cv_analytic):
    """
    Compute the absolute and relative error between a numerically
    computed Cv(T) curve and the exact analytic Cv(T) curve, point
    by point. NaN entries in `cv_numeric` (e.g. where xi/n
    convergence failed at that temperature) are preserved as NaN in
    the output rather than silently dropped, so the error curve lines
    up 1:1 with the temperature axis.

    Parameters
    ----------
    cv_numeric : ndarray
        Numerically computed Cv(T) (e.g. cv_quantum from `run()`).
    cv_analytic : ndarray
        Exact analytic Cv(T) over the same temperature grid.

    Returns
    -------
    dict with keys:
        abs_error, rel_error : ndarray
            Per-temperature absolute/relative error (NaN preserved).
        max_abs_error, max_abs_idx : float, int
            Largest absolute error (ignoring NaNs) and its index.
        mean_abs_error, mean_rel_error : float
            NaN-safe mean absolute/relative error.
    """
    cv_numeric = np.asarray(cv_numeric, dtype=float)
    cv_analytic = np.asarray(cv_analytic, dtype=float)

    abs_error = np.abs(cv_numeric - cv_analytic)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel_error = abs_error / np.abs(cv_analytic)

    if np.all(np.isnan(abs_error)):
        max_abs_idx = -1
        max_abs_error = np.nan
    else:
        max_abs_idx = int(np.nanargmax(abs_error))
        max_abs_error = float(abs_error[max_abs_idx])

    return {
        "abs_error": abs_error,
        "rel_error": rel_error,
        "max_abs_error": max_abs_error,
        "max_abs_idx": max_abs_idx,
        "mean_abs_error": float(np.nanmean(abs_error)),
        "mean_rel_error": float(np.nanmean(rel_error)),
    }


# =====================================================================
# Benchmark plot: numeric vs analytic Cv(T), classical limits, and error
# =====================================================================
def plot_ho_cv_benchmark(T_arr, cv_quantum_numeric, cv_quantum_analytic,
                          cv_classical_numeric, cv_classical_analytic,
                          cv_error, T_units_label=r"$k_B T \,/\, \hbar\omega$"):
    """
    Two-panel benchmark figure for the Harmonic Oscillator:
        Top panel: quantum Cv(T) (numeric DVR-based vs exact analytic)
                   plus the classical-limit Cv (numeric vs exact k_B),
                   all on one set of axes.
        Bottom panel: the absolute error between numeric and analytic
                   quantum Cv(T), on a log y-axis so both the
                   low-T and high-T error behavior are visible.

    Parameters
    ----------
    T_arr : ndarray
        Temperature axis (shared by all curves).
    cv_quantum_numeric : ndarray
        Quantum Cv(T) computed numerically from DVR energies.
    cv_quantum_analytic : ndarray
        Exact analytic quantum Cv(T) (Einstein oscillator formula).
    cv_classical_numeric : ndarray
        Numerically found classical-limit Cv(T) (xi/n convergence).
    cv_classical_analytic : float
        Exact analytic classical limit (k_B for a 1-D HO).
    cv_error : dict
        Output of `compute_cv_benchmark_error`.
    T_units_label : str, optional
        X-axis label (LaTeX-formatted).

    Returns
    -------
    None (displays the figure).
    """
    BLUE, ORANGE, GREEN, PURPLE = "#1f77b4", "#d62728", "#2ca02c", "#9467bd"

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                                          gridspec_kw={"height_ratios": [2.2, 1]})
    fig.suptitle("Harmonic Oscillator \u2014 Numerical vs Analytical Benchmark", fontsize=13, fontweight="bold")

    # --- Top panel: the four Cv curves together ---
    ax_top.plot(T_arr, cv_quantum_numeric, color=BLUE, linewidth=2, label="Quantum Cv(T) \u2014 numerical (DVR)")
    ax_top.plot(T_arr, cv_quantum_analytic, color=ORANGE, linewidth=1.6, linestyle="--", label="Quantum Cv(T) \u2014 analytic (Einstein)")
    ax_top.plot(T_arr, cv_classical_numeric, color=GREEN, linewidth=1.8, linestyle="-.", label="Classical limit \u2014 numerical")
    ax_top.axhline(cv_classical_analytic, color=PURPLE, linewidth=1.5, linestyle=":", label=f"Classical limit \u2014 analytic ($k_B$ = {cv_classical_analytic:.3g})")
    ax_top.set_ylabel(r"$C_v \,/\, k_B$", fontsize=12)
    ax_top.set_xscale("log")
    ax_top.legend(fontsize=9, loc="upper left")
    ax_top.grid(True, linestyle="--", alpha=0.4)

    # --- Bottom panel: the actual quantum-Cv error curve ---
    ax_bot.plot(T_arr, cv_error["abs_error"], color=BLUE, linewidth=1.5)
    if cv_error["max_abs_idx"] >= 0:
        ax_bot.scatter([T_arr[cv_error["max_abs_idx"]]], [cv_error["max_abs_error"]],
                        color=ORANGE, zorder=5, s=60,
                        label=f"Max error = {cv_error['max_abs_error']:.2e}")
        ax_bot.legend(fontsize=9, loc="upper right")
    ax_bot.set_xlabel(T_units_label, fontsize=12)
    ax_bot.set_ylabel(r"$|Cv_{num} - Cv_{analytic}|$", fontsize=11)
    ax_bot.set_yscale("log")
    ax_bot.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()


# =====================================================================
# Orchestrator: build the analytic curves and run the full benchmark
# =====================================================================
def run_ho_benchmark(numeric_results, hbar=1.0, omega=1.0, kB=1.0):
    """
    Take the dict returned by Quantum_Classical_Combined_1_9.run()
    for the HO system, compute the matching analytic quantum Cv(T)
    curve and analytic classical limit over the same temperature
    grid, quantify the numeric-vs-analytic agreement, print a short
    summary, and produce the benchmark figure.

    Parameters
    ----------
    numeric_results : dict
        The dict returned by `Quantum_Classical_Combined_1_9.run()`
        when called on HO energy levels. Must contain "T_arr",
        "cv_quantum", and "cv_classical".
    hbar, omega, kB : float, optional
        Physical constants defining the HO system (default 1.0 each,
        matching the dimensionless units used throughout this project).

    Returns
    -------
    dict with keys:
        T_arr : ndarray
        cv_quantum_numeric, cv_quantum_analytic : ndarray
        cv_classical_numeric : ndarray
        cv_classical_analytic : float
        cv_error : dict
            Output of `compute_cv_benchmark_error`.
    """
    T_arr = numeric_results["T_arr"]
    cv_quantum_numeric = numeric_results["cv_quantum"]
    cv_classical_numeric = numeric_results["cv_classical"]

    cv_quantum_analytic = analytic_cv_HO_quantum(T_arr, hbar=hbar, omega=omega, kB=kB)
    cv_classical_analytic = analytic_cv_HO_classical(kB=kB)

    cv_error = compute_cv_benchmark_error(cv_quantum_numeric, cv_quantum_analytic)

    print(f"\n{'-'*60}")
    print("  Harmonic Oscillator: numerical vs analytical Cv(T) benchmark")
    print(f"{'-'*60}")
    print(f"  Mean |Cv error|:   {cv_error['mean_abs_error']:.3e}")
    print(f"  Max  |Cv error|:   {cv_error['max_abs_error']:.3e}  (at T = {T_arr[cv_error['max_abs_idx']]:.3g})" if cv_error['max_abs_idx'] >= 0 else "  Max  |Cv error|:   n/a (all NaN)")
    print(f"  Mean relative error: {cv_error['mean_rel_error']:.3e}")
    print(f"{'-'*60}\n")

    plot_ho_cv_benchmark(T_arr, cv_quantum_numeric, cv_quantum_analytic,
                          cv_classical_numeric, cv_classical_analytic, cv_error)

    return {
        "T_arr": T_arr,
        "cv_quantum_numeric": cv_quantum_numeric,
        "cv_quantum_analytic": cv_quantum_analytic,
        "cv_classical_numeric": cv_classical_numeric,
        "cv_classical_analytic": cv_classical_analytic,
        "cv_error": cv_error,
    }
