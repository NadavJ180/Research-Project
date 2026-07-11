"""
Quantum_Classical_Combined_1_9.py
=====================================================================
WHAT THIS FILE DOES
---------------------------------------------------------------------
General-purpose (system-agnostic) Cv pipeline: given an array of
energy levels for ANY system, this file computes the true quantum
Cv(T) curve directly from the spectrum, calls into
Classical_Limit_Numerical_1_0.py to find the numerical classical
limit Cv(T) across the same temperature range, and produces all the
diagnostic + summary plots (xi-convergence diagnostic, n-convergence
diagnostic, and the combined Cv(T) curve plot). Everything here is
driven by `run(energies, ...)`, which takes a pre-computed spectrum
in -- it has no opinion about where those energies came from (DVR,
analytic formula, anything).

This file does NOT contain any hard-coded physical systems (no Box,
no HO, no Double Well). Those now live in their own driver
files/sections so this stays a reusable, system-agnostic pipeline.

CHANGELOG (v1.8 -> v1.9)
---------------------------------------------------------------------
- MAJOR RESTRUCTURE: extracted the core numerical engine (compute_cv,
  converge_xi, converge_n, sweep_temperature_range) out into
  Classical_Limit_Numerical_1_0.py. This file now imports that engine
  rather than defining it locally.
- REMOVED `auto_configure_dvr` (moved into DVR_Algorithm_1_3.py,
  where it now belongs alongside the smooth-only DVR solver it feeds).
- REMOVED all hard-coded `__main__` system definitions (Box, HO,
  Double Well) that used to live at the bottom of this file. Per the
  current project scope, only the HO system is exercised, and it now
  lives in its own benchmark/master files so this pipeline file can
  stay strictly general-purpose and reusable for future systems.
- No change to the plotting code or to `run()`'s control flow versus
  v1.8 -- this is a pure code-organization split, not a physics change.
- Expanded module/function docstrings.
=====================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

from Classical_Limit_Numerical import compute_cv, sweep_temperature_range


# =====================================================================
# True quantum Cv(T) curve, directly from a (finite) energy spectrum
# =====================================================================
def compute_quantum_heat_capacity_curve(energies, beta_arr, xi=1.0):
    """
    Evaluate the literal quantum heat capacity Cv(T) (no xi-rescaling
    by default) at every beta in `beta_arr`, using the given,
    truncated energy spectrum directly.

    Parameters
    ----------
    energies : array_like
        Energy eigenvalues for the system (e.g. from DVR).
    beta_arr : array_like
        Inverse temperatures to evaluate Cv at.
    xi : float, optional
        Spectrum scaling factor (default 1.0 -- the real, unscaled
        quantum Cv). Non-default values are mainly useful for
        diagnostics; physically you want xi=1.0 here.

    Returns
    -------
    cv_curve : ndarray, shape (len(beta_arr),)
        Cv/k_B at each beta.
    """
    return np.array([compute_cv(energies, b, xi) for b in beta_arr])


# =====================================================================
# Diagnostic plot: how Cv(xi) behaved at the hardest-to-converge T
# =====================================================================
def plot_xi_convergence_diagnostic(xi_result, beta_val, T_K_val, tol_xi, system_name):
    """
    Plot Cv as a function of the scaling factor xi for a single
    (hardest-converging) temperature, color-coding each point by
    whether it was part of a stable plateau, falling toward a
    finite-N collapse, or still rising.

    Parameters
    ----------
    xi_result : dict
        Output of `converge_xi` for this temperature.
    beta_val : float
        The inverse temperature this diagnostic corresponds to.
    T_K_val : float
        The corresponding temperature (1/beta_val), used only for the title.
    tol_xi : float
        The tolerance used during xi-convergence (for coloring/legend).
    system_name : str
        Used in the plot title.

    Returns
    -------
    None (displays the figure).
    """
    BLUE, GREEN, ORANGE, YELLOW, GRAY = "#1f77b4", "#2ca02c", "#d62728", "#bcbd22", "#7f7f7f"
    xis, cvs, deltas = xi_result["xi_values"], xi_result["cv_values"], xi_result["deltas"]
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(f"{system_name} \u2014 \u03be-Convergence Diagnostic\nHardest T: {T_K_val:.2f}  (\u03b2 = {beta_val:.4f})", fontsize=12, fontweight="bold")
    ax.plot(xis, cvs, color=BLUE, linewidth=1.5, marker="s", markersize=5, zorder=3, label="Cv(\u03be)")
    for i in range(len(xis)):
        d = deltas[i]
        if d is not None and d < tol_xi:
            c = GREEN
        elif i > 0 and cvs[i] < cvs[i - 1]:
            c = YELLOW
        else:
            c = GRAY
        ax.scatter([xis[i]], [cvs[i]], color=c, zorder=5, s=60)
    if xi_result["converged"]:
        xc, cc = xi_result["xi_converged"], xi_result["cv_converged"]
        ax.axvline(xc, color=GREEN, linestyle=":", linewidth=1.3, label=f"\u03be_conv = {xc:.3f}")
        ax.scatter([xc], [cc], color=GREEN, zorder=6, s=100, label=f"Cv_conv = {cc:.4f}")
        ann, ann_colour = f"Converged \u2713\n\u03be_conv = {xc:.3f}\nCv_conv/kB = {cc:.5f}", GREEN
    else:
        ann, ann_colour = f"NOT converged\n({xi_result['stop_reason']})", ORANGE
    ax.text(0.97, 0.97, ann, transform=ax.transAxes, ha="right", va="top", fontsize=9, color=ann_colour,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=ann_colour, alpha=0.9))
    dot_legend = [mpatches.Patch(color=GREEN, label="stable  |\u0394Cv| < tol"), mpatches.Patch(color=YELLOW, label="falling (finite-N collapse)"), mpatches.Patch(color=GRAY, label="rising / first point")]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + dot_legend, fontsize=9, loc="lower left")
    ax.set_xlabel("Scaling factor  \u03be", fontsize=11)
    ax.set_ylabel("Cv / kB", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


# =====================================================================
# Diagnostic plot: how Cv(n) behaved at the hardest-to-converge T
# =====================================================================
def plot_n_convergence_diagnostic(n_result, beta_val, T_K_val, tol_cv, system_name):
    """
    Plot Cv as a function of the number of included energy levels n
    for a single (hardest-converging) temperature, marking where (if
    anywhere) it became stable for the remainder of the spectrum.

    Parameters
    ----------
    n_result : dict
        Output of `converge_n` for this temperature.
    beta_val : float
        The inverse temperature this diagnostic corresponds to.
    T_K_val : float
        The corresponding temperature (1/beta_val), used only for the title.
    tol_cv : float
        The tolerance used during n-convergence (for annotation only).
    system_name : str
        Used in the plot title.

    Returns
    -------
    None (displays the figure).
    """
    BLUE, GREEN, ORANGE, PURPLE = "#1f77b4", "#2ca02c", "#d62728", "#9467bd"
    ns, cvs, deltas = n_result["n_values"], n_result["cv_values"], n_result["deltas"]
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(f"{system_name} \u2014 n-Convergence Diagnostic\nHardest T: {T_K_val:.2f}  (\u03b2 = {beta_val:.4f})", fontsize=12, fontweight="bold")
    ax.plot(ns, cvs, color=BLUE, linewidth=1.5, marker="o", markersize=3, zorder=3, label="Cv(n levels)")
    if n_result["converged"]:
        nc = n_result["n_converged"]
        idx = ns.index(nc)
        ax.axvline(nc, color=GREEN, linestyle=":", linewidth=1.3, label=f"Converged at n = {nc}")
        ax.scatter([nc], [cvs[idx]], color=GREEN, zorder=6, s=100)
        ann, ann_colour = f"Converged \u2713  at n = {nc}\nCv/kB = {cvs[idx]:.5f}", GREEN
    else:
        ann, ann_colour = "NOT converged within N levels\nIncrease N_MAX", ORANGE
    ax.text(0.97, 0.05, ann, transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color=ann_colour,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=ann_colour, alpha=0.9))
    ax.set_xlabel("Number of energy levels  n", fontsize=11)
    ax.set_ylabel("Cv / kB", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


# =====================================================================
# Summary plot: quantum Cv(T), numerical classical limit, optional analytic overlay
# =====================================================================
def plot_cv_curves(T_arr, cv_quantum, cv_classical, xi_conv_arr, n_conv_arr, system_name,
                    cv_analytic_classical=None, T_units_label=r"$k_B T \,/\, E_0$"):
    """
    Plot the true quantum Cv(T) curve together with the numerically
    found classical-limit Cv(T) curve, plus (optionally) a reference
    analytic classical-limit curve/value. A secondary y-axis shows
    the converged xi and n values across temperature, which is a
    useful at-a-glance indicator of how hard convergence was at each T.

    Parameters
    ----------
    T_arr : ndarray
        Temperature values (= 1/beta_arr).
    cv_quantum : ndarray
        True quantum Cv(T), from `compute_quantum_heat_capacity_curve`.
    cv_classical : ndarray
        Numerical classical-limit Cv(T), from `sweep_temperature_range`.
    xi_conv_arr, n_conv_arr : ndarray
        Converged xi/n at each temperature (NaN where convergence failed).
    system_name : str
        Used in the plot title.
    cv_analytic_classical : float, ndarray, or None, optional
        If provided, overlaid as a reference "Analytic classical limit"
        curve (constant if a scalar is given).
    T_units_label : str, optional
        X-axis label (LaTeX-formatted), defaults to a generic
        dimensionless temperature label.

    Returns
    -------
    None (displays the figure).
    """
    BLUE, GREEN, ORANGE, PURPLE, RED = "#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#d62728"
    fig, ax1 = plt.subplots(figsize=(9, 6))
    fig.suptitle(f"{system_name} \u2014 Cv(T)", fontsize=13, fontweight="bold")
    ax1.plot(T_arr, cv_quantum, color=BLUE, linewidth=2, label="Quantum Cv(T)")
    ax1.plot(T_arr, cv_classical, color=GREEN, linewidth=2, linestyle="--", label="Numerical classical limit")
    if cv_analytic_classical is not None:
        cv_ref = np.full_like(T_arr, cv_analytic_classical) if np.isscalar(cv_analytic_classical) else cv_analytic_classical
        ax1.plot(T_arr, cv_ref, color=ORANGE, linewidth=1.5, linestyle=":", label="Analytic classical limit")
    ax1.set_xlabel(T_units_label, fontsize=12)
    ax1.set_ylabel(r"$C_v \,/\, k_B$", fontsize=12)
    ax1.set_xscale("log")
    ax1.legend(fontsize=10, loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax2 = ax1.twinx()
    valid_xi = ~np.isnan(xi_conv_arr)
    valid_n = ~np.isnan(n_conv_arr)
    ax2.plot(T_arr[valid_xi], xi_conv_arr[valid_xi], color=PURPLE, linewidth=1, linestyle="-.", alpha=0.6, label="\u03be_conv(T)")
    ax2.plot(T_arr[valid_n], n_conv_arr[valid_n], color=RED, linewidth=1, linestyle=":", alpha=0.6, label="n_conv(T)")
    ax2.set_ylabel("Converged \u03be  /  n  (secondary axis)", fontsize=10, color=PURPLE)
    ax2.tick_params(axis="y", colors=PURPLE)
    ax2.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    plt.show()


# =====================================================================
# Full pipeline: quantum Cv + numerical classical limit + all plots
# =====================================================================
def run(energies, system_name,
        beta_min=0.02, beta_max=5.0, n_beta=200,
        xi_start=1.0, tol_xi=1e-3, min_stable_xi=5, xi_multiplier=1.3, max_xi_steps=80,
        tol_cv=1e-4, min_stable_n=3,
        cv_analytic=None, T_units_label=r"$k_B T \,/\, E_0$"):
    """
    Run the full general-purpose Cv pipeline for ANY system given its
    energy spectrum: sweep the temperature range, find the numerical
    classical limit at every T (via Classical_Limit_Numerical_1_0),
    compute the true quantum Cv(T) curve, and produce the
    xi-convergence diagnostic, n-convergence diagnostic (each shown
    at the single hardest-to-converge temperature), and the combined
    Cv(T) summary plot.

    Parameters
    ----------
    energies : array_like
        Energy eigenvalues for the system (e.g. from DVR), ascending.
    system_name : str
        Human-readable system name, used in plot titles/console output.
    beta_min, beta_max, n_beta : float, float, int, optional
        Inverse-temperature sweep range and point count.
    xi_start, tol_xi, min_stable_xi, xi_multiplier, max_xi_steps :
        Passed through to `converge_xi` at every temperature.
    tol_cv, min_stable_n :
        Passed through to `converge_n` at every temperature.
    cv_analytic : float, ndarray, or None, optional
        If known, the analytic classical-limit Cv to overlay as a
        reference curve on the summary plot.
    T_units_label : str, optional
        X-axis label for the summary plot.

    Returns
    -------
    dict with keys:
        beta_arr, T_arr : ndarray
        cv_quantum, cv_classical : ndarray
        xi_conv, n_conv : ndarray
        sweep : dict
            Full output of `sweep_temperature_range` (includes
            per-temperature convergence traces for further inspection).
    """
    beta_arr = np.linspace(beta_min, beta_max, n_beta)
    T_arr = 1.0 / beta_arr

    print(f"\n{'\u2550'*60}\n  {system_name}\n{'\u2550'*60}")
    print(f"  {len(energies)} levels, E_min={energies[0]:.3g}, E_max={energies[-1]:.3g}")
    print(f"  \u03b2: {beta_min} \u2192 {beta_max}  ({n_beta} points)")

    sweep = sweep_temperature_range(
        energies, beta_arr,
        xi_start, tol_xi, min_stable_xi, xi_multiplier, max_xi_steps,
        tol_cv, min_stable_n, verbose=True,
    )
    cv_classical = sweep["cv_classical"]
    xi_conv = sweep["xi_conv"]
    n_conv = sweep["n_conv"]

    # Use the largest converged n found anywhere in the sweep to define
    # how many levels the "true quantum Cv(T)" curve below should use.
    valid_n = n_conv[~np.isnan(n_conv)]
    n_quantum = int(np.max(valid_n)) if len(valid_n) > 0 else len(energies)
    cv_quantum = compute_quantum_heat_capacity_curve(energies[:n_quantum], beta_arr, xi=1.0)

    valid_xi_mask = ~np.isnan(xi_conv)
    if valid_xi_mask.any():
        idx_hard_xi = int(np.nanargmax(xi_conv))
        print(f"  Hardest \u03be-convergence: T*={T_arr[idx_hard_xi]:.4g}, \u03be_conv={xi_conv[idx_hard_xi]:.3f}")
        plot_xi_convergence_diagnostic(sweep["xi_results"][idx_hard_xi], beta_arr[idx_hard_xi], T_arr[idx_hard_xi], tol_xi, system_name)

    valid_n_mask = ~np.isnan(n_conv)
    if valid_n_mask.any():
        idx_hard_n = int(np.nanargmax(n_conv))
        print(f"  Hardest n-convergence:  T*={T_arr[idx_hard_n]:.4g}, n_conv={int(n_conv[idx_hard_n])}")
        nr_hard = sweep["n_results"][idx_hard_n]
        if nr_hard is not None:
            plot_n_convergence_diagnostic(nr_hard, beta_arr[idx_hard_n], T_arr[idx_hard_n], tol_cv, system_name)

    plot_cv_curves(T_arr, cv_quantum, cv_classical, xi_conv, n_conv, system_name, cv_analytic_classical=cv_analytic, T_units_label=T_units_label)

    print(f"\n  \u03be-conv: {valid_xi_mask.sum()}/{n_beta}  (max \u03be={np.nanmax(xi_conv):.2f})" if valid_xi_mask.any() else "  \u03be-conv: not applicable")
    print(f"  n-conv: {valid_n_mask.sum()}/{n_beta}  (max n={int(np.nanmax(n_conv))})" if valid_n_mask.any() else "  n-conv: failed at all T")
    print(f"{'\u2550'*60}\n")

    return {"beta_arr": beta_arr, "T_arr": T_arr, "cv_quantum": cv_quantum, "cv_classical": cv_classical, "xi_conv": xi_conv, "n_conv": n_conv, "sweep": sweep}
