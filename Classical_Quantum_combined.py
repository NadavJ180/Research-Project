"""
================================================================================
Quantum Heat Capacity — Generalised Solver
================================================================================

PURPOSE
───────
Compute and plot the quantum heat capacity Cv(T) for ANY discrete energy
spectrum, together with its NUMERICAL classical limit (ξ → ∞) and the
minimum number of energy levels needed to converge the result at each
temperature.

This code replaces the analytical classical-limit lines in
Heat_Capacity_Graphs_HO_BOX.py with fully numerical ones, determined by
the same convergence engine developed in Classical_Limit_Box.py.

HOW IT WORKS
────────────
For EACH temperature T in the plot range the code performs two sweeps:

  1. ξ-CONVERGENCE SWEEP
     ξ is increased multiplicatively from xi_start until Cv(ξ) plateaus
     (|ΔCv| < tol_xi for min_stable_cl consecutive steps) and then
     collapses to zero as finite levels are exhausted.  The converged ξ
     is the mean of the last 3 points on the plateau just before collapse.

  2. n-CONVERGENCE SWEEP
     Starting from n = 2 levels, levels are added one at a time until
     |ΔCv| < tol_cv for min_stable_lc consecutive steps.  The converged
     n is the first level of that final stable run.

The converged (ξ, n) pair at each T is then used to evaluate Cv(T)
numerically.  The result is plotted as the "numerical classical limit".

DIAGNOSTIC PLOTS
────────────────
Rather than showing a convergence plot for every temperature (which would
be hundreds of plots), we show only the TWO HARDEST CASES:

  • ξ-convergence at the temperature that required the LARGEST converged ξ
    (usually the highest temperature, where many levels are populated and
    a bigger ξ is needed to wash out the quantum spacing).

  • n-convergence at the temperature that required the LARGEST converged n
    (usually the lowest temperature, where the sum converges slowly because
    the Boltzmann weights decay very gradually across many levels).

GENERALISATION
──────────────
The only system-specific input is the array `energies`.  Swap it for any
discrete spectrum — particle-in-a-box (n²), harmonic oscillator (n+½),
numerically computed levels from a potential well, etc.  Everything else
(convergence logic, plotting) is energy-level-agnostic.

FORMULA
───────
Cv / kB  =  (β² / ξ⁴) · [ <En²>_β  −  <En>_β² ]

where the thermal averages use scaled Boltzmann weights:
    w_n  = exp( −β · En / ξ² )    (shift-stabilised to prevent overflow)
    <f>  = Σ f(En) · w_n / Z ,    Z = Σ w_n

Setting ξ = 1 recovers the standard quantum Cv formula.
Setting ξ → ∞ washes out the discrete spacing → classical limit.

PARAMETERS (natural units)
──────────────────────────
    β_nat = E₀ / (kB · T[K])      dimensionless inverse temperature
    T[K]  = E₀ / (kB · β_nat)     Kelvin equivalent

E₀ is the energy scale used to define the dimensionless energies En.
For a 1-D box:  E₀ = E_ground = ħ²π²/(2mL²),  En = n² (dimensionless).
For a harmonic oscillator:  E₀ = ħω,  En = n + ½ (dimensionless).
================================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm   # progress bar 

# ── Physical constants ────────────────────────────────────────────────────────
KB_SI = 1.380649e-23    # Boltzmann constant in SI [J/K]

# When working in natural units (kB = 1, energies measured in energy units),
# pass E0_J = None (or omit it).  In that mode the β array is used directly
# as the inverse temperature axis, and T_K_arr is set to 1/β (dimensionless).
# The x-axis label should then be something like r"$k_B T / E_0$".


# ================================================================================
#  SECTION 1 — CORE Cv COMPUTATION
#  This function is the numerical heart of everything.  All convergence sweeps
#  and all Cv(T) evaluations call this one routine.
# ================================================================================

def compute_cv(energies, beta, xi=1.0):
    """
    Compute Cv / kB for a set of discrete energy levels at inverse temperature β
    and scaling factor ξ.

    Formula
    -------
        Cv/kB = (β²/ξ⁴) · ( <En²>  −  <En>² )

    where thermal averages use shifted Boltzmann weights for numerical stability:
        a_n     = β · En / ξ²          raw exponent
        w_n     = exp(−(a_n − a_min))  shift so largest weight = 1 (no overflow)
        <f>     = Σ f(En) · w_n / Z

    Parameters
    ----------
    energies : array-like   — dimensionless energy levels En (units of E₀)
    beta     : float        — natural-unit β = E₀ / (kB · T[K])
    xi       : float        — scaling factor ξ; default 1 = pure quantum formula

    Returns
    -------
    float   — Cv / kB,  or np.nan if partition function is degenerate / non-finite
    """
    energies = np.asarray(energies, dtype=float)

    # Compute Boltzmann exponents with ξ scaling, then subtract minimum to
    # prevent numerical overflow (shift cancels in all ratios).
    a      = beta * energies / (xi ** 2)
    a_min  = np.min(a)
    w      = np.exp(-(a - a_min))   # all weights in (0, 1], largest = 1

    Z = w.sum()
    if Z == 0 or not np.isfinite(Z):
        return np.nan   # degenerate partition function — skip this point

    avg_E  = np.dot(w, energies)      / Z   # first moment  <En>
    avg_E2 = np.dot(w, energies ** 2) / Z   # second moment <En²>

    # Variance of energy → Cv / kB
    return float((beta ** 2 / xi ** 4) * (avg_E2 - avg_E ** 2))


# ================================================================================
#  SECTION 2 — ξ CONVERGENCE AT A SINGLE TEMPERATURE
#  Sweeps ξ upward until Cv plateaus and then collapses to zero.
#  Returns the converged ξ (mean of last 3 plateau points) and diagnostic data.
# ================================================================================

def converge_xi(energies, beta, xi_start, tol_xi, min_stable, xi_multiplier, max_steps):
    """
    Find the ξ at which Cv(ξ) reaches the classical limit plateau at this β.

    Strategy
    --------
    ξ is multiplied by xi_multiplier at each step.  Two events are tracked:

      PLATEAU  — min_stable consecutive steps with |ΔCv| < tol_xi.
                 This is the classical limit region.

      ZERO FLOOR — last 3 computed Cv values all within atol=0.05 of zero.
                   This means finite levels are exhausted; we stop here.

    Convergence is declared if a plateau of length ≥ min_stable existed
    anywhere in the pre-zero region of the sweep.  The converged ξ is the
    mean of the last 3 plateau points (robust against edge noise).

    Parameters
    ----------
    energies      : dimensionless energy levels
    beta          : natural-unit β for this temperature
    xi_start      : starting ξ value
    tol_xi        : |ΔCv| threshold for plateau detection
    min_stable    : minimum consecutive stable steps for a valid plateau
    xi_multiplier : multiplicative step size for ξ
    max_steps     : hard cap on iterations

    Returns
    -------
    dict:
        xi_converged  — converged ξ value (None if not converged)
        cv_converged  — Cv at xi_converged (None if not converged)
        converged     — True if a valid plateau was found
        xi_values     — full list of ξ values swept
        cv_values     — full list of Cv values swept
        deltas        — list of |ΔCv| (None for first entry)
    """
    xis, cvs = [], []
    xi = xi_start
    stop_reason = "max_steps"

    for step in range(max_steps):
        cv = compute_cv(energies, beta, xi)
        xis.append(xi)
        cvs.append(cv)

        if step >= 2:
            # Check if the last 3 Cv values have all collapsed to the zero floor.
            # atol=0.05 is generous enough to catch near-zero but well above
            # typical plateau values (0.3–0.5 for box, ~1 for HO).
            last3 = np.array(cvs[-3:])
            if np.isclose(last3, 0.0, atol=0.2).all():
                # Zero floor reached — determine whether a real plateau preceded it.
                # Look at everything before the 3 zero-floor points.
                pre = cvs[:-3]
                if len(pre) >= min_stable + 1:
                    pre_d = [abs(pre[i] - pre[i-1]) for i in range(1, len(pre))]
                    # Scan for ANY contiguous run of min_stable stable steps.
                    # The plateau may be early in the sweep (before finite-N collapse).
                    streak, found = 0, False
                    for d in pre_d:
                        if d < tol_xi:
                            streak += 1
                            if streak >= min_stable:
                                found = True
                                break
                        else:
                            streak = 0
                    stop_reason = "converged" if found else "finite_n"
                else:
                    stop_reason = "finite_n"   # sweep too short for a plateau
                break

        xi *= xi_multiplier

    # Compute successive differences |ΔCv| for the full sweep (used for plots)
    deltas = [None] + [abs(cvs[i] - cvs[i-1]) for i in range(1, len(cvs))]

    # ── Extract converged ξ from the last 3 points of the plateau ────────────
    # The plateau is the last contiguous stable run in the pre-zero region.
    # Using the mean of 3 points (rather than the last single point) reduces
    # sensitivity to numerical noise at the plateau edge.
    if stop_reason == "converged":
        pre_cvs = cvs[:-3]   # drop the zero-floor entries
        pre_d   = [None] + [abs(pre_cvs[i] - pre_cvs[i-1])
                             for i in range(1, len(pre_cvs))]

        # Walk backwards to find the last stable index in the pre-zero region
        plat_end = None
        for i in range(len(pre_d) - 1, 0, -1):
            if pre_d[i] is not None and pre_d[i] < tol_xi:
                plat_end = i
                break

        if plat_end is not None:
            # Walk backward from plat_end to find the start of this stable run
            plat_start = plat_end
            while (plat_start > 1
                   and pre_d[plat_start - 1] is not None
                   and pre_d[plat_start - 1] < tol_xi):
                plat_start -= 1

            # Take the last (up to) 3 points of this plateau
            n_take     = min(3, plat_end - plat_start + 1)
            idx_start  = plat_end - n_take + 1
            stable_xis = xis[idx_start : plat_end + 1]
            stable_cvs = cvs[idx_start : plat_end + 1]
        else:
            # Fallback: shouldn't happen if stop_reason == "converged", but be safe
            stable_xis = xis[-6:-3]
            stable_cvs = cvs[-6:-3]

        xi_converged = float(np.mean(stable_xis))
        cv_converged = float(np.mean(stable_cvs))
    else:
        xi_converged = None
        cv_converged = None

    return {
        "xi_converged" : xi_converged,
        "cv_converged" : cv_converged,
        "converged"    : stop_reason == "converged",
        "stop_reason"  : stop_reason,
        "xi_values"    : xis,
        "cv_values"    : cvs,
        "deltas"       : deltas,
    }


# ================================================================================
#  SECTION 3 — n CONVERGENCE AT A SINGLE TEMPERATURE
#  Adds energy levels one at a time until Cv stops changing.
#  Returns the minimum n needed for convergence and diagnostic data.
# ================================================================================

def converge_n(energies, beta, xi, tol_cv, min_stable):
    """
    Find the minimum number of energy levels for Cv to converge at this β and ξ.

    Strategy
    --------
    Compute Cv for n = 2, 3, 4, … levels (using the first n entries of the
    energies array).  At each step compute |ΔCv| = |Cv(n) − Cv(n−1)|.
    Convergence is declared when min_stable consecutive steps all satisfy
    |ΔCv| < tol_cv AND this stable run extends to the last level n = N.

    The converged n is the first level of that final stable run.

    Parameters
    ----------
    energies   : FULL array of N energy levels (we use prefixes energies[:n])
    beta       : natural-unit β for this temperature
    xi         : converged ξ value from converge_xi (use xi=1 for quantum Cv)
    tol_cv     : |ΔCv| threshold for convergence
    min_stable : minimum consecutive stable steps

    Returns
    -------
    dict:
        n_converged — minimum n for convergence (None if not converged)
        cv_converged— Cv at n_converged
        converged   — True if convergence was found
        n_values    — list of n values tested
        cv_values   — Cv at each n
        deltas      — |ΔCv| at each n (None for first entry)
    """
    N = len(energies)
    n_values  = list(range(2, N + 1))
    cv_values = [compute_cv(energies[:n], beta, xi) for n in n_values]

    # Successive differences
    deltas = [None] + [abs(cv_values[i] - cv_values[i-1])
                       for i in range(1, len(cv_values))]

    # Boolean stability mask: True where |ΔCv| < tol_cv
    stable = [False] + [(d < tol_cv) for d in deltas[1:]]

    # Find contiguous stable runs of length ≥ min_stable
    runs = []   # list of (start_index, end_index) in n_values indexing
    i = 0
    while i < len(stable):
        if stable[i]:
            j = i
            while j < len(stable) and stable[j]:
                j += 1
            if j - i >= min_stable:
                runs.append((i, j - 1))
            i = j
        else:
            i += 1

    # A run ending at the last index = true convergence; earlier runs = shoulders
    last_idx     = len(n_values) - 1
    n_converged  = None
    cv_conv      = None

    for start_idx, end_idx in runs:
        if end_idx == last_idx:
            # The converged n is the first level of the final stable run
            n_converged = n_values[start_idx]
            cv_conv     = cv_values[start_idx]
            break   # only one true convergence run by definition

    return {
        "n_converged" : n_converged,
        "cv_converged": cv_conv,
        "converged"   : n_converged is not None,
        "n_values"    : n_values,
        "cv_values"   : cv_values,
        "deltas"      : deltas,
    }


# ================================================================================
#  SECTION 4 — PER-TEMPERATURE CONVERGENCE SWEEP OVER THE FULL T RANGE
#  Runs Sections 2 and 3 at every temperature point and collects results.
# ================================================================================

def sweep_temperature_range(
    energies, beta_arr,
    xi_start, tol_xi, min_stable_xi, xi_multiplier, max_xi_steps,
    tol_cv,   min_stable_n,
    verbose=True,
):
    """
    Run ξ-convergence and n-convergence at every β in beta_arr.

    For each temperature the function:
      1. Calls converge_xi  → finds xi_conv(T) and cv_classical(T)
      2. Calls converge_n   → finds n_conv(T) using xi_conv(T)

    If ξ-convergence fails at a temperature, the classical Cv is marked as NaN
    and n-convergence is skipped (we can't trust n without a valid ξ).

    Parameters
    ----------
    energies        : full energy level array
    beta_arr        : array of natural-unit β values (one per temperature point)
    xi_start        : starting ξ for the sweep
    tol_xi          : |ΔCv| tolerance for ξ-convergence
    min_stable_xi   : min consecutive stable steps for ξ-convergence
    xi_multiplier   : multiplicative ξ step
    max_xi_steps    : hard cap on ξ iterations per temperature
    tol_cv          : |ΔCv| tolerance for n-convergence
    min_stable_n    : min consecutive stable steps for n-convergence
    verbose         : show a tqdm progress bar

    Returns
    -------
    dict with arrays (one entry per temperature):
        cv_classical    — numerical classical Cv/kB at each T (NaN if failed)
        xi_conv         — converged ξ at each T (NaN if failed)
        n_conv          — converged n at each T (NaN if failed)
        xi_results      — list of raw dicts from converge_xi (for diagnostics)
        n_results       — list of raw dicts from converge_n  (for diagnostics)
        xi_fail_mask    — boolean array, True where ξ-convergence failed
        n_fail_mask     — boolean array, True where n-convergence failed
    """
    n_T = len(beta_arr)

    cv_classical  = np.full(n_T, np.nan)   # classical Cv at each T
    xi_conv_arr   = np.full(n_T, np.nan)   # converged ξ at each T
    n_conv_arr    = np.full(n_T, np.nan)   # converged n at each T
    xi_results    = []                      # raw convergence dicts for diagnostics
    n_results     = []

    iterator = tqdm(range(n_T), desc="  Sweeping T range", unit="T") if verbose else range(n_T)

    for idx in iterator:
        beta = beta_arr[idx]

        # ── Step A: ξ-convergence ─────────────────────────────────────────────
        xr = converge_xi(
            energies, beta,
            xi_start, tol_xi, min_stable_xi, xi_multiplier, max_xi_steps,
        )
        xi_results.append(xr)

        if not xr["converged"]:
            # ξ-sweep did not find a plateau.
            # Physical reason: for some spectra (e.g. harmonic oscillator) the
            # ξ-scaling crushes Cv to zero before a plateau can form — the
            # classical limit is reached simply by taking T high enough (small β)
            # with ξ=1.  In this case we fall back to ξ=1 for n-convergence:
            # the converged Cv at ξ=1 IS the numerical classical limit at that T.
            xi_use = 1.0
        else:
            # ξ-sweep succeeded: record the converged ξ and use it for n-sweep.
            xi_use = xr["xi_converged"]
            xi_conv_arr[idx]  = xr["xi_converged"]

        # ── Step B: n-convergence at the appropriate ξ ───────────────────────
        # If ξ-convergence succeeded, use xi_converged (classical-limit regime).
        # If it failed, use ξ=1 (standard quantum formula) — at high T this IS
        # the classical limit, so the converged Cv still gives the right answer.
        nr = converge_n(
            energies, beta,
            xi=xi_use,
            tol_cv=tol_cv,
            min_stable=min_stable_n,
        )
        n_results.append(nr)

        if nr["converged"]:
            n_conv_arr[idx] = nr["n_converged"]
            # If ξ-convergence failed but n-convergence succeeded, the classical
            # Cv at this T is the converged Cv from n-convergence at ξ=1.
            if not xr["converged"]:
                cv_classical[idx] = nr["cv_converged"]
        
        # Record the cv for successful xi-convergence cases here (after n-sweep)
        if xr["converged"]:
            cv_classical[idx] = xr["cv_converged"]

    xi_fail_mask = np.isnan(xi_conv_arr)
    n_fail_mask  = np.isnan(n_conv_arr)

    if verbose:
        n_xi_fail = xi_fail_mask.sum()
        n_n_fail  = n_fail_mask.sum()
        if n_xi_fail:
            print(f"  ⚠  ξ-convergence failed at {n_xi_fail}/{n_T} temperatures.")
        if n_n_fail:
            print(f"  ⚠  n-convergence failed at {n_n_fail}/{n_T} temperatures.")
        if not n_xi_fail and not n_n_fail:
            print(f"  ✓  Both ξ and n converged at all {n_T} temperatures.")

    return {
        "cv_classical"  : cv_classical,
        "xi_conv"       : xi_conv_arr,
        "n_conv"        : n_conv_arr,
        "xi_results"    : xi_results,
        "n_results"     : n_results,
        "xi_fail_mask"  : xi_fail_mask,
        "n_fail_mask"   : n_fail_mask,
    }


# ================================================================================
#  SECTION 5 — QUANTUM Cv(T) SWEEP  (ξ = 1, fixed n)
#  Standard quantum formula evaluated at a fixed number of levels.
# ================================================================================

def compute_quantum_cv_curve(energies, beta_arr, xi=1.0):
    """
    Compute the quantum Cv(T) curve using a fixed set of energy levels.

    This is the standard partition-function formula with ξ = 1 (no classical
    scaling).  The result depends on how many levels are included; use the
    n_conv values from sweep_temperature_range to choose a safe N.

    Parameters
    ----------
    energies : array of energy levels (first N of the full spectrum)
    beta_arr : array of β values
    xi       : scaling factor — 1.0 for standard quantum formula

    Returns
    -------
    cv_arr : array of Cv/kB values, one per β
    """
    return np.array([compute_cv(energies, b, xi) for b in beta_arr])


# ================================================================================
#  SECTION 6 — DIAGNOSTIC PLOT: ξ SWEEP AT THE HARDEST TEMPERATURE
#  Shows the full ξ sweep (Cv vs ξ) at the temperature requiring the largest ξ.
# ================================================================================

def plot_xi_convergence_diagnostic(xi_result, beta_val, T_K_val, tol_xi, system_name):
    """
    Plot the ξ-sweep convergence diagnostic for one temperature.

    Called with the result from converge_xi at the 'hardest' temperature
    (the one that required the largest converged ξ — worst case for convergence).

    Shows:
      - Cv vs ξ line with colour-coded dots (stable / falling / rising)
      - Marker at the converged ξ value
      - Annotation box summarising the outcome

    Parameters
    ----------
    xi_result   : dict returned by converge_xi
    beta_val    : β used at this temperature
    T_K_val     : temperature in Kelvin (for the title)
    tol_xi      : tolerance used (for the colour-coding threshold)
    system_name : string label for the plot title
    """
    BLUE   = "#1f77b4"
    GREEN  = "#2ca02c"
    ORANGE = "#d62728"
    YELLOW = "#bcbd22"
    GRAY   = "#7f7f7f"

    xis    = xi_result["xi_values"]
    cvs    = xi_result["cv_values"]
    deltas = xi_result["deltas"]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        f"{system_name} — ξ-Convergence Diagnostic\n"
        f"Hardest temperature: T = {T_K_val:.2f} K  (β = {beta_val:.4f})",
        fontsize=12, fontweight="bold",
    )

    # Main Cv vs ξ line
    ax.plot(xis, cvs, color=BLUE, linewidth=1.5, marker="s",
            markersize=5, zorder=3, label="Cv(ξ)")

    # Colour-code individual dots by local behaviour:
    #   GREEN  → |ΔCv| < tol_xi  (on the plateau)
    #   YELLOW → Cv falling (finite-N collapse)
    #   GRAY   → Cv rising or first point
    for i in range(len(xis)):
        d = deltas[i]
        if d is not None and d < tol_xi:
            c = GREEN
        elif i > 0 and cvs[i] < cvs[i - 1]:
            c = YELLOW
        else:
            c = GRAY
        ax.scatter([xis[i]], [cvs[i]], color=c, zorder=5, s=60)

    # Mark the converged ξ if found
    if xi_result["converged"]:
        xc = xi_result["xi_converged"]
        cc = xi_result["cv_converged"]
        ax.axvline(xc, color=GREEN, linestyle=":", linewidth=1.3,
                   label=f"ξ_conv = {xc:.3f}")
        ax.scatter([xc], [cc], color=GREEN, zorder=6, s=100,
                   label=f"Cv_conv = {cc:.4f}")
        ann = f"Converged ✓\nξ_conv = {xc:.3f}\nCv_conv/kB = {cc:.5f}"
        ann_colour = GREEN
    else:
        ann = f"NOT converged\n({xi_result['stop_reason']})\nIncrease N or adjust tol"
        ann_colour = ORANGE

    ax.text(0.97, 0.97, ann, transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color=ann_colour,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=ann_colour, alpha=0.9))

    # Dot-colour legend patches
    dot_legend = [
        mpatches.Patch(color=GREEN,  label="stable  |ΔCv| < tol"),
        mpatches.Patch(color=YELLOW, label="falling (finite-N collapse)"),
        mpatches.Patch(color=GRAY,   label="rising / first point"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + dot_legend, fontsize=9, loc="lower left")

    ax.set_xlabel("Scaling factor  ξ", fontsize=11)
    ax.set_ylabel("Cv / kB", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


# ================================================================================
#  SECTION 7 — DIAGNOSTIC PLOT: n SWEEP AT THE HARDEST TEMPERATURE
#  Shows Cv vs number of levels at the temperature needing the most levels.
# ================================================================================

def plot_n_convergence_diagnostic(n_result, beta_val, T_K_val, tol_cv, system_name):
    """
    Plot the n-convergence diagnostic for one temperature.

    Called with the result from converge_n at the 'hardest' temperature
    (the one that required the largest n to converge — worst case).

    Shows:
      - Cv vs n line
      - Vertical marker at the converged n
      - Shaded shoulder regions (temporary false convergences)

    Parameters
    ----------
    n_result    : dict returned by converge_n
    beta_val    : β used at this temperature
    T_K_val     : temperature in Kelvin (for the title)
    tol_cv      : tolerance used (for the colour-coding threshold)
    system_name : string label for the plot title
    """
    BLUE   = "#1f77b4"
    GREEN  = "#2ca02c"
    ORANGE = "#d62728"
    PURPLE = "#9467bd"

    ns     = n_result["n_values"]
    cvs    = n_result["cv_values"]
    deltas = n_result["deltas"]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(
        f"{system_name} — n-Convergence Diagnostic\n"
        f"Hardest temperature: T = {T_K_val:.2f} K  (β = {beta_val:.4f})",
        fontsize=12, fontweight="bold",
    )

    ax.plot(ns, cvs, color=BLUE, linewidth=1.5, marker="o",
            markersize=3, zorder=3, label="Cv(n levels)")

    # Mark convergence onset if found
    if n_result["converged"]:
        nc = n_result["n_converged"]
        idx = ns.index(nc)
        ax.axvline(nc, color=GREEN, linestyle=":", linewidth=1.3,
                   label=f"Converged at n = {nc}")
        ax.scatter([nc], [cvs[idx]], color=GREEN, zorder=6, s=100)
        ann = f"Converged ✓  at n = {nc}\nCv/kB = {cvs[idx]:.5f}"
        ann_colour = GREEN
    else:
        ann = "NOT converged within N levels\nIncrease N_MAX"
        ann_colour = ORANGE

    ax.text(0.97, 0.05, ann, transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9, color=ann_colour,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor=ann_colour, alpha=0.9))

    # Detect and shade shoulder regions (runs of ≥ 3 stable steps that end
    # before the last level — temporary false convergences)
    stable = [False] + [(d is not None and d < tol_cv) for d in deltas[1:]]
    i = 0
    while i < len(stable):
        if stable[i]:
            j = i
            while j < len(stable) and stable[j]:
                j += 1
            run_len = j - i
            if run_len >= 3 and j < len(stable):
                # This run ended before the last level → shoulder
                ax.axvspan(ns[i], ns[j - 1], color=PURPLE, alpha=0.18)
            i = j
        else:
            i += 1

    # Add a legend patch for shoulder shading if any were drawn
    handles, labels = ax.get_legend_handles_labels()
    # Add shoulder patch only if at least one shoulder was detected
    shoulder_exists = any(
        (not stable[k] or k == 0) is False and
        not (k == len(stable) - 1)
        for k in range(len(stable))
    )
    handles.append(mpatches.Patch(color=PURPLE, alpha=0.35, label="shoulder region"))
    labels.append("shoulder region")
    ax.legend(handles, labels, fontsize=9)

    ax.set_xlabel("Number of energy levels  n", fontsize=11)
    ax.set_ylabel("Cv / kB", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


# ================================================================================
#  SECTION 8 — MAIN Cv(T) FIGURE
#  Plots quantum Cv(T), numerical classical limit, and (optionally) an analytic
#  reference — all on a single figure with a log temperature axis.
# ================================================================================

def plot_cv_curves(
    T_K_arr, cv_quantum, cv_numerical_classical,
    xi_conv_arr, n_conv_arr,
    system_name,
    cv_analytic_classical=None,   # optional analytic reference (array or scalar)
    T_units_label=r"$k_B T \,/\, E_0$",
    T_scale_factor=1.0,           # divide T_K_arr by this to get the x-axis units
):
    """
    Plot the main Cv(T) figure for one system.

    Three curves:
      1. Quantum Cv(T)            — computed with ξ = 1
      2. Numerical classical limit — converged ξ and n at each T
      3. Analytic classical limit  — optional reference (e.g. 0.5 kB for box)

    A second y-axis (right side) shows the converged ξ(T) and n(T) to help
    the user understand how demanding convergence is at each temperature.

    Parameters
    ----------
    T_K_arr                  : temperature array in Kelvin
    cv_quantum               : quantum Cv/kB array
    cv_numerical_classical   : numerical classical Cv/kB array
    xi_conv_arr              : converged ξ at each T
    n_conv_arr               : converged n at each T
    system_name              : string for the plot title
    cv_analytic_classical    : optional analytic limit (scalar or array)
    T_units_label            : x-axis label
    T_scale_factor           : T_K_arr / T_scale_factor → x-axis values
    """
    BLUE   = "#1f77b4"
    GREEN  = "#2ca02c"
    ORANGE = "#d62728"
    RED    = "#d62728"
    PURPLE = "#9467bd"

    T_plot = T_K_arr / T_scale_factor   # convert Kelvin → desired x-axis units

    fig, ax1 = plt.subplots(figsize=(9, 6))
    fig.suptitle(f"{system_name} — Cv(T)", fontsize=13, fontweight="bold")

    # ── Cv curves ────────────────────────────────────────────────────────────
    ax1.plot(T_plot, cv_quantum, color=BLUE, linewidth=2,
             label="Quantum Cv(T)")

    ax1.plot(T_plot, cv_numerical_classical, color=GREEN,
             linewidth=2, linestyle="--",
             label="Numerical classical limit")

    if cv_analytic_classical is not None:
        # scalar → broadcast; array → plot as-is
        cv_ref = np.full_like(T_plot, cv_analytic_classical) \
                 if np.isscalar(cv_analytic_classical) else cv_analytic_classical
        ax1.plot(T_plot, cv_ref, color=ORANGE, linewidth=1.5,
                 linestyle=":", label="Analytic classical limit")

    ax1.set_xlabel(T_units_label, fontsize=12)
    ax1.set_ylabel(r"$C_v \,/\, k_B$", fontsize=12)
    ax1.set_xscale("log")
    ax1.legend(fontsize=10, loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.4)

    # ── Secondary axis: converged ξ(T) and n(T) ──────────────────────────────
    # Plotted on the right y-axis so they don't crowd the Cv scale.
    ax2 = ax1.twinx()

    # Only plot where convergence succeeded (non-NaN)
    valid_xi = ~np.isnan(xi_conv_arr)
    valid_n  = ~np.isnan(n_conv_arr)

    ax2.plot(T_plot[valid_xi], xi_conv_arr[valid_xi],
             color=PURPLE, linewidth=1, linestyle="-.",
             alpha=0.6, label="ξ_conv(T)")
    ax2.plot(T_plot[valid_n],  n_conv_arr[valid_n],
             color=RED, linewidth=1, linestyle=":",
             alpha=0.6, label="n_conv(T)")

    ax2.set_ylabel("Converged ξ  /  n  (secondary axis)", fontsize=10, color=PURPLE)
    ax2.tick_params(axis="y", colors=PURPLE)
    ax2.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    plt.show()


# ================================================================================
#  SECTION 9 — FULL PIPELINE
#  Ties everything together: energy levels in → all plots out.
# ================================================================================

def run(
    energies,
    E0_J,
    system_name,
    # ── Temperature range for the Cv(T) plot ─────────────────────────────────
    beta_min      = 0.02,   # lowest  β (highest T) — classical regime
    beta_max      = 5.0,    # highest β (lowest  T) — quantum regime
    n_beta        = 200,    # number of temperature points
    # ── ξ-convergence settings ───────────────────────────────────────────────
    xi_start      = 3.0,    # starting ξ for the sweep at each T
    tol_xi        = 5e-3,   # |ΔCv| tolerance for the ξ plateau
    min_stable_xi = 5,      # min consecutive stable steps on the ξ plateau
    xi_multiplier = 1.3,    # multiplicative step size (ξ *= xi_multiplier)
    max_xi_steps  = 80,     # hard cap on ξ iterations per temperature
    # ── n-convergence settings ───────────────────────────────────────────────
    tol_cv        = 1e-3,   # |ΔCv| tolerance for n-convergence
    min_stable_n  = 3,      # min consecutive stable steps on the n plateau
    # ── Optional analytic reference ──────────────────────────────────────────
    cv_analytic   = None,   # scalar or array; set to 0.5 for 1-D box, 1.0 for HO
    # ── Axis labels ──────────────────────────────────────────────────────────
    T_units_label = r"$k_B T \,/\, E_0$",   # x-axis label on the Cv(T) plot
    T_scale_factor= None,   # if None, computed as E0_J / KB_SI (gives kBT/E0 axis)
):
    """
    Full pipeline for one system:
      1. Build temperature (β) array
      2. Run ξ-convergence and n-convergence at every T   [Section 4]
      3. Compute quantum Cv(T) at fixed N = max(n_conv)   [Section 5]
      4. Show ξ-diagnostic at hardest T                   [Section 6]
      5. Show n-diagnostic at hardest T                   [Section 7]
      6. Show main Cv(T) plot                             [Section 8]

    Parameters
    ----------
    energies        : 1-D array of dimensionless energy levels (units of E₀)
    E0_J            : physical energy scale E₀ in Joules
                      Used only for Kelvin ↔ β conversion.
    system_name     : label string for plot titles
    beta_min/max    : β range for the temperature sweep
    n_beta          : number of temperature points
    xi_start        : starting ξ for every per-T sweep
    tol_xi          : ξ-plateau tolerance
    min_stable_xi   : ξ-plateau window
    xi_multiplier   : ξ step multiplier
    max_xi_steps    : ξ iteration cap
    tol_cv          : n-convergence tolerance
    min_stable_n    : n-convergence window
    cv_analytic     : optional analytic classical limit (scalar or array of length n_beta)
    T_units_label   : x-axis label
    T_scale_factor  : divisor to convert T[K] to desired x-axis units
                      (default: E0_J/KB_SI so x = kBT/E0)

    Returns
    -------
    results : dict with keys
        beta_arr, T_K_arr, cv_quantum, cv_classical,
        xi_conv, n_conv, sweep (raw sweep dict)
    """
    # ── Determine unit mode ───────────────────────────────────────────────────
    # NATURAL-UNIT MODE  (E0_J is None):
    #   beta is already dimensionless.  T* = 1/beta is kBT/E0.
    #   No Kelvin conversion needed; T_K_arr stores the dimensionless T*.
    #
    # SI MODE  (E0_J is a float in Joules):
    #   T[K] = E0_J / (KB_SI * beta)
    #   T_scale_factor converts T[K] to the desired x-axis units.
    #   Default (None): x = kBT/E0 = 1/beta (dimensionless).
    natural_units = (E0_J is None)

    if natural_units:
        T_scale_factor = 1.0           # x-axis = T* = 1/beta directly
    elif T_scale_factor is None:
        T_scale_factor = E0_J / KB_SI  # converts T[K] to kBT/E0

    # ── Build beta array and temperature array ────────────────────────────────
    beta_arr = np.linspace(beta_min, beta_max, n_beta)
    if natural_units:
        T_K_arr = 1.0 / beta_arr          # dimensionless T* = kBT/E0
    else:
        T_K_arr = E0_J / (KB_SI * beta_arr)   # temperature in Kelvin

    print(f"\n{'═'*64}")
    print(f"  {system_name}")
    print(f"{'═'*64}")
    print(f"  Energy levels : {len(energies)} levels,  "
          f"E_min = {energies[0]:.3g},  E_max = {energies[-1]:.3g}")
    e0_str = f"{E0_J:.3e} J" if E0_J is not None else "natural units (kB=1)"
    print(f"  E₀            : {e0_str}")
    print(f"  β range       : {beta_min} → {beta_max}   "
          f"({n_beta} points)")
    t_unit = "T*" if natural_units else "K"
    print(f"  T range       : {T_K_arr.min():.4g} {t_unit} → {T_K_arr.max():.4g} {t_unit}")
    print(f"  ξ settings    : start={xi_start},  mult={xi_multiplier},  "
          f"tol={tol_xi},  window={min_stable_xi}")
    print(f"  n settings    : tol={tol_cv},  window={min_stable_n}")

    # ── Step 1: Per-temperature convergence sweep ─────────────────────────────
    # This is the core computational step — may take a few seconds depending on
    # N, n_beta, and max_xi_steps.
    print(f"\n  Running per-temperature convergence sweep …")
    sweep = sweep_temperature_range(
        energies      = energies,
        beta_arr      = beta_arr,
        xi_start      = xi_start,
        tol_xi        = tol_xi,
        min_stable_xi = min_stable_xi,
        xi_multiplier = xi_multiplier,
        max_xi_steps  = max_xi_steps,
        tol_cv        = tol_cv,
        min_stable_n  = min_stable_n,
        verbose       = True,
    )

    cv_classical = sweep["cv_classical"]   # numerical classical limit
    xi_conv      = sweep["xi_conv"]        # converged ξ at each T
    n_conv       = sweep["n_conv"]         # converged n at each T

    # ── Step 2: Quantum Cv(T) with a fixed safe number of levels ─────────────
    # Use max(n_conv) levels — the worst-case number across all temperatures.
    # This guarantees the quantum curve is converged everywhere.
    # If n-convergence failed anywhere, fall back to all available levels.
    valid_n = n_conv[~np.isnan(n_conv)]  # ~ -> turn NaN to False and valid number to True
    if len(valid_n) > 0:
        n_quantum = int(np.max(valid_n))
        print(f"\n  Quantum Cv(T) computed with n = {n_quantum} levels "
              f"(worst-case converged n across T range).")
    else:
        n_quantum = len(energies)
        print(f"\n  WARNING: n-convergence failed at all temperatures.  "
              f"Using all {n_quantum} levels.")

    cv_quantum = compute_quantum_cv_curve(energies[:n_quantum], beta_arr, xi=1.0)

    # ── Step 3: Identify hardest temperatures for diagnostic plots ────────────
    # Hardest for ξ: temperature with the LARGEST converged ξ.
    #   Higher T → more levels thermally populated → needs larger ξ to wash
    #   out the spacing.  This is typically the hottest (smallest β) point.
    valid_xi_mask = ~np.isnan(xi_conv)
    if valid_xi_mask.any():
        idx_hard_xi = int(np.nanargmax(xi_conv))   # index of largest xi_conv
        T_hard_xi   = T_K_arr[idx_hard_xi]
        beta_hard_xi= beta_arr[idx_hard_xi]
        t_lbl = "T*" if natural_units else "K"
        print(f"\n  Hardest temperature for ξ-convergence: "
              f"T = {T_hard_xi:.4g} {t_lbl}  (β = {beta_hard_xi:.4f}),  "
              f"ξ_conv = {xi_conv[idx_hard_xi]:.3f}")
    else:
        idx_hard_xi = None
        print("\n  WARNING: ξ-convergence failed at ALL temperatures.  "
              "No ξ-diagnostic plot will be shown.")

    # Hardest for n: temperature with the LARGEST converged n.
    #   Lower T → more levels needed because Boltzmann weights decay slowly
    #   across a dense low-energy spectrum.
    valid_n_mask = ~np.isnan(n_conv)
    if valid_n_mask.any():
        idx_hard_n  = int(np.nanargmax(n_conv))    # index of largest n_conv
        T_hard_n    = T_K_arr[idx_hard_n]
        beta_hard_n = beta_arr[idx_hard_n]
        t_lbl2 = "T*" if natural_units else "K"
        print(f"  Hardest temperature for  n-convergence: "
              f"T = {T_hard_n:.4g} {t_lbl2}  (β = {beta_hard_n:.4f}),  "
              f"n_conv = {int(n_conv[idx_hard_n])}")
    else:
        idx_hard_n = None
        print("  WARNING: n-convergence failed at ALL temperatures.  "
              "No n-diagnostic plot will be shown.")

    # ── Step 4: Show ξ-diagnostic plot for the hardest temperature ────────────
    if idx_hard_xi is not None:
        xr_hard = sweep["xi_results"][idx_hard_xi]
        plot_xi_convergence_diagnostic(
            xi_result   = xr_hard,
            beta_val    = beta_hard_xi,
            T_K_val     = T_hard_xi,
            tol_xi      = tol_xi,
            system_name = system_name,
        )

    # ── Step 5: Show n-diagnostic plot for the hardest temperature ────────────
    if idx_hard_n is not None:
        nr_hard = sweep["n_results"][idx_hard_n]
        if nr_hard is not None:
            plot_n_convergence_diagnostic(
                n_result    = nr_hard,
                beta_val    = beta_hard_n,
                T_K_val     = T_hard_n,
                tol_cv      = tol_cv,
                system_name = system_name,
            )

    # ── Step 6: Main Cv(T) plot ───────────────────────────────────────────────
    plot_cv_curves(
        T_K_arr                = T_K_arr,
        cv_quantum             = cv_quantum,
        cv_numerical_classical = cv_classical,
        xi_conv_arr            = xi_conv,
        n_conv_arr             = n_conv,
        system_name            = system_name,
        cv_analytic_classical  = cv_analytic,
        T_units_label          = T_units_label,
        T_scale_factor         = T_scale_factor,
    )

    # ── Summary printout ──────────────────────────────────────────────────────
    print(f"\n  Summary for {system_name}")
    print(f"  {'─'*50}")
    print(f"  ξ-convergence : "
          f"{valid_xi_mask.sum()}/{n_beta} temperatures converged  "
          f"(max ξ = {np.nanmax(xi_conv):.2f} | N/A if all failed)" if valid_xi_mask.any() else "(ξ-convergence not applicable for this spectrum)")
    print(f"  n-convergence : "
          f"{valid_n_mask.sum()}/{n_beta} temperatures converged  "
          f"(max n = {int(np.nanmax(n_conv)) if valid_n_mask.any() else 'N/A'})")
    print(f"{'═'*64}\n")

    return {
        "beta_arr"    : beta_arr,
        "T_K_arr"     : T_K_arr,
        "cv_quantum"  : cv_quantum,
        "cv_classical": cv_classical,
        "xi_conv"     : xi_conv,
        "n_conv"      : n_conv,
        "sweep"       : sweep,
    }


# ================================================================================
#  SECTION 10 — ENTRY POINT
#  Configure your system here and call run().
#  To add a new system: define its energy levels array and call run() again.
# ================================================================================

if __name__ == "__main__":

    # ── Shared physical parameters ────────────────────────────────────────────
    # These are matched to the Heat_Capacity_Graphs_HO_BOX.py conventions:
    #   kB = 1,  m = 1,  L = 1,  hbar = 0.1
    hbar = 1 * 10**-34
    m    = 1.0
    L    = 1.0
    kB   = 1.0

    # Ground-state energy of the 1-D box (= E₀ for the box)
    E_g = (hbar**2 * np.pi**2) / (2 * m * L**2)

    # ── Shared convergence / sweep settings ──────────────────────────────────
    # These are used for BOTH systems below. Adjust per-system if needed.
    BETA_MIN      = 0.02     # lowest β  → highest T (classical regime)
    BETA_MAX      = 5.0      # highest β → lowest  T (quantum regime)
    N_BETA        = 200      # number of temperature points (reduce for speed)

    XI_START      = 3.0      # starting ξ for every per-T sweep
    TOL_XI        = 5e-3     # |ΔCv| tolerance for ξ-plateau detection
    MIN_STABLE_XI = 5        # min consecutive stable steps on the ξ plateau
    XI_MULT       = 1.3      # multiplicative ξ step (ξ *= XI_MULT each step)
    MAX_XI_STEPS  = 80       # hard cap on ξ iterations per temperature

    TOL_CV        = 1e-3     # |ΔCv| tolerance for n-convergence
    MIN_STABLE_N  = 3        # min consecutive stable steps on the n plateau

    # ────────────────────────────────────────────────────────────────────────
    #  SYSTEM 1 — 1-D PARTICLE-IN-A-BOX
    #  En = n² · E_g   (dimensionless: En_dimless = n²,  E₀ = E_g)
    #  Classical limit: Cv/kB = 0.5  (one quadratic degree of freedom)
    # ────────────────────────────────────────────────────────────────────────

    N_BOX   = 500   # number of box levels to include
                    # Increase if n-convergence warnings appear at low T.
    # Build dimensionless energy levels: En = n², n = 1, 2, …, N_BOX
    # The physical energies are En_dimless * E_g, but compute_cv works in
    # dimensionless units — E₀ = E_g is factored into β via E0_J.
    energies_box = np.array([n**2 for n in range(1, N_BOX + 1)], dtype=float)
    '''
    results_box = run(
        energies      = energies_box,
        E0_J          = None,         # natural-unit mode: kB=1, beta is dimensionless
        system_name   = "1-D Particle-in-a-Box",
        beta_min      = BETA_MIN,
        beta_max      = BETA_MAX,
        n_beta        = N_BETA,
        xi_start      = XI_START,
        tol_xi        = TOL_XI,
        min_stable_xi = MIN_STABLE_XI,
        xi_multiplier = XI_MULT,
        max_xi_steps  = MAX_XI_STEPS,
        tol_cv        = TOL_CV,
        min_stable_n  = MIN_STABLE_N,
        cv_analytic   = 0.5,          # analytic classical limit for 1-D box
        T_units_label = r"$k_B T / E_g$",
        # T_scale_factor is None -> defaults to 1.0 in natural-unit mode
    )
    '''
    # ────────────────────────────────────────────────────────────────────────
    #  SYSTEM 2 — 1-D QUANTUM HARMONIC OSCILLATOR
    #  En = (n + ½) · ħω   (dimensionless: En_dimless = n + 0.5,  E₀ = ħω)
    #  Classical limit: Cv/kB = 1.0  (two quadratic degrees of freedom)
    #
    #  Note: for the HO the ξ-convergence typically needs fewer levels
    #  than the box (evenly spaced spectrum converges faster), but we
    #  still run the same numerical procedure for consistency.
    # ────────────────────────────────────────────────────────────────────────

    hw   = 1.0      # ħω in the same energy units (kB = 1 here)
    N_HO = 200      # number of HO levels to include

    # Dimensionless HO energies: En = n + 0.5,  n = 0, 1, 2, …, N_HO-1
    # E₀ = ħω;  physical energy = En_dimless * ħω.
    energies_ho = np.array([n + 0.5 for n in range(N_HO)], dtype=float)

    results_ho = run(
        energies      = energies_ho,
        E0_J          = None,         # natural-unit mode: kB=1, beta is dimensionless
        system_name   = "1-D Harmonic Oscillator",
        beta_min      = BETA_MIN,
        beta_max      = BETA_MAX,
        n_beta        = N_BETA,
        xi_start      = XI_START,
        tol_xi        = TOL_XI,
        min_stable_xi = MIN_STABLE_XI,
        xi_multiplier = XI_MULT,
        max_xi_steps  = MAX_XI_STEPS,
        tol_cv        = TOL_CV,
        min_stable_n  = MIN_STABLE_N,
        cv_analytic   = 1.0,          # analytic classical limit for 1-D HO
        T_units_label = r"$k_B T / \hbar\omega$",
        # T_scale_factor is None -> defaults to 1.0 in natural-unit mode
    )

    # ────────────────────────────────────────────────────────────────────────
    #  TO ADD A NEW SYSTEM IN FUTURE:
    #  1. Define: energies_new = np.array([...], dtype=float)
    #  2. Call:   results_new = run(energies=energies_new, E0_J=..., ...)
    #  No other changes are needed — all convergence logic is energy-agnostic.
    # ────────────────────────────────────────────────────────────────────────
    energies_ho1 = np.array([n + 0.5 for n in range(10 * N_HO)], dtype=float)
    results_ho1 = run(
        energies      = energies_ho1,
        E0_J          = None,         # natural-unit mode: kB=1, beta is dimensionless
        system_name   = "1-D Harmonic Oscillator",
        beta_min      = BETA_MIN,
        beta_max      = BETA_MAX,
        n_beta        = N_BETA,
        xi_start      = 10 * XI_START,
        tol_xi        = TOL_XI,
        min_stable_xi = MIN_STABLE_XI,
        xi_multiplier = XI_MULT,
        max_xi_steps  = MAX_XI_STEPS,
        tol_cv        = TOL_CV,
        min_stable_n  = MIN_STABLE_N,
        cv_analytic   = 1.0,          # analytic classical limit for 1-D HO
        T_units_label = r"$k_B T / \hbar\omega$",
        # T_scale_factor is None -> defaults to 1.0 in natural-unit mode
    )