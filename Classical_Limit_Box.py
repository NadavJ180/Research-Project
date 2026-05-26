"""
================================================================================
Quantum Heat Capacity — Classical Limit Checker
================================================================================

FORMULA (from the image)
─────────────────────────
The heat capacity at constant volume for a quantum system with discrete energy
levels is derived from the partition function Z = Σ exp(-β·En/ξ²) and reads:

    Cv / kB  =  (β² / ξ⁴) · [ <En²>_β  −  <En>_β² ]

where the Boltzmann-weighted average is defined as:

    <f(En)>_β  =  Σ f(En) · exp(-β·En/ξ²)  /  Z

Parameters
──────────
    β  = 1 / (kB·T)  — inverse temperature in natural units (1/E₀)
                       T is in Kelvin; β = E₀ / (kB_SI · T[K])
    ξ              — independent scaling factor in the energy denominator;
                     increasing ξ → classical limit

Temperature in Kelvin
─────────────────────
Energies En are dimensionless multiples of a physical energy scale E₀ [J].
The inverse temperature used in the formula is:

    β_nat = E₀ / (kB_SI · T[K])          (dimensionless / natural-unit β)

so the temperature axis is:

    T[K] = E₀ / (kB_SI · β_nat)

E₀ is a user-supplied parameter (e.g. the ground-state energy of your box).

Pipeline (order of operations)
───────────────────────────────
1. CLASSICAL LIMIT CHECK first  — sweep ξ upward at fixed β,
   find the converged ξ and the average Cv over the stable window.
2. LEVEL CONVERGENCE CHECK next — run at the CONVERGED ξ found in step 1.
3. Cv(T) SWEEP                  — use the same CONVERGED ξ.

This guarantees that both diagnostics and the plot reflect the classical limit.

Convergence rules
─────────────────
• Classical limit: min_stable consecutive steps with |ΔCv| < tol_xi.
  Converged ξ  = mean of ξ values in the stable window.
  Converged Cv = mean of Cv values in the stable window.
  Direction-agnostic (Cv may approach limit from above OR below).
  Finite-N bail-out if Cv falls monotonically for bail_streak steps.

• Level convergence: |ΔCv| < tol_cv sustained to the last level.
  Shoulders (temporary plateaux) are detected and shaded separately.

Classical limit for 1D box:  Cv_classical / kB = 0.5
================================================================================
"""

from ast import Global
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Boltzmann constant in SI [J/K]
KB_SI = 1.380649e-23


# ================================================================================
#  CORE COMPUTATION
# ================================================================================

def compute_cv(energies, beta, xi):
    """
    Compute Cv / kB for a set of discrete energy levels.

    Formula
    -------
        Cv/kB = (β²/ξ⁴) · [ <En²>_β  −  <En>_β² ]

    where  <f>_β = Σ f(En) · w_n / Z,  w_n = exp(-β·En/ξ²),  Z = Σ w_n.

    Numerical stability fix
    -----------------------
    For large n (e.g. n > 1000, En = n² → 10⁶) the raw exponents β·En/ξ²
    grow large.  We subtract the MINIMUM exponent before calling exp so that
    every argument to exp is ≤ 0 (underflow only, never overflow).

        shifted weight: w_n = exp(-(a_n - a_min))  where a_n = β·En/ξ²

    Since a_n ≥ a_min, the argument -(a_n - a_min) ≤ 0 → no overflow.
    High-n levels whose weight underflows to 0 are simply negligible — correct.
    The shift cancels in all ratios so Cv is unchanged.

    Parameters
    ----------
    energies : 1-D array  — energy level values En  (dimensionless, units of E₀)
    beta     : float      — natural-unit β = E₀ / (kB_SI · T[K])
    xi       : float      — scaling factor ξ (independent of β)

    Returns
    -------
    float  — Cv / kB,  or np.nan if the partition function is degenerate.
    """
    energies  = np.asarray(energies, dtype=float)
    a         = beta * energies / (xi ** 2)    # exponents a_n = β·En/ξ²
    a_min     = np.min(a)                       # subtract minimum → all ≤ 0 after negation
    weights   = np.exp(-(a - a_min))            # w_n = exp(-(a_n - a_min)) ≤ 1

    Z = weights.sum()
    if Z == 0 or not np.isfinite(Z):
        return np.nan

    avg_E  = np.dot(weights, energies)       / Z   # <En>_β
    avg_E2 = np.dot(weights, energies ** 2) / Z    # <En²>_β

    return float((beta ** 2 / xi ** 4) * (avg_E2 - avg_E ** 2))


# ================================================================================
#  CHECK 1 — CLASSICAL LIMIT (ξ → ∞), DIRECTION-AGNOSTIC
#  Run this FIRST so the converged ξ can be passed to the level-convergence check.
# ================================================================================

def check_classical_limit(
    energies, beta, xi_start, cv_classical,
    xi_multiplier=1.5, tol_xi=0.005,
    min_stable=5, bail_streak=8, max_xi_steps=80,
    verbose=True,
):
    """
    Sweep ξ upward from xi_start and determine whether Cv stabilises.

    Convergence rule (direction-agnostic)
    ──────────────────────────────────────
    Only |ΔCv| between successive ξ values matters — NOT the sign.
    This means Cv can approach the classical limit from above OR below.
    Convergence is declared when min_stable consecutive steps all satisfy
    |ΔCv| < tol_xi.

    Converged values (averages over the stable window)
    ──────────────────────────────────────────────────
    Once stable, we average over all ξ and Cv values in the stable window
    rather than using only the last point. This gives a more robust estimate
    that is less sensitive to numerical noise at the edge of the window.

        xi_converged  = mean(ξ   in stable window)
        cv_converged  = mean(Cv  in stable window)

    These averages are returned and used downstream for:
      • The level-convergence check (xi_converged as the fixed ξ)
      • The Cv(T) sweep (xi_converged as the fixed ξ)

    Finite-N bail-out
    -----------------
    If Cv falls monotonically for bail_streak consecutive steps AND the
    cumulative drop exceeds tol_xi, we have exhausted the energy levels
    and stop to avoid a spurious collapse to zero.

    Parameters
    ----------
    energies      : full array of N energy levels
    beta          : natural-unit inverse temperature (fixed during sweep)
    xi_start      : initial value of ξ
    cv_classical  : expected classical Cv/kB (for annotation only)
    xi_multiplier : ξ is multiplied by this factor at each step
    tol_xi        : tolerance on |ΔCv| for stability
    min_stable    : number of consecutive stable steps required
    bail_streak   : monotone-fall steps before finite-N bail-out
    max_xi_steps  : hard cap on iterations

    Returns
    -------
    dict with keys:
        xi_values         — list of all ξ values evaluated
        cv_values         — Cv at each ξ
        deltas            — |ΔCv| (None for first entry)
        classical_reached — True if convergence was declared
        stopped_reason    — 'converged' | 'finite_n' | 'max_steps'
        xi_converged      — mean ξ over stable window  (or None)
        cv_converged      — mean Cv over stable window  (or None)
    """
    xis, cvs   = [], []
    xi          = xi_start
    stable_cnt  = 0
    stop_reason = "max_steps"

    for step in range(max_xi_steps):
        cv = compute_cv(energies, beta, xi)
        xis.append(xi)
        cvs.append(cv)

        if step >= 1:
            delta = abs(cv - cvs[-2])

            # ── Direction-agnostic convergence ───────────────────────────────
            # Count consecutive steps with |ΔCv| < tol_xi regardless of sign.
            if delta < tol_xi:
                stable_cnt += 1
                if stable_cnt >= min_stable:
                    stop_reason = "converged"
                    break
            else:
                stable_cnt = 0   # reset if a large step interrupts the plateau

            # ── Finite-N bail-out ─────────────────────────────────────────────
            # Detect a runaway monotone collapse that cannot self-correct.
            if step >= bail_streak:
                recent = cvs[-(bail_streak + 1):]
                diffs  = [recent[k+1] - recent[k] for k in range(bail_streak)]
                if all(d < 0 for d in diffs) and (recent[0] - recent[-1]) > tol_xi:
                    stop_reason = "finite_n"
                    break

        xi *= xi_multiplier

    # Compute |ΔCv| between every consecutive pair of ξ values
    deltas = [None] + [abs(cvs[i] - cvs[i-1]) for i in range(1, len(cvs))]

    classical_reached = (stop_reason == "converged")

    # ── Compute converged ξ and Cv as AVERAGES over the stable window ────────
    # The stable window is the last min_stable entries when convergence was met.
    # Using the mean reduces sensitivity to numerical noise at window edges.
    if classical_reached:
        stable_xis = xis[-min_stable:]       # ξ values in the stable window
        stable_cvs = cvs[-min_stable:]       # Cv values in the stable window
        xi_converged = float(np.mean(stable_xis))
        cv_converged = float(np.mean(stable_cvs))
    else:
        xi_converged = None
        cv_converged = None

    # ── Detect any transient shoulder (a short plateau that didn't persist) ──
    # A shoulder in the ξ sweep looks like a brief stable run followed by
    # renewed change — a sign of a temporary equilibrium before the final limit.
    stable_flags = [False] + [(d is not None and d < tol_xi) for d in deltas[1:]]
    shoulder_xi  = None
    i = 0
    while i < len(stable_flags):
        if stable_flags[i]:
            j = i
            while j < len(stable_flags) and stable_flags[j]:
                j += 1
            if j - i >= 2 and j < len(stable_flags):  # run ends before last step → shoulder
                shoulder_xi = xis[i]
                break
            i = j
        else:
            i += 1

    # ── Optional console output ───────────────────────────────────────────────
    if verbose:
        print(f"\n{'─'*64}")
        print(f"  Classical limit   β={beta:.4f}   tol={tol_xi}   "
              f"window={min_stable}   bail={bail_streak}")
        print(f"  Expected Cv_classical/kB = {cv_classical}")
        print(f"{'─'*64}")
        print(f"  {'ξ':>12}  {'Cv/kB':>14}  {'|ΔCv|':>14}  status")
        print(f"  {'─'*12}  {'─'*14}  {'─'*14}  ──────")
        for i in range(len(xis)):
            d   = deltas[i]
            d_s = f"{d:.8f}" if d is not None else "       —        "
            if d is not None and d < tol_xi:
                label = " ✓ stable"
            elif i > 0 and cvs[i] < cvs[i-1]:
                label = " ↓ falling"
            else:
                label = " ↑ rising"
            print(f"  {xis[i]:>12.4f}  {cvs[i]:>14.8f}  {d_s:>14}  {label}")

        if shoulder_xi is not None:
            print(f"\n  Transient shoulder detected near ξ = {shoulder_xi:.4f}")
        if stop_reason == "converged":
            outcome_str = (f"REACHED   ξ_conv = {xi_converged:.4f}   "
                           f"Cv_conv/kB = {cv_converged:.6f}  "
                           f"(average over {min_stable}-step window)")
        elif stop_reason == "finite_n":
            outcome_str = f"STOPPED — finite-N collapse (monotone fall × {bail_streak})"
        else:
            outcome_str = f"STOPPED — max steps ({max_xi_steps})"
        print(f"\n  → Classical limit: {outcome_str}")
        print(f"{'─'*64}")

    return {
        "xi_values"       : xis,
        "cv_values"       : cvs,
        "deltas"          : deltas,
        "classical_reached": classical_reached,
        "stopped_reason"  : stop_reason,
        "xi_converged"    : xi_converged,   # mean ξ over stable window
        "cv_converged"    : cv_converged,   # mean Cv over stable window
        "shoulder_xi"     : shoulder_xi,
    }


# ================================================================================
#  CHECK 2 — LEVEL CONVERGENCE WITH SHOULDER DETECTION
#  Run AFTER the classical limit check, using xi_converged as the fixed ξ.
# ================================================================================

def check_level_convergence(energies, beta, xi, tol_cv, min_stable=5, verbose=True):
    """
    Test whether Cv has converged with respect to the number of energy levels,
    evaluated at the CONVERGED ξ from the classical limit check.

    Procedure
    ---------
    Start with n = 2 levels and add one at a time up to n = N.
    At each step, record Cv and compute |ΔCv| = |Cv(n) − Cv(n−1)|.

    Shoulder detection
    ------------------
    A 'shoulder' is a run of ≥ min_stable consecutive stable steps
    (|ΔCv| < tol_cv) followed by further significant changes.
    It indicates a temporary false convergence that later resumes —
    e.g. when only the low-lying levels are thermally accessible but
    higher levels begin contributing as more are added.

    True convergence
    ----------------
    A stable run that extends all the way to the last level (n = N).
    'converged_at' is the first n of that final plateau.

    Parameters
    ----------
    energies   : full array of N energy levels
    beta       : natural-unit inverse temperature
    xi         : ξ value to use — should be xi_converged from classical check
    tol_cv     : tolerance on |ΔCv| for stability
    min_stable : minimum consecutive stable steps to count as a plateau
    verbose    : print detailed table if True

    Returns
    -------
    dict with keys:
        n_values, cv_values, deltas,
        converged, converged_at, final_cv, shoulders
    """

    # ── Step 1: compute Cv for every prefix of the energy array ──────────────
    n_values  = list(range(2, len(energies) + 1))
    cv_values = [compute_cv(energies[:n], beta, xi) for n in n_values]

    # ── Step 2: successive differences |ΔCv(n) − Cv(n−1)| ───────────────────
    deltas = [None] + [abs(cv_values[i] - cv_values[i-1])
                       for i in range(1, len(cv_values))]

    # ── Step 3: boolean stability mask ───────────────────────────────────────
    stable = [False] + [(d < tol_cv) for d in deltas[1:]]

    # ── Step 4: find contiguous stable runs of length ≥ min_stable ───────────
    runs = []   # list of (start_index, end_index) inclusive, in n_values space
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

    # ── Step 5: classify runs as shoulders or true convergence ────────────────
    # A run ending at the last index is true convergence; all others are shoulders.
    last_idx     = len(n_values) - 1
    shoulders    = []
    converged_at = None

    for start_idx, end_idx in runs:
        if end_idx == last_idx:
            converged_at = n_values[start_idx]
        else:
            shoulders.append({
                "start_n" : n_values[start_idx],
                "end_n"   : n_values[end_idx],
                "cv_value": cv_values[start_idx],
            })

    # ── Step 6: optional console output ───────────────────────────────────────
    if verbose:
        print(f"\n{'─'*64}")
        print(f"  Level convergence   β={beta:.4f}   ξ={xi:.4f}   "
              f"tol={tol_cv}   window={min_stable}")
        print(f"{'─'*64}")
        print(f"  {'n':>6}  {'Cv/kB':>14}  {'|ΔCv|':>14}  status")
        print(f"  {'─'*6}  {'─'*14}  {'─'*14}  ──────")

        row_label = [""] * len(n_values)
        for sh in shoulders:
            for idx, nv in enumerate(n_values):
                if sh["start_n"] <= nv <= sh["end_n"]:
                    row_label[idx] = "shoulder"
        if converged_at is not None:
            for idx, nv in enumerate(n_values):
                if nv >= converged_at:
                    row_label[idx] = "converged"

        symbols = {"converged": " ✓", "shoulder": " ~", "": ""}
        for i, n in enumerate(n_values):
            d_str = f"{deltas[i]:.8f}" if deltas[i] is not None else "       —        "
            lbl   = row_label[i]
            print(f"  {n:>6}  {cv_values[i]:>14.8f}  {d_str:>14}  "
                  f"{lbl}{symbols[lbl]}")

        if shoulders:
            print(f"\n  Shoulders detected ({len(shoulders)}):")
            for sh in shoulders:
                print(f"    n = {sh['start_n']}–{sh['end_n']}  "
                      f"Cv ≈ {sh['cv_value']:.6f}")
        if converged_at is not None:
            cv_conv = cv_values[n_values.index(converged_at)]
            print(f"\n  → True convergence from n = {converged_at},  "
                  f"Cv/kB = {cv_conv:.8f}")
        else:
            print(f"\n  → NOT converged within {len(energies)} levels")
        print(f"{'─'*64}")

    return {
        "n_values"    : n_values,
        "cv_values"   : cv_values,
        "deltas"      : deltas,
        "converged"   : converged_at is not None,
        "converged_at": converged_at,
        "final_cv"    : cv_values[-1],
        "shoulders"   : shoulders,
    }


# ================================================================================
#  PLOTTING — THREE SEPARATE FIGURES
# ================================================================================

def make_plots(lc, cl, T_K_arr, cv_arr, beta_check,
               cv_classical, tol_cv, tol_xi, n_levels, save_dir):
    """
    Produce three separate matplotlib figures and save them as PNG files.

    The temperature axis is in Kelvin throughout.

    Figure 1 — Cv vs Temperature [K]
        Cv(T) sweep at the CONVERGED ξ (or xi_start if not converged).
        Classical limit shown as a dashed horizontal line.

    Figure 2 — Level convergence
        Cv vs number of levels n, evaluated at the converged ξ.
        Shoulder regions are shaded in purple.

    Figure 3 — ξ sweep (classical limit check)
        Cv vs ξ at fixed β.  Dot colours indicate local behaviour.

    Parameters
    ----------
    lc          : result dict from check_level_convergence
    cl          : result dict from check_classical_limit
    T_K_arr     : temperature array in KELVIN for the Cv(T) sweep
    cv_arr      : corresponding Cv/kB array
    beta_check  : β used for convergence checks (natural units)
    cv_classical: expected classical Cv/kB
    tol_cv, tol_xi : tolerances (for dot colour cutoff in ξ sweep)
    n_levels    : total number of energy levels N
    save_dir    : directory where PNG files are written
    """
    BLUE   = "#1f77b4"
    ORANGE = "#d62728"
    GREEN  = "#2ca02c"
    PURPLE = "#9467bd"
    YELLOW = "#bcbd22"
    GRAY   = "#7f7f7f"

    # Retrieve converged ξ for labels (None if not reached)
    xi_conv = cl["xi_converged"]
    xi_label = f"ξ_conv = {xi_conv:.3f}" if xi_conv is not None else "ξ_start (not converged)"
    T_check_K = T_K_arr[len(T_K_arr)//4]   # rough marker position; actual value below

    
    # ──  Create Layout with Subplot  ────────────────────────────────
    
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), layout="constrained")
    ax1 = axes[0, 0] # Top Left
    ax2 = axes[0, 1] # Top Right
    ax3 = axes[1, 0] # Bottom Left (temporarily)

    # Hide the unused bottom-right spot
    axes[1,1].set_axis_off()

    # Center the bottom plot by spanning it across the middle
    # (This shifts ax3 to start at 25% width instead of 0%)
    # Coordinates are [left, bottom, width, height] in figure fraction units.
    ax3.set_position([0.2, 0.12, 0.5, 0.28]) 


    # Master title for the combined figure
    fig.suptitle(f"Combined Quantum Thermodynamic Analysis (N = {n_levels} levels)", fontsize=16, fontweight='bold')

    # ── Figure 1: Cv vs Temperature in Kelvin ────────────────────────────────
    ax1.set_title(
        f"Cv / kB  vs  Temperature   ({xi_label},  N = {n_levels} levels)",
        fontsize=12,
    )

    ax1.plot(T_K_arr, cv_arr, color=BLUE, linewidth=2, label=f"Cv(T)  [{xi_label}]")

    # Classical limit as a horizontal dashed line
    ax1.axhline(cv_classical, color=ORANGE, linewidth=1.5, linestyle="--",
                label=f"Classical limit  Cv/kB = {cv_classical}")

    # Mark the temperature used for the convergence checks.
    # T_K_arr was built as E0/(KB_SI * beta_arr) with beta_arr linearly spaced
    # between beta_min and beta_max.  The check was done at beta_check, so we
    # find the index in beta_arr closest to beta_check, then take that T value.
    # beta_arr is proportional to 1/T_K_arr, so: beta_arr ~ 1/T_K_arr.
    # We reconstruct the closest index without storing beta_arr by using
    # the fact that beta is proportional to (1/T):
    inv_T = 1.0 / T_K_arr                           # proportional to beta_arr
    inv_T_check = beta_check / (1.0 / T_K_arr).max() * inv_T.max()
    idx_check = int(np.argmin(np.abs(inv_T - inv_T.max() * beta_check / inv_T.max())))
    T_check_K = T_K_arr[idx_check]
    ax1.axvline(T_check_K, color=GREEN, linewidth=1, linestyle=":",
                label=f"T_check = {T_check_K:.0f} K  (beta={beta_check})")

    ax1.set_xlabel("Temperature  T  [K]", fontsize=11)
    ax1.set_ylabel("Cv / kB", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # ── Figure 2: Level convergence ───────────────────────────────────────────
    xi_used_str = f"{xi_conv:.4f}" if xi_conv is not None else "xi_start (fallback)"
    ax2.set_title(
        f"Level Convergence   (beta = {beta_check},  xi = {xi_used_str} [converged])",
        fontsize=12,
    )

    ns    = lc["n_values"]
    cvs_l = lc["cv_values"]
    ax2.plot(ns, cvs_l, color=BLUE, linewidth=1.5, marker="o",
             markersize=3, label="Cv(n levels)")
    ax2.axhline(cv_classical, color=ORANGE, linewidth=1.2, linestyle="--",
                label=f"Classical {cv_classical}", alpha=0.8)

    # Shade every shoulder region
    for sh in lc["shoulders"]:
        ax2.axvspan(sh["start_n"], sh["end_n"], color=PURPLE, alpha=0.18)
        mid_n = (sh["start_n"] + sh["end_n"]) / 2
        ax2.annotate(
            f"shoulder\nCv ≈ {sh['cv_value']:.4f}",
            xy=(mid_n, sh["cv_value"]),
            fontsize=7, color=PURPLE, ha="center", va="bottom",
        )

    # Mark the onset of true convergence
    if lc["converged"]:
        idx = ns.index(lc["converged_at"])
        ax2.axvline(lc["converged_at"], color=GREEN, linestyle=":",
                    linewidth=1.3, label=f"Converged at n = {lc['converged_at']}")
        ax2.scatter([lc["converged_at"]], [cvs_l[idx]], color=GREEN, zorder=5, s=70)
    else:
        ax2.text(0.97, 0.05, "Not converged within N levels",
                 transform=ax2.transAxes, ha="right", va="bottom",
                 fontsize=8, color=ORANGE)

    handles, labels = ax2.get_legend_handles_labels()
    if lc["shoulders"]:
        handles.append(mpatches.Patch(color=PURPLE, alpha=0.35, label="shoulder region"))
        labels.append("shoulder region")
    ax2.legend(handles, labels, fontsize=9)
    ax2.set_xlabel("Number of energy levels  n", fontsize=11)
    ax2.set_ylabel("Cv / kB", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    
    # ── Figure 3: ξ sweep ─────────────────────────────────────────────────────
    ax3.set_title(
        f"Classical Limit — ξ Sweep   (β = {beta_check},  N = {n_levels} levels)",
        fontsize=12,
    )

    xis_cl = cl["xi_values"]
    cvs_cl = cl["cv_values"]
    ax3.plot(xis_cl, cvs_cl, color=BLUE, linewidth=1.5,
             marker="s", markersize=5, label="Cv(ξ)")
    ax3.axhline(cv_classical, color=ORANGE, linewidth=1.5, linestyle="--",
                label=f"Classical limit {cv_classical}")

    # If converged, shade the stable window and mark the mean ξ
    if cl["classical_reached"]:
        stable_xis = xis_cl[-len([d for d in cl["deltas"][1:] if d is not None and d < tol_xi]):]
        # Simpler: shade from xi_converged back across min_stable points
        ax3.axvline(cl["xi_converged"], color=GREEN, linewidth=1.2, linestyle=":",
                    label=f"ξ_converged = {cl['xi_converged']:.3f}")
        ax3.scatter([cl["xi_converged"]], [cl["cv_converged"]],
                    color=GREEN, zorder=6, s=80,
                    label=f"Cv_converged = {cl['cv_converged']:.4f}")

    # Colour individual dots by local behaviour
    for i in range(len(xis_cl)):
        d = cl["deltas"][i]
        if d is not None and d < tol_xi:
            colour = GREEN    # within the stable window
        elif i > 0 and cvs_cl[i] < cvs_cl[i-1]:
            colour = YELLOW   # Cv is falling
        else:
            colour = GRAY     # Cv is rising or first point
        ax3.scatter([xis_cl[i]], [cvs_cl[i]], color=colour, zorder=5, s=55)

    # Mark any transient shoulder
    if cl["shoulder_xi"] is not None:
        ax3.axvline(cl["shoulder_xi"], color=PURPLE, linewidth=1, linestyle=":",
                    label=f"Transient plateau  ξ ≈ {cl['shoulder_xi']:.2f}")

    # Stop-reason annotation box — built conditionally to avoid formatting None
    stop = cl["stopped_reason"]
    if stop == "converged":
        ann_text = (f"Classical limit reached ✓\n"
                    f"xi_conv = {cl['xi_converged']:.3f}   "
                    f"Cv_conv = {cl['cv_converged']:.5f}")
    elif stop == "finite_n":
        ann_text = "Stopped: finite-N collapse\n(increase N to go further)"
    else:
        ann_text = "Stopped: max xi steps reached"
    reason_colour = {"converged": GREEN, "finite_n": YELLOW, "max_steps": GRAY}
    ax3.text(0.97, 0.06, ann_text, transform=ax3.transAxes,
             ha="right", va="bottom", fontsize=8, color=reason_colour[stop],
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor=reason_colour[stop], alpha=0.85))

    # Legend: line + dot colour explanation
    dot_handles = [
        mpatches.Patch(color=GREEN,  label="stable  |ΔCv| < tol"),
        mpatches.Patch(color=YELLOW, label="falling Cv"),
        mpatches.Patch(color=GRAY,   label="rising Cv"),
    ]
    h3, l3 = ax3.get_legend_handles_labels()
    ax3.legend(handles=h3 + dot_handles, fontsize=9, bbox_to_anchor=(1.02, 1.0), loc="upper left")
    ax3.set_xlabel("Scaling factor  ξ", fontsize=11)
    ax3.set_ylabel("Cv / kB", fontsize=11)
    ax3.grid(True, linestyle="--", alpha=0.5)
    
    # ── Save Combined Figure ─────────────────────────────────────────────
    plt.show()
    
    combined_path = os.path.join(save_dir, "quantum_thermo_analysis.png")
    fig.savefig(combined_path, dpi=150)
    plt.close(fig)
    print(f"  Saved Combined Layout → {combined_path}")

# ================================================================================
#  FULL PIPELINE
# ================================================================================

def run(
    energies,
    E0_J,
    beta_check, xi_start,
    cv_classical  = 0.5,
    tol_cv        = 1e-3,
    tol_xi        = 5e-3,
    min_stable_lc = 5,
    min_stable_cl = 5,
    bail_streak   = 8,
    xi_multiplier = 1.5,
    max_xi_steps  = 80,
    beta_min      = 0.02,
    beta_max      = 5.0,
    n_beta        = 400,
    save_dir      = None,
):
    """
    Full pipeline: classical limit → level convergence → Cv(T) sweep → plots.

    Order of operations
    -------------------
    1. check_classical_limit  — find xi_converged (mean ξ over stable window)
                                and cv_converged (mean Cv over stable window).
    2. check_level_convergence — evaluated at xi_converged (not a user constant).
    3. Cv(T) sweep             — also uses xi_converged as the fixed ξ.
    4. make_plots              — temperature axis in Kelvin.

    If the classical limit is NOT reached, steps 2–4 fall back to xi_start
    and a warning is printed.

    Parameters
    ----------
    energies      : 1-D array of energy level values (units of E₀)
    E0_J          : physical energy scale E₀ in Joules.
                    Converts natural temperature to Kelvin: T[K] = E₀/(kB·β_nat).
                    Example for a 1D box of length L with particle mass m:
                        E0_J = (hbar·π)² / (2·m·L²)
    beta_check    : natural-unit β = E₀/(kB·T) for the convergence checks
    xi_start      : starting value of ξ for the classical limit sweep
    cv_classical  : expected classical Cv/kB  (annotation only)
    tol_cv        : |ΔCv| tolerance for level convergence
    tol_xi        : |ΔCv| tolerance for classical limit / ξ sweep
    min_stable_lc : window size for level convergence plateau detection
    min_stable_cl : window size for classical limit convergence
    bail_streak   : consecutive monotone-fall steps → finite-N bail-out
    xi_multiplier : ξ *= xi_multiplier at each step
    max_xi_steps  : hard cap on ξ iterations
    beta_min/max  : range of natural-unit β for the Cv(T) sweep
    n_beta        : number of β points in the sweep
    save_dir      : where to save PNG files (defaults to script directory)
    """
    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(__file__))

    # ── Step 1: Classical limit check ────────────────────────────────────────
    # Run this first so we know xi_converged before anything else.
    cl = check_classical_limit(
        energies, beta_check, xi_start, cv_classical,
        xi_multiplier, tol_xi, min_stable_cl, bail_streak, max_xi_steps,
    )

    # ── Determine which ξ to use for the remaining steps ─────────────────────
    # If the classical limit was reached, use the converged ξ (average over
    # the stable window).  If not, fall back to xi_start and warn the user.
    if cl["classical_reached"]:
        xi_for_lc  = cl["xi_converged"]   # mean ξ over stable window
        print(f"\n  Using converged ξ = {xi_for_lc:.4f} for level convergence and Cv(T) sweep.")
    else:
        xi_for_lc  = xi_start
        print(f"\n  WARNING: classical limit not reached.  "
              f"Falling back to xi_start = {xi_start} for remaining steps.")

    # ── Step 2: Level convergence at xi_converged ─────────────────────────────
    lc = check_level_convergence(
        energies, beta_check, xi_for_lc, tol_cv, min_stable_lc
    )

    # ── Step 3: Cv(T) sweep at xi_converged ──────────────────────────────────
    # β_nat = E₀/(kB·T[K])  →  T[K] = E₀/(kB · β_nat)
    # Sweep β from beta_min to beta_max; convert each β to T in Kelvin.
    beta_arr = np.linspace(beta_min, beta_max, n_beta)
    T_K_arr  = E0_J / (KB_SI * beta_arr)         # temperature in Kelvin
    cv_arr   = np.array([compute_cv(energies, b, xi_for_lc) for b in beta_arr])

    # ── Step 4: Plots ──────────────────────────────────────────────────────────
    print("\n  Saving figures...")
    make_plots(lc, cl, T_K_arr, cv_arr, beta_check,
               cv_classical, tol_cv, tol_xi, len(energies), save_dir)

    return lc, cl, T_K_arr, cv_arr


# ================================================================================
#  ENTRY POINT
# ================================================================================

if __name__ == "__main__":

    # ── Energy levels ─────────────────────────────────────────────────────────
    # Particle-in-a-box:  En = n² · E₀  (dimensionless: n = 1, 2, 3, …)
    # Swap this array for any discrete spectrum (harmonic oscillator, etc.)
    N_LEVELS = 200
    energies = np.array([n**2 for n in range(1, N_LEVELS + 1)], dtype=float)

    # ── Physical energy scale ─────────────────────────────────────────────────
    # E₀ sets the conversion from natural units to Kelvin: T[K] = E₀/(kB·β_nat).
    # Example: electron in a 1 nm box.
    #   E0_J = (1.055e-34 * π)² / (2 * 9.109e-31 * (1e-9)²) ≈ 6.02e-20 J ≈ 0.376 eV
    # Adjust this to match your physical system.
    E0_J = 6.02e-20   # [J]  — energy scale of the system

    # ── Convergence check parameters ─────────────────────────────────────────
    # β is in natural units: β_nat = E₀ / (kB · T[K])
    # β_nat = 1 corresponds to T = E₀/kB ≈ 4360 K for E0_J = 6.02e-20 J
    BETA_CHECK    = 0.05    # β_nat for both convergence checks
                            # → T_check ≈ E₀/(kB·0.05) = 20 · E₀/kB

    # Starting ξ for the classical limit sweep (will be increased multiplicatively)
    XI_START      = 3.0

    # Expected classical Cv/kB for a 1D box (one quadratic degree of freedom)
    CV_CLASSICAL  = 0.5

    # ── Tolerances ────────────────────────────────────────────────────────────
    TOL_CV        = 1e-3   # level convergence: |ΔCv| < tol_cv
    TOL_XI        = 5e-3   # classical limit:   |ΔCv| < tol_xi

    # ── Plateau window and bail-out ───────────────────────────────────────────
    MIN_STABLE_LC = 5      # min consecutive stable steps → shoulder or convergence
    MIN_STABLE_CL = 5      # min consecutive stable steps → classical convergence
    BAIL_STREAK   = 8      # monotone-fall steps before finite-N bail-out

    # ── ξ sweep settings ──────────────────────────────────────────────────────
    XI_MULT       = 1.3    # ξ is multiplied by this factor at each step
    MAX_XI_STEPS  = 80     # hard cap on number of ξ values tried

    # ── Cv(T) sweep settings ─────────────────────────────────────────────────
    # β range in natural units; T[K] = E₀/(kB · β_nat)
    BETA_MIN      = 0.02   # → T_max ≈ 50 · E₀/kB
    BETA_MAX      = 5.0    # → T_min ≈ 0.2 · E₀/kB
    N_BETA        = 400    # number of temperature points in the sweep

    # ── Print header ──────────────────────────────────────────────────────────
    T_check_K = E0_J / (KB_SI * BETA_CHECK)
    T_min_K   = E0_J / (KB_SI * BETA_MAX)
    T_max_K   = E0_J / (KB_SI * BETA_MIN)
    print("\n" + "═"*64)
    print("  Quantum Cv — Classical Limit Calculator")
    print("═"*64)
    print(f"  Energy levels : En = n²,  n = 1 … {N_LEVELS}")
    print(f"  E₀            : {E0_J:.3e} J  ({E0_J/1.602e-19*1000:.2f} meV)")
    print(f"  β_check       : {BETA_CHECK}  →  T_check = {T_check_K:.1f} K")
    print(f"  T sweep       : {T_min_K:.1f} K — {T_max_K:.1f} K")
    print(f"  ξ_start       : {XI_START}")
    print(f"  Classical Cv  : {CV_CLASSICAL} kB")
    print(f"  tol_cv = {TOL_CV},  tol_xi = {TOL_XI},  "
          f"window = {MIN_STABLE_LC},  bail = {BAIL_STREAK}")

    lc, cl, T_K_arr, cv_arr = run(
        energies      = energies,
        E0_J          = E0_J,
        beta_check    = BETA_CHECK,
        xi_start      = XI_START,
        cv_classical  = CV_CLASSICAL,
        tol_cv        = TOL_CV,
        tol_xi        = TOL_XI,
        min_stable_lc = MIN_STABLE_LC,
        min_stable_cl = MIN_STABLE_CL,
        bail_streak   = BAIL_STREAK,
        xi_multiplier = XI_MULT,
        max_xi_steps  = MAX_XI_STEPS,
        beta_min      = BETA_MIN,
        beta_max      = BETA_MAX,
        n_beta        = N_BETA,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "═"*64)
    print("  Summary")
    print("═"*64)

    if cl["classical_reached"]:
        print(f"  Classical limit   : ✓  ξ_conv = {cl['xi_converged']:.4f}   "
              f"Cv_conv/kB = {cl['cv_converged']:.6f}")
    else:
        print(f"  Classical limit   : ✗  ({cl['stopped_reason']})")

    if cl["shoulder_xi"] is not None:
        print(f"    transient plateau near ξ = {cl['shoulder_xi']:.4f}")

    if lc["converged"]:
        cv_at = lc["cv_values"][lc["n_values"].index(lc["converged_at"])]
        print(f"  Level convergence : ✓  from n = {lc['converged_at']},  "
              f"Cv/kB = {cv_at:.6f}")
    else:
        print(f"  Level convergence : ✗  not converged within {N_LEVELS} levels")

    if lc["shoulders"]:
        for sh in lc["shoulders"]:
            print(f"    shoulder at n = {sh['start_n']}–{sh['end_n']}  "
                  f"Cv ≈ {sh['cv_value']:.6f}")

    print("═"*64 + "\n")