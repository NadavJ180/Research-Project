"""
Quantum Heat Capacity — Classical Limit Checker
================================================
Formula (from image):

    Cv/kB = β² · [ <(En/ξ²)²>_β  −  (<En/ξ²>_β)² ]
          = (β²/ξ⁴) · [ <En²>_β  −  <En>_β² ]

where:
    β  = 1/(kB·T)          — inverse temperature (independent of ξ)
    ξ  — scaling factor in the energy denominator (independent parameter)
    <·>_β = Σ (·) e^{−β·En/ξ²} / Z,   Z = Σ e^{−β·En/ξ²}

Two convergence checks:
  1. Level convergence  : at fixed (β, ξ), add levels one by one.
       — Shoulder-aware: uses a sliding window of width `min_stable`.
         A flat region that later resumes changing is flagged as a shoulder,
         not true convergence. All shoulder locations are reported.
       — Converged only if the plateau is sustained to the final level.

  2. Classical limit    : fixed β, increase ξ multiplicatively.
       — Decrease-aware: Cv is allowed to fall and re-stabilise.
         Only the RATE OF CHANGE matters, not whether Cv went up or down.
         A sustained window of |ΔCv| < tol_xi signals convergence.
       — Finite-N bail-out: if Cv is STILL declining with no sign of
         stabilising (monotone drop for `bail_streak` consecutive steps),
         we stop to avoid wasting steps on a clear artefact.
       — This correctly handles both Cv → limit from below AND from above.

Classical limit for 1D box:  Cv_classical/kB = 0.5
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import warnings
import os


# ──────────────────────────────────────────────────────────────
#  Core computation
# ──────────────────────────────────────────────────────────────

def compute_cv(energies: np.ndarray, beta: float, xi: float) -> float:
    """
    Cv/kB = (β²/ξ⁴) · [ <En²>_β − <En>_β² ]

    Boltzmann weights: w_n = exp(−β·En/ξ²)
    Numerical stability: subtract max exponent before exp.
    """
    energies  = np.asarray(energies, dtype=float)
    exponents = beta * energies / (xi ** 2)
    shift     = np.max(exponents)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        weights = np.exp(-(exponents - shift))
    weights = np.where(np.isfinite(weights), weights, 0.0)
    Z         = weights.sum()
    if Z == 0 or not np.isfinite(Z):
        return np.nan
    avg_E  = np.dot(weights, energies) / Z
    avg_E2 = np.dot(weights, energies ** 2) / Z
    return float((beta ** 2 / xi ** 4) * (avg_E2 - avg_E ** 2))


# ──────────────────────────────────────────────────────────────
#  Check 1 — level convergence with shoulder detection
# ──────────────────────────────────────────────────────────────

def check_level_convergence(
    energies   : np.ndarray,
    beta       : float,
    xi         : float,
    tol_cv     : float,
    min_stable : int  = 5,    # consecutive steps below tol to count as a plateau
    verbose    : bool = True,
) -> dict:
    """
    Add energy levels one by one and record Cv at each step.

    Shoulder detection
    ------------------
    A 'shoulder' is a plateau of ≥ min_stable steps where |ΔCv| < tol_cv,
    followed by a resumption of significant changes.  Shoulders are stored
    in `shoulders` as a list of dicts {start_n, end_n, cv_value}.

    True convergence
    ----------------
    Declared only when |ΔCv| < tol_cv for ALL remaining levels up to N.
    The first n at which this permanent stability begins is `converged_at`.

    Returns
    -------
    n_values, cv_values, deltas, converged, converged_at, final_cv, shoulders
    """
    # ── Compute all Cv values first ───────────────────────────
    n_values, cv_values, deltas = [2], [], [None]
    cv_values.append(compute_cv(energies[:2], beta, xi))

    for n in range(3, len(energies) + 1):
        cv = compute_cv(energies[:n], beta, xi)
        n_values.append(n)
        cv_values.append(cv)
        deltas.append(abs(cv - cv_values[-2]))

    # ── Classify each step: stable (S) or active (A) ─────────
    stable = [False] + [
        (d is not None and d < tol_cv) for d in deltas[1:]
    ]  # stable[i] = True if deltas[i] < tol

    # ── Identify contiguous stable runs ───────────────────────
    runs = []   # list of (start_idx, end_idx) in n_values index space
    i = 0
    while i < len(stable):
        if stable[i]:
            j = i
            while j < len(stable) and stable[j]:
                j += 1
            if j - i >= min_stable:
                runs.append((i, j - 1))   # inclusive
            i = j
        else:
            i += 1

    # ── Separate shoulders from true convergence ───────────────
    # True convergence: the run extends to the very last index.
    shoulders    = []
    converged_at = None

    for start_idx, end_idx in runs:
        is_final = (end_idx == len(n_values) - 1)
        if is_final:
            converged_at = n_values[start_idx]
        else:
            # It's a shoulder: plateau that later resumes
            shoulders.append(dict(
                start_n  = n_values[start_idx],
                end_n    = n_values[end_idx],
                cv_value = cv_values[start_idx],
            ))

    # ── Verbose output ─────────────────────────────────────────
    if verbose:
        print(f"\n{'─'*64}")
        print(f"  Level convergence   β={beta:.4f}   ξ={xi:.4f}   tol={tol_cv}"
              f"   window={min_stable}")
        print(f"{'─'*64}")
        print(f"  {'n':>5}  {'Cv/kB':>14}  {'|ΔCv|':>14}  label")
        print(f"  {'─'*5}  {'─'*14}  {'─'*14}  ─────")
        # Build per-row labels
        row_label = [""] * len(n_values)
        for sh in shoulders:
            for idx, nv in enumerate(n_values):
                if sh["start_n"] <= nv <= sh["end_n"]:
                    row_label[idx] = "shoulder"
        if converged_at is not None:
            for idx, nv in enumerate(n_values):
                if nv >= converged_at:
                    row_label[idx] = "converged"

        for i, n in enumerate(n_values):
            d_str = f"{deltas[i]:.8f}" if deltas[i] is not None else "       —        "
            lbl   = row_label[i]
            sym   = {"converged": " ✓", "shoulder": " ~", "":" "}[lbl]
            print(f"  {n:>5}  {cv_values[i]:>14.8f}  {d_str:>14}  {lbl}{sym}")

        if shoulders:
            print(f"\n  Shoulders detected ({len(shoulders)}):")
            for sh in shoulders:
                print(f"    n={sh['start_n']}–{sh['end_n']}  Cv≈{sh['cv_value']:.6f}")
        if converged_at is not None:
            print(f"\n  → True convergence from n={converged_at}"
                  f"  Cv/kB={cv_values[n_values.index(converged_at)]:.8f}")
        else:
            print(f"\n  → NOT converged (no sustained plateau to end of levels)")
        print(f"{'─'*64}")

    return dict(
        n_values     = n_values,
        cv_values    = cv_values,
        deltas       = deltas,
        converged    = converged_at is not None,
        converged_at = converged_at,
        final_cv     = cv_values[-1],
        shoulders    = shoulders,
    )


# ──────────────────────────────────────────────────────────────
#  Check 2 — classical limit, decrease-aware
# ──────────────────────────────────────────────────────────────

def check_classical_limit(
    energies      : np.ndarray,
    beta          : float,
    xi_start      : float,
    cv_classical  : float,
    xi_multiplier : float = 1.5,
    tol_xi        : float = 0.005,
    min_stable    : int   = 5,   # consecutive |ΔCv|<tol steps to declare convergence
    bail_streak   : int   = 8,   # consecutive monotone-decreasing steps → finite-N bail
    max_xi_steps  : int   = 80,
    verbose       : bool  = True,
) -> dict:
    """
    Sweep ξ from xi_start multiplicatively and check whether Cv stabilises.

    Decrease-aware logic
    --------------------
    We do NOT use a peak guard.  Instead we track the RATE OF CHANGE:

    • Convergence (any direction): |ΔCv| < tol_xi for min_stable consecutive
      steps.  Works whether Cv approaches the limit from above or below.

    • Finite-N bail-out: Cv has been *strictly* monotone-decreasing for
      bail_streak steps AND the cumulative drop over those steps exceeds
      tol_xi.  This identifies a runaway collapse, not just overshoot.

    Returns
    -------
    xi_values, cv_values, deltas, classical_reached, stopped_reason,
    converged_cv, shoulder_xi (ξ where a transient flat region was seen, if any)
    """
    xis, cvs = [], []
    xi           = xi_start
    stable_count = 0
    stopped_reason = "max_steps"

    for step in range(max_xi_steps):
        cv = compute_cv(energies, beta, xi)
        xis.append(xi)
        cvs.append(cv)

        if step >= 1:
            delta = abs(cv - cvs[-2])

            # ── Convergence check (direction-agnostic) ──────
            if delta < tol_xi:
                stable_count += 1
                if stable_count >= min_stable:
                    stopped_reason = "converged"
                    break
            else:
                stable_count = 0

            # ── Finite-N bail-out: sustained monotone fall ──
            if step >= bail_streak:
                window = cvs[-(bail_streak + 1):]
                diffs  = [window[k+1] - window[k] for k in range(bail_streak)]
                all_falling = all(d < 0 for d in diffs)
                total_drop  = window[0] - window[-1]
                if all_falling and total_drop > tol_xi:
                    stopped_reason = "finite_n"
                    break

        xi *= xi_multiplier

    deltas = [None] + [abs(cvs[i] - cvs[i-1]) for i in range(1, len(cvs))]
    classical_reached = (stopped_reason == "converged")
    converged_cv      = cvs[-1] if classical_reached else None

    # Detect any transient shoulder in the ξ sweep
    # (a flat run that did not persist to the end)
    stable_flags = [False] + [
        (d is not None and d < tol_xi) for d in deltas[1:]
    ]
    shoulder_xi = None
    i = 0
    while i < len(stable_flags):
        if stable_flags[i]:
            j = i
            while j < len(stable_flags) and stable_flags[j]:
                j += 1
            if j - i >= 2 and j < len(stable_flags):
                shoulder_xi = xis[i]   # earliest transient plateau
                break
            i = j
        else:
            i += 1

    if verbose:
        print(f"\n{'─'*64}")
        print(f"  Classical limit   β={beta:.4f}   tol={tol_xi}"
              f"   window={min_stable}   bail={bail_streak}")
        print(f"  Expected Cv_classical/kB = {cv_classical}")
        print(f"{'─'*64}")
        print(f"  {'ξ':>12}  {'Cv/kB':>14}  {'|ΔCv|':>14}  label")
        print(f"  {'─'*12}  {'─'*14}  {'─'*14}  ─────")
        for i in range(len(xis)):
            d   = deltas[i]
            d_s = f"{d:.8f}" if d is not None else "       —        "
            if d is not None and d < tol_xi:
                lbl = " ✓ stable"
            elif i > 0 and cvs[i] < cvs[i-1]:
                lbl = " ↓ falling"
            else:
                lbl = " ↑ rising"
            print(f"  {xis[i]:>12.4f}  {cvs[i]:>14.8f}  {d_s:>14}  {lbl}")
        outcomes = {
            "converged": f"REACHED  Cv/kB = {cvs[-1]:.6f}",
            "finite_n" : f"STOPPED — finite-N collapse (monotone fall for {bail_streak} steps)",
            "max_steps": f"STOPPED — max steps ({max_xi_steps})",
        }
        if shoulder_xi is not None:
            print(f"\n  Transient shoulder in ξ sweep detected near ξ={shoulder_xi:.4f}")
        print(f"\n  → Classical limit: {outcomes[stopped_reason]}")
        print(f"{'─'*64}")

    return dict(
        xi_values       = xis,
        cv_values       = cvs,
        deltas          = deltas,
        classical_reached = classical_reached,
        stopped_reason  = stopped_reason,
        converged_cv    = converged_cv,
        shoulder_xi     = shoulder_xi,
    )


# ──────────────────────────────────────────────────────────────
#  Full pipeline: sweep β and plot
# ──────────────────────────────────────────────────────────────

def sweep_and_plot(
    energies      : np.ndarray,
    beta_check    : float,
    xi_check      : float,
    cv_classical  : float  = 0.5,
    tol_cv        : float  = 1e-3,
    tol_xi        : float  = 5e-3,
    min_stable_lc : int    = 5,
    min_stable_cl : int    = 5,
    bail_streak   : int    = 8,
    xi_multiplier : float  = 1.5,
    max_xi_steps  : int    = 80,
    beta_min      : float  = 0.02,
    beta_max      : float  = 5.0,
    n_beta        : int    = 400,
    save_path     : str | None = None,
):
    # ── 1. Level convergence (shoulder-aware) ─────────────────
    lc = check_level_convergence(
        energies, beta_check, xi_check, tol_cv, min_stable_lc
    )

    # ── 2. Classical limit (decrease-aware) ───────────────────
    cl = check_classical_limit(
        energies, beta_check, xi_check, cv_classical,
        xi_multiplier, tol_xi, min_stable_cl, bail_streak, max_xi_steps,
    )

    # ── 3. Cv vs T sweep (ξ fixed at xi_check) ────────────────
    beta_arr = np.linspace(beta_min, beta_max, n_beta)
    cv_arr   = np.array([compute_cv(energies, b, xi_check) for b in beta_arr])
    T_arr    = 1.0 / beta_arr

    # ── 4. Figure ─────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 11))
    fig.patch.set_facecolor("#0f1117")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.48, wspace=0.36)
    ax_main = fig.add_subplot(gs[0, :])
    ax_lev  = fig.add_subplot(gs[1, 0])
    ax_xi   = fig.add_subplot(gs[1, 1])

    PANEL_BG = "#161b22"
    GRID_CLR = "#2d333b"
    TEXT_CLR = "#c9d1d9"
    BLUE     = "#58a6ff"
    ORANGE   = "#f78166"
    GREEN    = "#3fb950"
    YELLOW   = "#e3b341"
    PURPLE   = "#bc8cff"
    MUTED    = "#8b949e"

    def style_ax(ax, title):
        ax.set_facecolor(PANEL_BG)
        for sp in ax.spines.values():
            sp.set_edgecolor(GRID_CLR)
        ax.tick_params(colors=TEXT_CLR, labelsize=9)
        ax.xaxis.label.set_color(TEXT_CLR)
        ax.yaxis.label.set_color(TEXT_CLR)
        ax.set_title(title, color=TEXT_CLR, fontsize=11, pad=8)
        ax.grid(True, color=GRID_CLR, linewidth=0.5, linestyle="--", alpha=0.6)

    # ── Main: Cv vs T ─────────────────────────────────────────
    style_ax(ax_main, f"Cᵥ / kB  vs  temperature   (ξ = {xi_check})")
    ax_main.plot(T_arr, cv_arr, color=BLUE, linewidth=2, label="Cᵥ(T, ξ)")
    ax_main.axhline(cv_classical, color=ORANGE, linewidth=1.5, linestyle="--",
                    label=f"Classical limit  Cᵥ/kB = {cv_classical}")
    T_check     = 1.0 / beta_check
    cv_at_check = compute_cv(energies, beta_check, xi_check)
    ax_main.axvline(T_check, color=GREEN, linewidth=1, linestyle=":",
                    label=f"T_check = {T_check:.1f}  (β={beta_check})")
    ax_main.scatter([T_check], [cv_at_check], color=GREEN, zorder=5, s=60)
    ax_main.set_xlabel("T  (units of E₀/kB)    [T = 1/β]", fontsize=10)
    ax_main.set_ylabel("Cᵥ / kB", fontsize=10)
    ax_main.legend(fontsize=9, framealpha=0.3, labelcolor=TEXT_CLR,
                   facecolor=PANEL_BG, edgecolor=GRID_CLR)
    ax_main.set_xlim(T_arr[0], T_arr[-1])

    cl_col  = GREEN if cl["classical_reached"] else ORANGE
    cl_text = (f"Classical limit ✓  Cv/kB = {cl['converged_cv']:.4f}"
               if cl["classical_reached"]
               else f"Classical limit not reached  (stop: {cl['stopped_reason']})")
    ax_main.text(0.98, 0.08, cl_text, transform=ax_main.transAxes,
                 ha="right", va="bottom", fontsize=9, color=cl_col,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor=PANEL_BG,
                           edgecolor=cl_col, alpha=0.85))

    # ── Level convergence plot ─────────────────────────────────
    style_ax(ax_lev, f"Level convergence  (β={beta_check}, ξ={xi_check})")
    ns    = lc["n_values"]
    cvs_l = lc["cv_values"]
    ax_lev.plot(ns, cvs_l, color=BLUE, linewidth=1.5, marker="o",
                markersize=3.5, label="Cᵥ vs n", zorder=2)
    ax_lev.axhline(cv_classical, color=ORANGE, linewidth=1, linestyle="--",
                   label=f"Classical {cv_classical}", alpha=0.7)

    # Shade shoulder regions in purple
    for sh in lc["shoulders"]:
        ax_lev.axvspan(sh["start_n"], sh["end_n"], color=PURPLE,
                       alpha=0.18, zorder=1)
        ax_lev.annotate(
            f"shoulder\n~{sh['cv_value']:.4f}",
            xy=((sh["start_n"] + sh["end_n"]) / 2, sh["cv_value"]),
            fontsize=7, color=PURPLE, ha="center", va="bottom",
        )

    # Mark true convergence onset
    if lc["converged"]:
        idx = ns.index(lc["converged_at"])
        ax_lev.axvline(lc["converged_at"], color=GREEN, linestyle=":",
                       linewidth=1.2, label=f"Converged n={lc['converged_at']}")
        ax_lev.scatter([lc["converged_at"]], [cvs_l[idx]],
                       color=GREEN, zorder=5, s=60)
    else:
        ax_lev.text(0.97, 0.06, "Not converged", transform=ax_lev.transAxes,
                    ha="right", va="bottom", fontsize=8, color=ORANGE)

    # Legend with shoulder patch
    handles, labels = ax_lev.get_legend_handles_labels()
    if lc["shoulders"]:
        handles.append(mpatches.Patch(color=PURPLE, alpha=0.4, label="shoulder region"))
        labels.append("shoulder region")
    ax_lev.legend(handles, labels, fontsize=8, framealpha=0.3,
                  labelcolor=TEXT_CLR, facecolor=PANEL_BG, edgecolor=GRID_CLR)
    ax_lev.set_xlabel("Number of levels n", fontsize=9)
    ax_lev.set_ylabel("Cᵥ / kB", fontsize=9)

    # ── ξ sweep plot ───────────────────────────────────────────
    style_ax(ax_xi, f"ξ sweep  (β={beta_check} fixed, N={len(energies)} levels)")
    xis_cl = cl["xi_values"]
    cvs_cl = cl["cv_values"]
    ax_xi.plot(xis_cl, cvs_cl, color=BLUE, linewidth=1.5,
               marker="s", markersize=5, label="Cᵥ vs ξ")
    ax_xi.axhline(cv_classical, color=ORANGE, linewidth=1.5, linestyle="--",
                  label=f"Classical {cv_classical}")

    # Colour dots by local behaviour
    for i, (xv, cv, d) in enumerate(zip(xis_cl, cvs_cl, cl["deltas"])):
        if d is not None and d < tol_xi:
            clr = GREEN      # stably converging
        elif i > 0 and cvs_cl[i] < cvs_cl[i-1]:
            clr = YELLOW     # falling
        else:
            clr = MUTED      # rising
        ax_xi.scatter([xv], [cv], color=clr, zorder=5, s=50)

    # Annotate transient shoulder in ξ sweep
    if cl["shoulder_xi"] is not None:
        ax_xi.axvline(cl["shoulder_xi"], color=PURPLE, linewidth=1,
                      linestyle=":", label=f"transient plateau ξ≈{cl['shoulder_xi']:.2f}")

    reason_style = {
        "converged": ("Classical limit reached ✓", GREEN),
        "finite_n" : ("Stopped: finite-N collapse ↓", YELLOW),
        "max_steps": ("Stopped: max steps", MUTED),
    }
    rlbl, rcol = reason_style[cl["stopped_reason"]]
    ax_xi.text(0.97, 0.07, rlbl, transform=ax_xi.transAxes,
               ha="right", va="bottom", fontsize=8, color=rcol,
               bbox=dict(boxstyle="round,pad=0.25", facecolor=PANEL_BG,
                         edgecolor=rcol, alpha=0.85))

    # Custom legend with dot colours
    handles2 = [
        mpatches.Patch(color=GREEN,  label="stable |ΔCv| < tol"),
        mpatches.Patch(color=YELLOW, label="falling Cv"),
        mpatches.Patch(color=MUTED,  label="rising Cv"),
    ]
    ax_xi.legend(handles=handles2, fontsize=8, framealpha=0.3,
                 labelcolor=TEXT_CLR, facecolor=PANEL_BG, edgecolor=GRID_CLR)
    ax_xi.set_xlabel("ξ  (scaling factor)", fontsize=9)
    ax_xi.set_ylabel("Cᵥ / kB", fontsize=9)

    fig.suptitle(
        f"Quantum heat capacity — {len(energies)} particle-in-a-box levels",
        color=TEXT_CLR, fontsize=13, y=0.985,
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"\n  Figure saved → {save_path}")
    else:
        plt.show()

    plt.close(fig)
    return lc, cl, T_arr, cv_arr


# ──────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Energy levels: particle-in-a-box  En = n²·E₀ ──────────
    N_LEVELS = 100
    energies = np.array([n**2 for n in range(1, N_LEVELS + 1)], dtype=float)

    # ── Parameters ────────────────────────────────────────────
    BETA_CHECK    = 0.05    # β = 1/(kB·T) for the convergence checks
    XI_CHECK      = 3.0     # starting ξ (independent scaling factor)
    CV_CLASSICAL  = 0.5     # expected classical limit (1D box: kB/2)

    TOL_CV        = 1e-3    # level-convergence tolerance
    TOL_XI        = 5e-3    # classical-limit tolerance
    MIN_STABLE_LC = 5       # sliding-window width for level convergence
    MIN_STABLE_CL = 5       # sliding-window width for ξ sweep
    BAIL_STREAK   = 8       # monotone-fall steps before finite-N bail-out
    XI_MULT       = 1.5     # ξ multiplier per step
    MAX_XI_STEPS  = 80      # hard cap on ξ iterations

    BETA_MIN      = 0.02    # β range for Cv(T) sweep
    BETA_MAX      = 5.0
    N_BETA        = 400

    # Save plot next to this script file (works on Windows, Mac, Linux)
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    SAVE_PATH = os.path.join(_script_dir, "quantum_cv_plot.png")

    print("\n" + "═"*64)
    print("  Quantum Cv — Classical Limit Calculator")
    print("═"*64)
    print(f"  Energy levels   : En = n²,  n = 1…{N_LEVELS}")
    print(f"  β_check         : {BETA_CHECK}  (T_check = {1/BETA_CHECK:.1f})")
    print(f"  ξ_start         : {XI_CHECK}  (independent scaling factor)")
    print(f"  Classical limit : Cv/kB = {CV_CLASSICAL}")
    print(f"  tol_cv={TOL_CV}  tol_xi={TOL_XI}  "
          f"window_lc={MIN_STABLE_LC}  window_cl={MIN_STABLE_CL}  bail={BAIL_STREAK}")

    lc, cl, T_arr, cv_arr = sweep_and_plot(
        energies      = energies,
        beta_check    = BETA_CHECK,
        xi_check      = XI_CHECK,
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
        save_path     = SAVE_PATH,
    )

    print("\n" + "═"*64)
    print("  Summary")
    print("═"*64)
    lc_str = (f"✓ YES  (from n={lc['converged_at']},  "
              f"Cv={lc['cv_values'][lc['n_values'].index(lc['converged_at'])]:.6f})"
              if lc["converged"] else "✗ NO")
    print(f"  Level convergence : {lc_str}")
    if lc["shoulders"]:
        for sh in lc["shoulders"]:
            print(f"    shoulder at n={sh['start_n']}–{sh['end_n']}  "
                  f"Cv≈{sh['cv_value']:.6f}")
    cl_str = (f"✓ YES  Cv/kB = {cl['converged_cv']:.6f}"
              if cl["classical_reached"] else f"✗ NO  ({cl['stopped_reason']})")
    print(f"  Classical limit   : {cl_str}")
    if cl["shoulder_xi"]:
        print(f"    transient plateau near ξ={cl['shoulder_xi']:.4f}")
    print("═"*64 + "\n")