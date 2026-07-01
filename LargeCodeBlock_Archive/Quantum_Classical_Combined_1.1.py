import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

# ── Changes from last version ────────────────────────────────────────────────
# 1.0 -> 1.1
# Simplified the whole code to make it shorter and more readable. 
# Reduced if/else blocks lines and grouped dict fields together.
# 1135 lines -> 406 lines

# ── Core Cv formula ─────────────────────────────────────────────────────────
# Cv/kB = (β²/ξ⁴) · (⟨E²⟩ − ⟨E⟩²)
# ξ=1 → quantum; ξ→∞ → classical limit
def compute_cv(energies, beta, xi=1.0):
    energies = np.asarray(energies, dtype=float)
    a = beta * energies / xi**2
    w = np.exp(-(a - a.min()))          # shift to avoid overflow
    Z = w.sum()
    if Z == 0 or not np.isfinite(Z):
        return np.nan
    avg_E  = np.dot(w, energies) / Z
    avg_E2 = np.dot(w, energies**2) / Z
    return float((beta**2 / xi**4) * (avg_E2 - avg_E**2))


# ── ξ-convergence: sweep ξ upward until Cv plateaus then collapses ──────────
def converge_xi(energies, beta, xi_start, tol_xi, min_stable, xi_mult, max_steps):
    xis, cvs = [], []
    xi = xi_start
    stop_reason = "max_steps"

    for step in range(max_steps):
        cv = compute_cv(energies, beta, xi)
        xis.append(xi)
        cvs.append(cv)

        if step >= 2:
            last3 = np.array(cvs[-3:])
            if np.isclose(last3, 0.0, atol=0.2).all():
                pre = cvs[:-3]
                if len(pre) >= min_stable + 1:
                    pre_d = [abs(pre[i] - pre[i-1]) for i in range(1, len(pre))]
                    streak, found = 0, False
                    for d in pre_d:
                        if d < tol_xi:
                            streak += 1
                            if streak >= min_stable:
                                found = True; break
                        else:
                            streak = 0
                    stop_reason = "converged" if found else "finite_n"
                else:
                    stop_reason = "finite_n"
                break
        xi *= xi_mult

    deltas = [None] + [abs(cvs[i] - cvs[i-1]) for i in range(1, len(cvs))]

    xi_converged = cv_converged = None
    if stop_reason == "converged":
        pre_cvs = cvs[:-3]
        pre_d   = [None] + [abs(pre_cvs[i] - pre_cvs[i-1]) for i in range(1, len(pre_cvs))]
        plat_end = None
        for i in range(len(pre_d)-1, 0, -1):
            if pre_d[i] is not None and pre_d[i] < tol_xi:
                plat_end = i; break
        if plat_end is not None:
            plat_start = plat_end
            while plat_start > 1 and pre_d[plat_start-1] is not None and pre_d[plat_start-1] < tol_xi:
                plat_start -= 1
            n_take = min(3, plat_end - plat_start + 1)
            idx_start = plat_end - n_take + 1
            xi_converged = float(np.mean(xis[idx_start:plat_end+1]))
            cv_converged = float(np.mean(cvs[idx_start:plat_end+1]))
        else:
            xi_converged = float(np.mean(xis[-6:-3]))
            cv_converged = float(np.mean(cvs[-6:-3]))

    return {
        "xi_converged": xi_converged, "cv_converged": cv_converged,
        "converged": stop_reason == "converged", "stop_reason": stop_reason,
        "xi_values": xis, "cv_values": cvs, "deltas": deltas,
    }


# ── n-convergence: add levels one at a time until Cv stops changing ──────────
def converge_n(energies, beta, xi, tol_cv, min_stable):
    N = len(energies)
    n_values  = list(range(2, N+1))
    cv_values = [compute_cv(energies[:n], beta, xi) for n in n_values]
    deltas    = [None] + [abs(cv_values[i] - cv_values[i-1]) for i in range(1, len(cv_values))]
    stable    = [False] + [(d < tol_cv) for d in deltas[1:]]

    runs = []
    i = 0
    while i < len(stable):
        if stable[i]:
            j = i
            while j < len(stable) and stable[j]: j += 1
            if j - i >= min_stable: runs.append((i, j-1))
            i = j
        else:
            i += 1

    last_idx = len(n_values) - 1
    n_converged = cv_conv = None
    for start_idx, end_idx in runs:
        if end_idx == last_idx:
            n_converged = n_values[start_idx]
            cv_conv     = cv_values[start_idx]
            break

    return {
        "n_converged": n_converged, "cv_converged": cv_conv,
        "converged": n_converged is not None,
        "n_values": n_values, "cv_values": cv_values, "deltas": deltas,
    }


# ── Sweep all temperatures, running ξ and n convergence at each ─────────────
def sweep_temperature_range(energies, beta_arr,
                             xi_start, tol_xi, min_stable_xi, xi_mult, max_xi_steps,
                             tol_cv, min_stable_n, verbose=True):
    n_T = len(beta_arr)
    cv_classical = np.full(n_T, np.nan)
    xi_conv_arr  = np.full(n_T, np.nan)
    n_conv_arr   = np.full(n_T, np.nan)
    xi_results, n_results = [], []

    it = tqdm(range(n_T), desc="  Sweeping T range", unit="T") if verbose else range(n_T)
    for idx in it:
        beta = beta_arr[idx]

        xr = converge_xi(energies, beta, xi_start, tol_xi, min_stable_xi, xi_mult, max_xi_steps)
        xi_results.append(xr)
        xi_use = xr["xi_converged"] if xr["converged"] else 1.0
        if xr["converged"]:
            xi_conv_arr[idx] = xr["xi_converged"]
            cv_classical[idx] = xr["cv_converged"]

        nr = converge_n(energies, beta, xi=xi_use, tol_cv=tol_cv, min_stable=min_stable_n)
        n_results.append(nr)
        if nr["converged"]:
            n_conv_arr[idx] = nr["n_converged"]
            if not xr["converged"]:
                cv_classical[idx] = nr["cv_converged"]

    if verbose:
        n_xi_fail = np.isnan(xi_conv_arr).sum()
        n_n_fail  = np.isnan(n_conv_arr).sum()
        if n_xi_fail: print(f"  ⚠  ξ-convergence failed at {n_xi_fail}/{n_T} temperatures.")
        if n_n_fail:  print(f"  ⚠  n-convergence failed at {n_n_fail}/{n_T} temperatures.")
        if not n_xi_fail and not n_n_fail: print(f"  ✓  Both ξ and n converged at all {n_T} temperatures.")

    return {
        "cv_classical": cv_classical, "xi_conv": xi_conv_arr, "n_conv": n_conv_arr,
        "xi_results": xi_results, "n_results": n_results,
        "xi_fail_mask": np.isnan(xi_conv_arr), "n_fail_mask": np.isnan(n_conv_arr),
    }


def compute_quantum_cv_curve(energies, beta_arr, xi=1.0):
    return np.array([compute_cv(energies, b, xi) for b in beta_arr])


# ── Diagnostic plot: ξ sweep at the hardest temperature ─────────────────────
def plot_xi_convergence_diagnostic(xi_result, beta_val, T_K_val, tol_xi, system_name):
    BLUE   = "#1f77b4"
    GREEN  = "#2ca02c"
    ORANGE = "#d62728"
    YELLOW = "#bcbd22"
    GRAY   = "#7f7f7f"

    xis, cvs, deltas = xi_result["xi_values"], xi_result["cv_values"], xi_result["deltas"]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(f"{system_name} — ξ-Convergence Diagnostic\n"
                 f"Hardest T: {T_K_val:.2f}  (β = {beta_val:.4f})",
                 fontsize=12, fontweight="bold")

    ax.plot(xis, cvs, color=BLUE, linewidth=1.5, marker="s", markersize=5, zorder=3, label="Cv(ξ)")

    # Colour-coded dots: green = on plateau, yellow = collapsing, gray = rising
    for i in range(len(xis)):
        d = deltas[i]
        if d is not None and d < tol_xi:       c = GREEN
        elif i > 0 and cvs[i] < cvs[i-1]:     c = YELLOW
        else:                                   c = GRAY
        ax.scatter([xis[i]], [cvs[i]], color=c, zorder=5, s=60)

    if xi_result["converged"]:
        xc, cc = xi_result["xi_converged"], xi_result["cv_converged"]
        ax.axvline(xc, color=GREEN, linestyle=":", linewidth=1.3, label=f"ξ_conv = {xc:.3f}")
        ax.scatter([xc], [cc], color=GREEN, zorder=6, s=100, label=f"Cv_conv = {cc:.4f}")
        ann, ann_colour = f"Converged ✓\nξ_conv = {xc:.3f}\nCv_conv/kB = {cc:.5f}", GREEN
    else:
        ann, ann_colour = f"NOT converged\n({xi_result['stop_reason']})", ORANGE

    ax.text(0.97, 0.97, ann, transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color=ann_colour,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=ann_colour, alpha=0.9))

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
    plt.tight_layout(); plt.show()


# ── Diagnostic plot: n sweep at the hardest temperature ─────────────────────
def plot_n_convergence_diagnostic(n_result, beta_val, T_K_val, tol_cv, system_name):
    BLUE   = "#1f77b4"
    GREEN  = "#2ca02c"
    ORANGE = "#d62728"
    PURPLE = "#9467bd"

    ns, cvs, deltas = n_result["n_values"], n_result["cv_values"], n_result["deltas"]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(f"{system_name} — n-Convergence Diagnostic\n"
                 f"Hardest T: {T_K_val:.2f}  (β = {beta_val:.4f})",
                 fontsize=12, fontweight="bold")

    ax.plot(ns, cvs, color=BLUE, linewidth=1.5, marker="o", markersize=3, zorder=3, label="Cv(n levels)")

    if n_result["converged"]:
        nc = n_result["n_converged"]
        idx = ns.index(nc)
        ax.axvline(nc, color=GREEN, linestyle=":", linewidth=1.3, label=f"Converged at n = {nc}")
        ax.scatter([nc], [cvs[idx]], color=GREEN, zorder=6, s=100)
        ann, ann_colour = f"Converged ✓  at n = {nc}\nCv/kB = {cvs[idx]:.5f}", GREEN
    else:
        ann, ann_colour = "NOT converged within N levels\nIncrease N_MAX", ORANGE

    ax.text(0.97, 0.05, ann, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color=ann_colour,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=ann_colour, alpha=0.9))

    # Shade shoulder regions (temporary false convergences that end before last level)
    stable = [False] + [(d is not None and d < tol_cv) for d in deltas[1:]]
    i = 0
    while i < len(stable):
        if stable[i]:
            j = i
            while j < len(stable) and stable[j]: j += 1
            if j - i >= 3 and j < len(stable):
                ax.axvspan(ns[i], ns[j-1], color=PURPLE, alpha=0.18)
            i = j
        else:
            i += 1

    handles, labels = ax.get_legend_handles_labels()
    handles.append(mpatches.Patch(color=PURPLE, alpha=0.35, label="shoulder region"))
    labels.append("shoulder region")
    ax.legend(handles, labels, fontsize=9)
    ax.set_xlabel("Number of energy levels  n", fontsize=11)
    ax.set_ylabel("Cv / kB", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout(); plt.show()


# ── Main Cv(T) figure ────────────────────────────────────────────────────────
def plot_cv_curves(T_K_arr, cv_quantum, cv_numerical_classical,
                   xi_conv_arr, n_conv_arr, system_name,
                   cv_analytic_classical=None, T_units_label=r"$k_B T \,/\, E_0$",
                   T_scale_factor=1.0):
    BLUE   = "#1f77b4"
    GREEN  = "#2ca02c"
    ORANGE = "#d62728"
    PURPLE = "#9467bd"
    RED    = "#d62728"

    T_plot = T_K_arr / T_scale_factor

    fig, ax1 = plt.subplots(figsize=(9, 6))
    fig.suptitle(f"{system_name} — Cv(T)", fontsize=13, fontweight="bold")

    ax1.plot(T_plot, cv_quantum, color=BLUE, linewidth=2, label="Quantum Cv(T)")
    ax1.plot(T_plot, cv_numerical_classical, color=GREEN, linewidth=2, linestyle="--",
             label="Numerical classical limit")

    if cv_analytic_classical is not None:
        cv_ref = np.full_like(T_plot, cv_analytic_classical) if np.isscalar(cv_analytic_classical) else cv_analytic_classical
        ax1.plot(T_plot, cv_ref, color=ORANGE, linewidth=1.5, linestyle=":", label="Analytic classical limit")

    ax1.set_xlabel(T_units_label, fontsize=12)
    ax1.set_ylabel(r"$C_v \,/\, k_B$", fontsize=12)
    ax1.set_xscale("log")
    ax1.legend(fontsize=10, loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    valid_xi = ~np.isnan(xi_conv_arr)
    valid_n  = ~np.isnan(n_conv_arr)
    ax2.plot(T_plot[valid_xi], xi_conv_arr[valid_xi], color=PURPLE, linewidth=1, linestyle="-.", alpha=0.6, label="ξ_conv(T)")
    ax2.plot(T_plot[valid_n],  n_conv_arr[valid_n],  color=RED,    linewidth=1, linestyle=":",   alpha=0.6, label="n_conv(T)")
    ax2.set_ylabel("Converged ξ  /  n  (secondary axis)", fontsize=10, color=PURPLE)
    ax2.tick_params(axis="y", colors=PURPLE)
    ax2.legend(fontsize=9, loc="upper right")

    plt.tight_layout(); plt.show()


# ── Full pipeline: energy levels in → all plots out ──────────────────────────
def run(energies, E0_J, system_name,
        beta_min=0.02, beta_max=5.0, n_beta=200,
        xi_start=3.0, tol_xi=5e-3, min_stable_xi=5, xi_multiplier=1.3, max_xi_steps=80,
        tol_cv=1e-3, min_stable_n=3,
        cv_analytic=None, T_units_label=r"$k_B T \,/\, E_0$", T_scale_factor=None):

    natural_units = (E0_J is None)
    if natural_units:
        T_scale_factor = 1.0
    elif T_scale_factor is None:
        T_scale_factor = E0_J / KB_SI

    beta_arr = np.linspace(beta_min, beta_max, n_beta)
    T_K_arr  = 1.0 / beta_arr if natural_units else E0_J / (KB_SI * beta_arr)

    print(f"\n{'═'*60}\n  {system_name}\n{'═'*60}")
    print(f"  {len(energies)} levels, E_min={energies[0]:.3g}, E_max={energies[-1]:.3g}")
    print(f"  β: {beta_min} → {beta_max}  ({n_beta} points)")

    sweep = sweep_temperature_range(
        energies, beta_arr,
        xi_start, tol_xi, min_stable_xi, xi_multiplier, max_xi_steps,
        tol_cv, min_stable_n, verbose=True,
    )
    cv_classical = sweep["cv_classical"]
    xi_conv      = sweep["xi_conv"]
    n_conv       = sweep["n_conv"]

    #valid_n = n_conv[~np.isnan(n_conv)]
    #n_quantum = int(np.max(valid_n)) if len(valid_n) > 0 else len(energies)
    #cv_quantum = compute_quantum_cv_curve(energies[:n_quantum], beta_arr, xi=1.0)

    # Calculate the exact quantum curve using ALL available energy levels
    cv_quantum = compute_quantum_cv_curve(energies, beta_arr, xi=1.0)

    # Diagnostic: show convergence plots for the "hardest" temperatures
    t_lbl = "T*" if natural_units else "K"

    valid_xi_mask = ~np.isnan(xi_conv)
    if valid_xi_mask.any():
        idx_hard_xi = int(np.nanargmax(xi_conv))
        print(f"  Hardest ξ-convergence: T={T_K_arr[idx_hard_xi]:.4g} {t_lbl}, ξ_conv={xi_conv[idx_hard_xi]:.3f}")
        plot_xi_convergence_diagnostic(sweep["xi_results"][idx_hard_xi],
                                       beta_arr[idx_hard_xi], T_K_arr[idx_hard_xi],
                                       tol_xi, system_name)

    valid_n_mask = ~np.isnan(n_conv)
    if valid_n_mask.any():
        idx_hard_n = int(np.nanargmax(n_conv))
        print(f"  Hardest n-convergence:  T={T_K_arr[idx_hard_n]:.4g} {t_lbl}, n_conv={int(n_conv[idx_hard_n])}")
        nr_hard = sweep["n_results"][idx_hard_n]
        if nr_hard is not None:
            plot_n_convergence_diagnostic(nr_hard, beta_arr[idx_hard_n], T_K_arr[idx_hard_n],
                                          tol_cv, system_name)

    plot_cv_curves(T_K_arr, cv_quantum, cv_classical, xi_conv, n_conv, system_name,
                   cv_analytic_classical=cv_analytic,
                   T_units_label=T_units_label, T_scale_factor=T_scale_factor)

    print(f"\n  ξ-conv: {valid_xi_mask.sum()}/{n_beta}  (max ξ={np.nanmax(xi_conv):.2f})" if valid_xi_mask.any() else "  ξ-conv: not applicable")
    print(f"  n-conv: {valid_n_mask.sum()}/{n_beta}  (max n={int(np.nanmax(n_conv))})" if valid_n_mask.any() else "  n-conv: failed at all T")
    print(f"{'═'*60}\n")

    return {"beta_arr": beta_arr, "T_K_arr": T_K_arr,
            "cv_quantum": cv_quantum, "cv_classical": cv_classical,
            "xi_conv": xi_conv, "n_conv": n_conv, "sweep": sweep}


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":

    hbar = 1e-34; m = 1.0; L = 1.0; kB = 1.0
    E_g  = (hbar**2 * np.pi**2) / (2 * m * L**2)

    # Shared sweep settings
    BETA_MIN, BETA_MAX, N_BETA = 0.02, 5.0, 200
    XI_START, TOL_XI, MIN_STABLE_XI, XI_MULT, MAX_XI_STEPS = 1.0, 1e-3, 4, 1.15, 100
    TOL_CV, MIN_STABLE_N = 1e-3, 3

    # System 1: 1-D Particle-in-a-Box  (En = n², classical limit Cv/kB = 0.5)
    energies_box = np.array([n**2 for n in range(1, 501)], dtype=float)
    results_box = run(
        energies=energies_box, E0_J=None,
        system_name="1-D Particle-in-a-Box",
        beta_min=BETA_MIN, beta_max=BETA_MAX, n_beta=N_BETA,
        xi_start=XI_START, tol_xi=TOL_XI, min_stable_xi=MIN_STABLE_XI,
        xi_multiplier=XI_MULT, max_xi_steps=MAX_XI_STEPS,
        tol_cv=TOL_CV, min_stable_n=MIN_STABLE_N,
        cv_analytic=0.5, T_units_label=r"$k_B T / E_g$",
    )

    # System 2: 1-D Harmonic Oscillator  (En = n+0.5, classical limit Cv/kB = 1.0)
    energies_ho = np.array([n + 0.5 for n in range(5000)], dtype=float)
    results_ho = run(
        energies=energies_ho, E0_J=None,
        system_name="1-D Harmonic Oscillator",
        beta_min=BETA_MIN, beta_max=BETA_MAX, n_beta=N_BETA,
        xi_start=XI_START, tol_xi=TOL_XI, min_stable_xi=MIN_STABLE_XI,
        xi_multiplier=XI_MULT, max_xi_steps=MAX_XI_STEPS,
        tol_cv=TOL_CV, min_stable_n=MIN_STABLE_N,
        cv_analytic=1.0, T_units_label=r"$k_B T / \hbar\omega$",
    )
