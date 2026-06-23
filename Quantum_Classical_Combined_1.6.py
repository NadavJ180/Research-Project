import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

from DVR_algorithm import colbert_miller_dvr_1d

# ── Changes from 1.4 to 1.6 ──────────────────────────────────────────────────
# 1. Added `auto_configure_dvr` to act as a Phase-Space Auto-Scanner.
# 2. Implemented `scipy.optimize.minimize` and `fsolve` for smooth boundary detection.
# 3. Implemented a domain scanner for hard-wall boundary detection.
# 4. 'NUM_STATES' and 'shape' are now the master controllers.

# ── Phase-Space Auto-Scanner & Boundary Detector ─────────────────────────────
def auto_configure_dvr(potential_func, num_levels, shape="smooth", mass=1.0, hbar=1.0):
    """
    Dynamically calculates optimal spatial boundaries and Nyquist grid density.
    """
    print(f"  [Auto-Scanner] Analyzing '{shape}' potential for {num_levels} states...")
    
    if shape == "hard_wall":
        # 1. Probe for the infinite walls
        x_scan = np.linspace(-50, 50, 10000)
        with np.errstate(over='ignore'):
            V_scan = potential_func(x_scan)
        
        valid_idx = np.where(~np.isinf(V_scan) & (V_scan < 1e6))[0]
        if len(valid_idx) == 0:
            raise ValueError("Could not find a valid potential well. Check your V(x) definition.")
            
        x_left, x_right = x_scan[valid_idx[0]], x_scan[valid_idx[-1]]
        
        # Pull slightly inward to avoid evaluating exactly on the singularity
        span = x_right - x_left
        x_min = x_left + (span * 1e-4)
        x_max = x_right - (span * 1e-4)
        
        # 2. Sinc-DVR truncation buffer (2.4x oversampling for hard walls)
        grid_points = int(2.4 * num_levels)
        
    elif shape == "smooth":
        # 1. Find the bottom of the potential well
        res = opt.minimize(potential_func, x0=0.0)
        x_bottom = res.x[0]
        v_min = res.fun
        
        # 2. Define a conservative energy ceiling (approx highest classical turning point)
        # Using a 1.5x heuristic scalar to ensure the bounding box is wide enough
        E_ceiling = v_min + (1.5 * num_levels)
        
        # 3. Root-finding to detect boundaries: V(x) - E_ceiling = 0
        root_func = lambda x: potential_func(x) - E_ceiling
        
        try:
            x_right = opt.fsolve(root_func, x0=x_bottom + 1.0)[0]
            x_left  = opt.fsolve(root_func, x0=x_bottom - 1.0)[0]
        except:
            raise RuntimeError("fsolve failed to find classical turning points. Potential may not be bound.")
            
        # Add a decay buffer for the wavefunction tails (~15% of span)
        span = abs(x_right - x_left)
        x_min = x_left - (0.15 * span) - 2.0
        x_max = x_right + (0.15 * span) + 2.0
        
        # 4. Calculate Nyquist grid density from max momentum at well bottom
        k_max = np.sqrt(2.0 * mass * (E_ceiling - v_min)) / hbar
        dx_target = np.pi / (2.0 * k_max)
        
        grid_points = int(np.ceil((x_max - x_min) / dx_target))
        # Enforce a physical mathematical floor (4x multiplier for phase-space safety)
        grid_points = max(grid_points, int(4.0 * num_levels))
        
    else:
        raise ValueError("Shape must be 'smooth' or 'hard_wall'")

    print(f"  [Auto-Scanner] Bounds: [{x_min:.3f}, {x_max:.3f}] | Grid Points: {grid_points}")
    return x_min, x_max, grid_points


# ── Core Cv formula ─────────────────────────────────────────────────────────
def compute_cv(energies, beta, xi=1.0):
    energies = np.asarray(energies, dtype=float)
    a = beta * energies / xi**2
    w = np.exp(-(a - a.min()))
    Z = w.sum()
    if Z == 0 or not np.isfinite(Z):
        return np.nan
    avg_E  = np.dot(w, energies) / Z
    avg_E2 = np.dot(w, energies**2) / Z
    return float((beta**2 / xi**4) * (avg_E2 - avg_E**2))


# ── ξ-convergence ───────────────────────────────────────────────────────────
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
            peak_cv = max(cvs)
            
            if (peak_cv > 1e-4 and (last3 < 0.02 * peak_cv).all()) or np.isclose(last3, 0.0, atol=1e-5).all():
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
            plat_len  = plat_end - plat_start + 1
            mid       = plat_start + plat_len // 2
            n_take    = min(3, plat_len)
            idx_start = max(plat_start, mid - n_take // 2)
            xi_converged = float(np.mean(xis[idx_start:idx_start+n_take]))
            cv_converged = float(np.mean(cvs[idx_start:idx_start+n_take]))
        else:
            xi_converged = float(np.mean(xis[-6:-3]))
            cv_converged = float(np.mean(cvs[-6:-3]))

    return {
        "xi_converged": xi_converged, "cv_converged": cv_converged,
        "converged": stop_reason == "converged", "stop_reason": stop_reason,
        "xi_values": xis, "cv_values": cvs, "deltas": deltas,
    }


# ── n-convergence ───────────────────────────────────────────────────────────
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


# ── Sweep all temperatures ──────────────────────────────────────────────────
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


# ── Diagnostic plots ─────────────────────────────────────────────────────────
def plot_xi_convergence_diagnostic(xi_result, beta_val, T_K_val, tol_xi, system_name):
    BLUE, GREEN, ORANGE, YELLOW, GRAY = "#1f77b4", "#2ca02c", "#d62728", "#bcbd22", "#7f7f7f"
    xis, cvs, deltas = xi_result["xi_values"], xi_result["cv_values"], xi_result["deltas"]
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(f"{system_name} — ξ-Convergence Diagnostic\nHardest T: {T_K_val:.2f}  (β = {beta_val:.4f})", fontsize=12, fontweight="bold")
    ax.plot(xis, cvs, color=BLUE, linewidth=1.5, marker="s", markersize=5, zorder=3, label="Cv(ξ)")
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
    ax.text(0.97, 0.97, ann, transform=ax.transAxes, ha="right", va="top", fontsize=9, color=ann_colour,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=ann_colour, alpha=0.9))
    dot_legend = [mpatches.Patch(color=GREEN, label="stable  |ΔCv| < tol"), mpatches.Patch(color=YELLOW, label="falling (finite-N collapse)"), mpatches.Patch(color=GRAY, label="rising / first point")]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + dot_legend, fontsize=9, loc="lower left")
    ax.set_xlabel("Scaling factor  ξ", fontsize=11)
    ax.set_ylabel("Cv / kB", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout(); plt.show()

def plot_n_convergence_diagnostic(n_result, beta_val, T_K_val, tol_cv, system_name):
    BLUE, GREEN, ORANGE, PURPLE = "#1f77b4", "#2ca02c", "#d62728", "#9467bd"
    ns, cvs, deltas = n_result["n_values"], n_result["cv_values"], n_result["deltas"]
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(f"{system_name} — n-Convergence Diagnostic\nHardest T: {T_K_val:.2f}  (β = {beta_val:.4f})", fontsize=12, fontweight="bold")
    ax.plot(ns, cvs, color=BLUE, linewidth=1.5, marker="o", markersize=3, zorder=3, label="Cv(n levels)")
    if n_result["converged"]:
        nc = n_result["n_converged"]
        idx = ns.index(nc)
        ax.axvline(nc, color=GREEN, linestyle=":", linewidth=1.3, label=f"Converged at n = {nc}")
        ax.scatter([nc], [cvs[idx]], color=GREEN, zorder=6, s=100)
        ann, ann_colour = f"Converged ✓  at n = {nc}\nCv/kB = {cvs[idx]:.5f}", GREEN
    else:
        ann, ann_colour = "NOT converged within N levels\nIncrease N_MAX", ORANGE
    ax.text(0.97, 0.05, ann, transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color=ann_colour,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=ann_colour, alpha=0.9))
    stable = [False] + [(d is not None and d < tol_cv) for d in deltas[1:]]
    i = 0
    while i < len(stable):
        if stable[i]:
            j = i
            while j < len(stable) and stable[j]: j += 1
            if j - i >= 3 and j < len(stable): ax.axvspan(ns[i], ns[j-1], color=PURPLE, alpha=0.18)
            i = j
        else: i += 1
    handles, labels = ax.get_legend_handles_labels()
    handles.append(mpatches.Patch(color=PURPLE, alpha=0.35, label="shoulder region"))
    labels.append("shoulder region")
    ax.legend(handles, labels, fontsize=9)
    ax.set_xlabel("Number of energy levels  n", fontsize=11)
    ax.set_ylabel("Cv / kB", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout(); plt.show()

def plot_cv_curves(T_arr, cv_quantum, cv_classical, xi_conv_arr, n_conv_arr, system_name,
                   cv_analytic_classical=None, T_units_label=r"$k_B T \,/\, E_0$"):
    BLUE, GREEN, ORANGE, PURPLE, RED = "#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#d62728"
    fig, ax1 = plt.subplots(figsize=(9, 6))
    fig.suptitle(f"{system_name} — Cv(T)", fontsize=13, fontweight="bold")
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
    valid_n  = ~np.isnan(n_conv_arr)
    ax2.plot(T_arr[valid_xi], xi_conv_arr[valid_xi], color=PURPLE, linewidth=1, linestyle="-.", alpha=0.6, label="ξ_conv(T)")
    ax2.plot(T_arr[valid_n],  n_conv_arr[valid_n],  color=RED,    linewidth=1, linestyle=":",   alpha=0.6, label="n_conv(T)")
    ax2.set_ylabel("Converged ξ  /  n  (secondary axis)", fontsize=10, color=PURPLE)
    ax2.tick_params(axis="y", colors=PURPLE)
    ax2.legend(fontsize=9, loc="upper right")
    plt.tight_layout(); plt.show()


# ── Full pipeline ────────────────────────────────────────────────────────────
def run(energies, system_name,
        beta_min=0.02, beta_max=5.0, n_beta=200,
        xi_start=1.0, tol_xi=1e-3, min_stable_xi=5, xi_multiplier=1.3, max_xi_steps=80,
        tol_cv=1e-4, min_stable_n=3,
        cv_analytic=None, T_units_label=r"$k_B T \,/\, E_0$"):

    beta_arr = np.linspace(beta_min, beta_max, n_beta)
    T_arr    = 1.0 / beta_arr

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

    valid_n = n_conv[~np.isnan(n_conv)]
    n_quantum = int(np.max(valid_n)) if len(valid_n) > 0 else len(energies)
    cv_quantum = compute_quantum_cv_curve(energies[:n_quantum], beta_arr, xi=1.0)

    valid_xi_mask = ~np.isnan(xi_conv)
    if valid_xi_mask.any():
        idx_hard_xi = int(np.nanargmax(xi_conv))
        print(f"  Hardest ξ-convergence: T*={T_arr[idx_hard_xi]:.4g}, ξ_conv={xi_conv[idx_hard_xi]:.3f}")
        plot_xi_convergence_diagnostic(sweep["xi_results"][idx_hard_xi], beta_arr[idx_hard_xi], T_arr[idx_hard_xi], tol_xi, system_name)

    valid_n_mask = ~np.isnan(n_conv)
    if valid_n_mask.any():
        idx_hard_n = int(np.nanargmax(n_conv))
        print(f"  Hardest n-convergence:  T*={T_arr[idx_hard_n]:.4g}, n_conv={int(n_conv[idx_hard_n])}")
        nr_hard = sweep["n_results"][idx_hard_n]
        if nr_hard is not None:
            plot_n_convergence_diagnostic(nr_hard, beta_arr[idx_hard_n], T_arr[idx_hard_n], tol_cv, system_name)

    plot_cv_curves(T_arr, cv_quantum, cv_classical, xi_conv, n_conv, system_name, cv_analytic_classical=cv_analytic, T_units_label=T_units_label)

    print(f"\n  ξ-conv: {valid_xi_mask.sum()}/{n_beta}  (max ξ={np.nanmax(xi_conv):.2f})" if valid_xi_mask.any() else "  ξ-conv: not applicable")
    print(f"  n-conv: {valid_n_mask.sum()}/{n_beta}  (max n={int(np.nanmax(n_conv))})" if valid_n_mask.any() else "  n-conv: failed at all T")
    print(f"{'═'*60}\n")

    return {"beta_arr": beta_arr, "T_arr": T_arr, "cv_quantum": cv_quantum, "cv_classical": cv_classical, "xi_conv": xi_conv, "n_conv": n_conv, "sweep": sweep}


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":

    BETA_MIN, BETA_MAX, N_BETA = 0.02, 5.0, 200
    XI_START, TOL_XI, MIN_STABLE_XI, XI_MULT, MAX_XI_STEPS = 1.0, 1e-3, 5, 1.3, 80
    TOL_CV, MIN_STABLE_N = 1e-4, 3

    NUM_STATES = 5000  

    # -------------------------------------------------------------------------
    # System 1: 1-D Particle-in-a-Box 
    # -------------------------------------------------------------------------
    def box_pot(x):
        # A true hard wall for the scanner to detect
        return np.where((x < 0) | (x > np.pi), np.inf, 0.0)
    
    x_min_box, x_max_box, N_box = auto_configure_dvr(box_pot, NUM_STATES, shape="hard_wall")
    
    energies_box = colbert_miller_dvr_1d(
        potential_func=box_pot, num_levels=NUM_STATES,
        x_min=x_min_box, x_max=x_max_box, num_points=N_box, mass=0.5, hbar=1.0
    )

    results_box = run(
        energies=energies_box, system_name="1-D Particle-in-a-Box",
        beta_min=BETA_MIN, beta_max=BETA_MAX, n_beta=N_BETA,
        xi_start=XI_START, tol_xi=TOL_XI, min_stable_xi=MIN_STABLE_XI,
        xi_multiplier=XI_MULT, max_xi_steps=MAX_XI_STEPS, tol_cv=TOL_CV, min_stable_n=MIN_STABLE_N,
        cv_analytic=0.5, T_units_label=r"$k_B T / E_g$",
    )

    # -------------------------------------------------------------------------
    # System 2: 1-D Harmonic Oscillator 
    # -------------------------------------------------------------------------
    ho_pot = lambda x: 0.5 * x**2
    
    x_min_ho, x_max_ho, N_ho = auto_configure_dvr(ho_pot, NUM_STATES, shape="smooth")
    
    energies_ho = colbert_miller_dvr_1d(
        potential_func=ho_pot, num_levels=NUM_STATES,
        x_min=x_min_ho, x_max=x_max_ho, num_points=N_ho, mass=1.0, hbar=1.0
    )

    results_ho = run(
        energies=energies_ho, system_name="1-D Harmonic Oscillator",
        beta_min=BETA_MIN, beta_max=BETA_MAX, n_beta=N_BETA,
        xi_start=XI_START, tol_xi=TOL_XI, min_stable_xi=MIN_STABLE_XI,
        xi_multiplier=XI_MULT, max_xi_steps=MAX_XI_STEPS, tol_cv=TOL_CV, min_stable_n=MIN_STABLE_N,
        cv_analytic=1.0, T_units_label=r"$k_B T / \hbar\omega$",
    )