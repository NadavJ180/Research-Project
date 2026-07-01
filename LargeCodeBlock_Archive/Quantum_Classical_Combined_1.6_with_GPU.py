import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm
import time
import threading
import sys

# Standard NumPy handles vectorized matrix broadcasting on your CPU cores
from DVR_algorithm import colbert_miller_dvr_1d

# ── Phase-Space Auto-Scanner (CPU-bound with Progress Bar) ───────────────────
def auto_configure_dvr(potential_func, num_levels, shape="smooth", mass=1.0, hbar=1.0):
    print(f"  [Auto-Scanner] Analyzing '{shape}' potential for {num_levels} states...")
    if shape == "hard_wall":
        raw_points = np.linspace(-50, 50, 10000)
        x_scan = np.zeros_like(raw_points)
        V_scan = np.zeros_like(raw_points)
        
        for i, x in enumerate(tqdm(raw_points, desc=" 🔍 Scanning grid boundaries", unit="point")):
            x_scan[i] = x
            with np.errstate(over='ignore'):
                V_scan[i] = potential_func(x)
                
        valid_idx = np.where(~np.isinf(V_scan) & (V_scan < 1e6))[0]
        if len(valid_idx) == 0: raise ValueError("Invalid potential.")
        x_left, x_right = x_scan[valid_idx[0]], x_scan[valid_idx[-1]]
        span = x_right - x_left
        x_min, x_max = x_left + (span * 1e-4), x_right - (span * 1e-4)
        grid_points = int(2.4 * num_levels)
    elif shape == "smooth":
        res = opt.minimize(potential_func, x0=0.0)
        x_bottom, v_min = res.x[0], res.fun
        E_ceiling = v_min + (1.5 * num_levels)
        root_func = lambda x: potential_func(x) - E_ceiling
        
        x_right = opt.fsolve(root_func, x0=x_bottom + 1.0)[0]
        x_left = opt.fsolve(root_func, x0=x_bottom - 1.0)[0]
        span = abs(x_right - x_left)
        x_min, x_max = x_left - (0.15 * span) - 2.0, x_right + (0.15 * span) + 2.0
        k_max = np.sqrt(2.0 * mass * (E_ceiling - v_min)) / hbar
        grid_points = max(int(np.ceil((x_max - x_min) / (np.pi / (2.0 * k_max)))), int(4.0 * num_levels))
    else: raise ValueError("Shape must be 'smooth' or 'hard_wall'")
    
    print(f"  [Auto-Scanner] Bounds: [{x_min:.3f}, {x_max:.3f}] | Grid Points: {grid_points}\n")
    return x_min, x_max, grid_points

# ── Core Cv formula (Vectorized NumPy) ───────────────────────────────────────
def compute_cv(energies, beta, xi=1.0):
    """Computes Cv using matrix broadcasting (High-performance CPU usage)."""
    E = np.asarray(energies, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    
    a = (beta[:, None] * E[None, :]) / (xi**2)
    a_min = np.min(a, axis=1, keepdims=True)
    w = np.exp(-(a - a_min))
    
    Z = np.sum(w, axis=1)
    avg_E = np.sum(w * E[None, :], axis=1) / Z
    avg_E2 = np.sum(w * (E[None, :]**2), axis=1) / Z
    
    cv = (beta**2 / xi**4) * (avg_E2 - avg_E**2)
    
    return cv if cv.size > 1 else cv.item()

# ── Convergence Utilities ────────────────────────────────────────────────────
def converge_xi(energies, beta, xi_start, tol_xi, min_stable, xi_mult, max_steps):
    xis, cvs = [], []
    xi = xi_start
    stop_reason = "max_steps"
    for step in range(max_steps):
        cv = compute_cv(energies, [beta], xi)
        xis.append(xi); cvs.append(cv)
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
                            if streak >= min_stable: found = True; break
                        else: streak = 0
                    stop_reason = "converged" if found else "finite_n"
                else: stop_reason = "finite_n"
                break
        xi *= xi_mult
    deltas = [None] + [abs(cvs[i] - cvs[i-1]) for i in range(1, len(cvs))]
    return {"xi_converged": np.mean(xis[-3:]) if stop_reason == "converged" else None, "cv_converged": np.mean(cvs[-3:]) if stop_reason == "converged" else None, "converged": stop_reason == "converged", "stop_reason": stop_reason, "xi_values": xis, "cv_values": cvs, "deltas": deltas}

def converge_n(energies, beta, xi, tol_cv, min_stable):
    N = len(energies)
    n_values = list(range(2, N+1))
    cv_values = [compute_cv(energies[:n], [beta], xi) for n in n_values]
    deltas = [None] + [abs(cv_values[i] - cv_values[i-1]) for i in range(1, len(cv_values))]
    stable = [False] + [(d is not None and d < tol_cv) for d in deltas[1:]]
    runs = []
    i = 0
    while i < len(stable):
        if stable[i]:
            j = i
            while j < len(stable) and stable[j]: j += 1
            if j - i >= min_stable: runs.append((i, j-1))
            i = j
        else: i += 1
    n_converged = cv_conv = None
    for start_idx, end_idx in runs:
        if end_idx == len(n_values) - 1:
            n_converged, cv_conv = n_values[start_idx], cv_values[start_idx]
            break
    return {"n_converged": n_converged, "cv_converged": cv_conv, "converged": n_converged is not None, "n_values": n_values, "cv_values": cv_values, "deltas": deltas}

# ── Sweep and Diagnostics ────────────────────────────────────────────────────
def sweep_temperature_range(energies, beta_arr, xi_start, tol_xi, min_stable_xi, xi_mult, max_xi_steps, tol_cv, min_stable_n, verbose=True):
    n_T = len(beta_arr)
    cv_classical, xi_conv_arr, n_conv_arr = np.full(n_T, np.nan), np.full(n_T, np.nan), np.full(n_T, np.nan)
    xi_results, n_results = [], []
    it = tqdm(range(n_T), desc=" 🌡️ Sweeping T range", unit="T") if verbose else range(n_T)
    for idx in it:
        beta = beta_arr[idx]
        xr = converge_xi(energies, beta, xi_start, tol_xi, min_stable_xi, xi_mult, max_xi_steps)
        xi_results.append(xr)
        xi_use = xr["xi_converged"] if xr["converged"] else 1.0
        if xr["converged"]: xi_conv_arr[idx], cv_classical[idx] = xr["xi_converged"], xr["cv_converged"]
        nr = converge_n(energies, beta, xi=xi_use, tol_cv=tol_cv, min_stable=min_stable_n)
        n_results.append(nr)
        if nr["converged"]: n_conv_arr[idx] = nr["n_converged"]
        if nr["converged"] and not xr["converged"]: cv_classical[idx] = nr["cv_converged"]
    return {"cv_classical": cv_classical, "xi_conv": xi_conv_arr, "n_conv": n_conv_arr, "xi_results": xi_results, "n_results": n_results}

def plot_cv_curves(T_arr, cv_quantum, cv_classical, xi_conv_arr, n_conv_arr, system_name, cv_analytic_classical=None):
    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax1.plot(T_arr, cv_quantum, label="Quantum Cv(T)")
    ax1.plot(T_arr, cv_classical, linestyle="--", label="Numerical classical limit")
    ax1.set_xscale("log"); ax1.legend(); plt.show()

# ── Clean Live Spinner Worker Function ───────────────────────────────────────
def spinner_worker(stop_event):
    spin_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    start_time = time.time()
    idx = 0
    while not stop_event.is_set():
        elapsed = time.time() - start_time
        # \r brings the cursor to the beginning of the line to overwrite it cleanly
        sys.stdout.write(f"\r {spin_chars[idx % len(spin_chars)]} Diagonalizing Hamiltonian... Elapsed time: {elapsed:.1f}s")
        sys.stdout.flush()
        idx += 1
        time.sleep(0.1)
    # Clear the entire line once complete
    sys.stdout.write("\r" + " " * 70 + "\r")
    sys.stdout.flush()

# ── Main Entry ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    NUM_STATES = 5000 
    BETA_MIN, BETA_MAX, N_BETA = 0.02, 5.0, 200
    
    # System 1: Particle-in-a-Box
    def box_pot(x): return np.where((x < 0) | (x > np.pi), np.inf, 0.0)
    
    x_min_b, x_max_b, N_b = auto_configure_dvr(box_pot, NUM_STATES, shape="hard_wall")
    
    print(f" ⚙️ Entering DVR solver (Matrix size: {N_b}x{N_b})...")
    
    # Initialize and boot the asynchronous thread spinner
    stop_spinner = threading.Event()
    spinner_thread = threading.Thread(target=spinner_worker, args=(stop_spinner,))
    
    t0 = time.time()
    spinner_thread.start()
    
    try:
        # Core linear algebra calculation executing on the main thread
        energies_box = colbert_miller_dvr_1d(box_pot, NUM_STATES, x_min_b, x_max_b, N_b, 0.5, 1.0)
    finally:
        # Ensure that regardless of success or error, the background thread closes cleanly
        stop_spinner.set()
        spinner_thread.join()
        
    print(f" ✅ Diagonalization completed successfully in {time.time() - t0:.2f} seconds.\n")
    
    sweep_box = sweep_temperature_range(energies_box, np.linspace(BETA_MIN, BETA_MAX, N_BETA), 1.0, 1e-3, 5, 1.3, 80, 1e-4, 3)
    
    # Plot results
    T_arr = 1.0 / np.linspace(BETA_MIN, BETA_MAX, N_BETA)
    plot_cv_curves(T_arr, compute_cv(energies_box[:NUM_STATES], np.linspace(BETA_MIN, BETA_MAX, N_BETA)), sweep_box["cv_classical"], sweep_box["xi_conv"], sweep_box["n_conv"], "1-D Particle-in-a-Box")