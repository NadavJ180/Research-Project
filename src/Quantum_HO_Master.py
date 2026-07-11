"""
Quantum_HO_Master_1_3.py
=====================================================================
WHAT THIS FILE DOES
---------------------------------------------------------------------
Single entry-point driver for the Harmonic Oscillator benchmark.
This file does NOT define any new physics or numerics itself -- it
only imports and calls functions from the other project files, in
clearly separated sections, so each piece can keep evolving
independently without this file needing structural changes:

    SECTION 0 -- Global configuration (parameters live here)
    SECTION 1 -- DVR setup & diagonalization      (DVR_Algorithm_1_4)
    SECTION 2 -- Analytic HO energy levels         (HO_Analytical_1_0)
    SECTION 3 -- Energy-level accuracy check       (HO_Energy_Level_Error_1_1)
    SECTION 4 -- General Cv pipeline (numerical)   (Quantum_Classical_Combined_1_9)
    SECTION 5 -- Numerical vs analytical benchmark (HO_Benchmark_1_1)
    SECTION 6 -- DVR limit analysis                (DVR_Limit_Finder_1_2)

CHANGELOG (v1.2 -> v1.3)
---------------------------------------------------------------------
- Added SimpleTimer: a lightweight daemon thread that prints elapsed
  time every 10 s so it is clear the code has not crashed during the
  long DVR solve (Section 1) and Cv T-range sweep (Section 4). Uses
  threading rather than multiprocessing -- zero process-spawn
  overhead. May occasionally miss a tick if LAPACK holds the GIL for
  longer than 10 s, but that is acceptable for a pure alive-indicator.
- Updated imports: DVR_Algorithm_1_4 (timer removed from DVR),
  DVR_Limit_Finder_1_2 (annotation removed from dx plot),
  HO_Benchmark_1_1 (Einstein label removed from Cv curves).
- Updated master parameters per project calibration:
    NUM_STATES = 500
    BETA_MIN=0.1, BETA_MAX=50.0, N_BETA=500
    XI_START=1.0, TOL_XI=5e-3, MIN_STABLE_XI=3, XI_MULT=1.1,
    MAX_XI_STEPS=80
    TOL_CV=1e-4, MIN_STABLE_N=3
=====================================================================
"""

import threading
import time
import multiprocessing

from DVR_Algorithm import auto_configure_dvr, get_fully_converged_energy_levels
from HO_Analytical import analytic_energy_levels_HO, analytic_cv_HO_classical
from HO_Energy_Level_Error import (
    compute_energy_level_errors,
    plot_energy_level_comparison,
    plot_energy_level_error,
    print_accuracy_summary,
)
from Quantum_Classical_Combined import run as run_general_cv_pipeline
from HO_Benchmark import run_ho_benchmark
from DVR_Limit_Finder import run_dvr_limit_analysis


# =====================================================================
# Lightweight alive-indicator (replaces multiprocessing DisappearingTimer)
# =====================================================================
class SimpleTimer:
    """
    Daemon thread that prints elapsed time every `interval` seconds
    to confirm the pipeline is still running.

    Why a thread instead of a process?
    The old DisappearingTimer (DVR_Algorithm ≤ 1.3) used a spawned
    process to avoid GIL starvation from LAPACK. That was necessary
    when the timer lived *inside* the DVR solver where it ran
    concurrently with eigvalsh. Here the timer wraps an entire
    pipeline section at the Master-file level, printing *between*
    LAPACK calls rather than during them -- a simple daemon thread
    with a 10 s sleep interval is sufficient and has negligible
    overhead. Even if LAPACK holds the GIL past one tick, the thread
    will print as soon as control returns, so at most one interval
    is missed.

    Parameters
    ----------
    label : str
        Short description shown in each tick line.
    interval : float, optional
        Seconds between ticks (default 10).

    Usage
    -----
    with SimpleTimer("Section 1: DVR solve"):
        energies = get_fully_converged_energy_levels(...)
    # prints "✓  [Section 1: DVR solve] done in 14.2s" on exit
    """

    def __init__(self, label="Running", interval=10):
        self.label = label
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
        self._start = None

    def _run(self):
        """Thread target: print a tick line every `interval` seconds."""
        tick = 0
        while not self._stop.wait(timeout=self.interval):
            tick += 1
            elapsed = time.time() - self._start
            print(f"  \u23f1  [{self.label}] still running ... {elapsed:.0f}s", flush=True)

    def __enter__(self):
        self._start = time.time()
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        elapsed = time.time() - self._start
        print(f"  \u2713  [{self.label}] done in {elapsed:.1f}s", flush=True)


# =====================================================================
# Entry point
# =====================================================================
if __name__ == "__main__":
    # Required for safe multiprocessing on Windows/macOS (used by tqdm
    # internals and any future multiprocessing calls).
    multiprocessing.freeze_support()

    # =================================================================
    # SECTION 0 -- Global configuration
    # Edit this section to change system parameters, temperature range,
    # convergence tolerances, or the limit-analysis settings.
    # =================================================================

    # --- Physical constants (dimensionless units: ℏ = m = ω = k_B = 1) ---
    MASS, HBAR, OMEGA, KB = 1.0, 1.0, 1.0, 1.0

    # --- Number of DVR energy levels to compute ---
    # 500 levels comfortably covers the hardest (β, ξ) combination in the
    # sweep below: the n-convergence diagnostic will confirm exactly how
    # many are actually needed, typically well below 500 for the HO.
    # See the response note in the project changelog for why fewer levels
    # are needed here than in earlier project iterations.
    NUM_STATES = 500

    # --- Temperature (β) sweep for the Cv pipeline ---
    # β_max = 50 probes very cold temperatures (T = 0.02 in ℏω/k_B units).
    # At such cold β the system is deep in the quantum ground state and
    # the classical limit is not reachable -- the ξ-convergence scan will
    # correctly report failure there, which is physically expected.
    BETA_MIN, BETA_MAX, N_BETA = 0.1, 50.0, 500

    # --- ξ / n convergence parameters ---
    XI_START      = 3.0    # initial scaling factor for the ξ scan
    TOL_XI        = 5e-3   # |ΔCv| plateau tolerance for ξ convergence
    MIN_STABLE_XI = 3      # minimum consecutive stable steps in the ξ scan
    XI_MULT       = 1.1    # geometric growth factor per ξ step
    MAX_XI_STEPS  = 80     # safety cap on ξ scan length
    TOL_CV        = 1e-4   # |ΔCv| tolerance for n convergence
    MIN_STABLE_N  = 3      # minimum consecutive stable steps in the n scan

    # --- DVR limit analysis tolerance ---
    LIMIT_TOLERANCE = 1e-6

    # --- The Harmonic Oscillator potential ---
    def ho_potential(x):
        """V(x) = ½ m ω² x²  (smooth, finite everywhere -- suitable for this DVR engine)."""
        return 0.5 * MASS * (OMEGA ** 2) * x ** 2

    # =================================================================
    # SECTION 1 -- DVR: auto-configure grid and compute energy levels
    # =================================================================
    print("\n" + "=" * 60)
    print("  SECTION 1: DVR energy levels")
    print("=" * 60)
    x_min, x_max, n_grid = auto_configure_dvr(
        ho_potential, NUM_STATES, mass=MASS, hbar=HBAR
    )
    with SimpleTimer("Section 1: DVR 3-pass convergence check"):
        energies_numeric = get_fully_converged_energy_levels(
            potential_func=ho_potential, num_levels=NUM_STATES,
            x_min=x_min, x_max=x_max, num_points=n_grid,
            mass=MASS, hbar=HBAR,
        )

    # =================================================================
    # SECTION 2 -- Analytic HO energy levels (exact ground truth)
    # =================================================================
    # E_n = ℏω(n + ½),  n = 0, 1, ..., NUM_STATES-1
    energies_analytic = analytic_energy_levels_HO(NUM_STATES, hbar=HBAR, omega=OMEGA)

    # =================================================================
    # SECTION 3 -- Energy-level accuracy: DVR vs analytic, level by level
    # =================================================================
    print("\n" + "=" * 60)
    print("  SECTION 3: Energy-level accuracy check")
    print("=" * 60)
    energy_error = compute_energy_level_errors(energies_numeric, energies_analytic)
    print_accuracy_summary(energy_error, NUM_STATES, system_name="Harmonic Oscillator")
    # Plot 1: full spectrum + zoom on worst-error state
    plot_energy_level_comparison(
        energies_numeric, energies_analytic,
        error_result=energy_error, zoom=True,
        system_name="Harmonic Oscillator",
    )
    # Plot 2: absolute and relative error vs state index n (log y-axis)
    plot_energy_level_error(energy_error, system_name="Harmonic Oscillator")

    # =================================================================
    # SECTION 4 -- General Cv pipeline: quantum Cv(T) + numerical
    #              classical limit + ξ/n convergence diagnostics
    # =================================================================
    print("\n" + "=" * 60)
    print("  SECTION 4: Cv(T) pipeline")
    print("=" * 60)
    with SimpleTimer("Section 4: Cv T-range sweep"):
        numeric_results = run_general_cv_pipeline(
            energies=energies_numeric,
            system_name="1-D Harmonic Oscillator",
            beta_min=BETA_MIN, beta_max=BETA_MAX, n_beta=N_BETA,
            xi_start=XI_START, tol_xi=TOL_XI,
            min_stable_xi=MIN_STABLE_XI,
            xi_multiplier=XI_MULT, max_xi_steps=MAX_XI_STEPS,
            tol_cv=TOL_CV, min_stable_n=MIN_STABLE_N,
            cv_analytic=analytic_cv_HO_classical(kB=KB),
            T_units_label=r"$k_B T / \hbar\omega$",
        )

    # =================================================================
    # SECTION 5 -- HO benchmark: numerical vs analytical Cv(T) overlay
    #              with a quantitative error curve
    # =================================================================
    print("\n" + "=" * 60)
    print("  SECTION 5: Numerical vs analytical benchmark")
    print("=" * 60)
    benchmark_results = run_ho_benchmark(
        numeric_results, hbar=HBAR, omega=OMEGA, kB=KB
    )

    # =================================================================
    # SECTION 6 -- DVR limit analysis: where does accuracy break down?
    #   Search A (dx sweep): minimum grid spacing for NUM_STATES levels
    #   Search B (n sweep):  maximum trustworthy n for the same grid
    # The reference spectrum here is HO_Analytical's exact energies.
    # For a future system with no analytic solution, replace this with
    # a separately verified, much finer numerical DVR run.
    # =================================================================
    print("\n" + "=" * 60)
    print("  SECTION 6: DVR limit analysis")
    print("=" * 60)
    reference_for_limits = analytic_energy_levels_HO(n_grid, hbar=HBAR, omega=OMEGA)
    with SimpleTimer("Section 6: DVR limit searches"):
        limit_results = run_dvr_limit_analysis(
            potential_func=ho_potential,
            system_name="1-D Harmonic Oscillator",
            reference_energies=reference_for_limits,
            num_levels_for_grid_search=NUM_STATES,
            x_min=x_min, x_max=x_max,
            num_points_for_level_search=n_grid,
            tolerance=LIMIT_TOLERANCE,
            metric="max_abs",
            mass=MASS, hbar=HBAR,
        )
