"""
Quantum_HO_Master_1_5.py
=====================================================================
WHAT THIS FILE DOES
---------------------------------------------------------------------
Fully numerical entry-point driver for the Harmonic Oscillator
(or any smooth potential). No analytical solutions, formulas, or
closed-form references are used anywhere in this file or in the
modules it calls. The only ground truth is a second, higher-quality
DVR computation on a finer/wider grid.

    SECTION 0 -- Configuration            (all parameters here)
    SECTION 1 -- DVR base computation     (DVR_Algorithm_1_4)
    SECTION 2 -- Numerical reference      (DVR_Reference_Generator_1_0)
    SECTION 3 -- Energy-level accuracy    (HO_Energy_Level_Error_1_1)
                 base DVR vs reference DVR, level by level
    SECTION 4 -- Cv pipeline              (Quantum_Classical_Combined_1_9)
                 quantum Cv(T) + numerical classical limit on base energies
    SECTION 5 -- DVR limit analysis       (DVR_Limit_Finder_1_2)
                 minimum dx and maximum n, both checked vs reference
    SECTION 6 -- Cv numerical benchmark   (Cv_Numerical_Benchmark_1_0)
                 quantum Cv and classical limit: base vs reference

GENERALITY
---------------------------------------------------------------------
Every function called here accepts any smooth V(x). To run this
pipeline on a different potential, change `my_potential` and the
label strings in Section 0. Nothing else needs to change.

CHANGELOG (v1.4 -> v1.5)
---------------------------------------------------------------------
- REMOVED all analytical sections: no HO_Analytical_1_0,
  no HO_Benchmark_1_1, no analytic reference anywhere.
- Numerical reference (formerly Section 7a) is now Section 2 so
  it is generated once and shared by Sections 3, 5, and 6.
- Section 3: energy-level comparison now uses numerical reference.
- Section 4: Cv pipeline no longer overlays an analytic classical
  limit curve (cv_analytic=None); the numerical classical limit
  from the xi/n sweep is the only curve shown.
- Section 5: DVR limit finder now uses the numerical reference as
  ground truth (was analytic_energy_levels_HO).
- Section 6: Cv numerical benchmark (quantum + classical limit)
  is the sole benchmark; the analytic benchmark is gone.
- XI_START remains 3.0 (set in v1.4).
=====================================================================
"""

import threading
import time
import multiprocessing

from DVR.DVR_Algorithm              import auto_configure_dvr, get_fully_converged_energy_levels
from error.error_energylevels      import (compute_energy_level_errors,
                                           plot_energy_level_comparison,
                                           plot_energy_level_error,
                                           print_accuracy_summary)
from Quantum_Classical_Combined import run as run_general_cv_pipeline
from DVR.DVR_Limit_Finder           import run_dvr_limit_analysis
from DVR.DVR_Reference_Generator    import generate_reference_energies
from Cv_Numerical_Benchmark     import run_cv_numerical_benchmark


# =====================================================================
# Lightweight alive-indicator (daemon thread, prints every interval s)
# =====================================================================
class SimpleTimer:
    """
    Daemon thread that prints elapsed time every `interval` seconds
    so it is obvious the pipeline is still running during long solves.

    Parameters
    ----------
    label    : str   -- short description shown in each tick
    interval : float -- seconds between ticks (default 10)
    """
    def __init__(self, label="Running", interval=10):
        self.label    = label
        self.interval = interval
        self._stop    = threading.Event()
        self._thread  = None
        self._start   = None

    def _run(self):
        while not self._stop.wait(timeout=self.interval):
            print(f"  \u23f1  [{self.label}] still running ... "
                  f"{time.time()-self._start:.0f}s", flush=True)

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
        print(f"  \u2713  [{self.label}] done in "
              f"{time.time()-self._start:.1f}s", flush=True)


# =====================================================================
if __name__ == "__main__":
    multiprocessing.freeze_support()

    # =================================================================
    # SECTION 0 -- Configuration
    # This is the ONLY section that needs editing when changing system,
    # parameters, or reference scaling.
    # =================================================================

    # --- Physical constants (dimensionless: ℏ = m = ω = k_B = 1) ---
    MASS, HBAR, OMEGA = 1.0, 1.0, 1.0

    # --- Potential function (swap this for any smooth V(x)) ---
    def my_potential(x):
        """1-D harmonic oscillator: V(x) = ½ m ω² x²."""
        return 0.5 * MASS * (OMEGA**2) * x**2

    SYSTEM_NAME   = "1-D Harmonic Oscillator"
    T_UNITS_LABEL = r"$k_B T \,/\, \hbar\omega$"

    # --- DVR base grid: number of energy levels ---
    # The n-convergence diagnostic (Section 4) will confirm the exact
    # number needed; 500 gives comfortable headroom for this system.
    NUM_STATES = 2500

    # --- Temperature sweep ---
    BETA_MIN, BETA_MAX, N_BETA = 0.1, 50.0, 500

    # --- xi / n convergence parameters ---
    # XI_START = 3.0: first probe is already at effective T/9, allowing
    # the classical-limit plateau to be found at much colder temperatures
    # than XI_START = 1.0 would permit.
    XI_START      = 3.0
    TOL_XI        = 5e-3
    MIN_STABLE_XI = 3
    XI_MULT       = 1.1
    MAX_XI_STEPS  = 80
    TOL_CV        = 1e-4
    MIN_STABLE_N  = 3

    # --- DVR limit analysis tolerance (Section 5) ---
    LIMIT_TOLERANCE = 1e-6

    # --- Numerical reference scaling (Sections 2, 3, 5, 6) ---
    # REFERENCE_SPAN_FACTOR: multiply base span by this (2.0 = double L)
    # REFERENCE_DX_FACTOR:   divide base dx   by this (2.0 = halve Δx)
    # Set INTERACTIVE = True to be prompted at runtime instead.
    INTERACTIVE_REFERENCE_SCALING = False
    REFERENCE_SPAN_FACTOR         = 2.0
    REFERENCE_DX_FACTOR           = 2.0

    # Human-readable label built from the scaling factors (used in plots).
    ref_label = (f"numerical reference  "
                 f"(span\u00d7{REFERENCE_SPAN_FACTOR:.2g}, "
                 f"dx\u00f7{REFERENCE_DX_FACTOR:.2g})")

    # =================================================================
    # SECTION 1 -- DVR base computation
    # Auto-configure a grid for NUM_STATES levels, run the 3-pass
    # convergence-checked solve, return the base energy spectrum.
    # =================================================================
    print("\n" + "="*60)
    print(f"  SECTION 1 — DVR base computation  ({SYSTEM_NAME})")
    print("="*60)

    x_min, x_max, n_grid = auto_configure_dvr(
        my_potential, NUM_STATES, mass=MASS, hbar=HBAR
    )

    with SimpleTimer("Section 1: DVR 3-pass convergence check"):
        energies_base = get_fully_converged_energy_levels(
            potential_func=my_potential,
            num_levels=NUM_STATES,
            x_min=x_min, x_max=x_max, num_points=n_grid,
            mass=MASS, hbar=HBAR,
        )

    # =================================================================
    # SECTION 2 -- Numerical reference generation
    # Run the DVR on a finer/wider grid to produce the high-precision
    # reference spectrum used as ground truth throughout Sections 3-6.
    # The reference is computed ONCE here and reused everywhere.
    #
    # NOTE on level count for Section 5 (DVR limit analysis):
    # The level-count search can test up to min(n_grid-2, NUM_STATES)
    # levels. If you want to probe beyond NUM_STATES in the limit search,
    # increase NUM_STATES or generate a separate reference with more
    # levels specifically for Section 5.
    # =================================================================
    print("\n" + "="*60)
    print("  SECTION 2 — Numerical reference generation")
    print("="*60)

    with SimpleTimer("Section 2: reference DVR solve"):
        reference_result = generate_reference_energies(
            my_potential, NUM_STATES,
            x_min, x_max, n_grid,
            span_factor=REFERENCE_SPAN_FACTOR,
            dx_factor=REFERENCE_DX_FACTOR,
            mass=MASS, hbar=HBAR,
            interactive=INTERACTIVE_REFERENCE_SCALING,
            verbose=True,
        )

    energies_ref = reference_result["energies"]

    # =================================================================
    # SECTION 3 -- Energy-level accuracy: base DVR vs numerical reference
    # Compares the two spectra level-by-level and plots the error.
    # Uses the generic functions from Energy_Level_Error which
    # only ever compare two plain arrays -- no system-specific logic.
    # =================================================================
    print("\n" + "="*60)
    print("  SECTION 3 — Energy-level accuracy (base DVR vs reference)")
    print("="*60)

    energy_error = compute_energy_level_errors(
        energies_base,
        energies_ref[:NUM_STATES],
    )
    print_accuracy_summary(
        energy_error, NUM_STATES,
        system_name=f"{SYSTEM_NAME}  [{ref_label}]",
    )
    # Plot 1: full spectrum + zoom on worst-error state
    plot_energy_level_comparison(
        energies_base, energies_ref[:NUM_STATES],
        error_result=energy_error, zoom=True,
        system_name=f"{SYSTEM_NAME} [{ref_label}]",
    )
    # Plot 2: absolute and relative error vs state index n (log y-axis)
    plot_energy_level_error(
        energy_error,
        system_name=f"{SYSTEM_NAME} — base DVR vs {ref_label}",
    )

    # =================================================================
    # SECTION 4 -- Cv pipeline
    # Computes quantum Cv(T) directly from the base energy spectrum,
    # and finds the numerical classical-limit Cv(T) via the xi/n
    # convergence sweep. Produces xi-convergence diagnostic,
    # n-convergence diagnostic, and the combined Cv(T) summary plot.
    # cv_analytic=None: no analytic overlay; only the numerical curves.
    # =================================================================
    print("\n" + "="*60)
    print("  SECTION 4 — Cv pipeline (base DVR energies)")
    print("="*60)

    with SimpleTimer("Section 4: Cv T-range sweep"):
        base_cv_results = run_general_cv_pipeline(
            energies=energies_base,
            system_name=SYSTEM_NAME,
            beta_min=BETA_MIN, beta_max=BETA_MAX, n_beta=N_BETA,
            xi_start=XI_START, tol_xi=TOL_XI,
            min_stable_xi=MIN_STABLE_XI,
            xi_multiplier=XI_MULT, max_xi_steps=MAX_XI_STEPS,
            tol_cv=TOL_CV, min_stable_n=MIN_STABLE_N,
            cv_analytic=None,          # no analytic overlay
            T_units_label=T_UNITS_LABEL,
        )

    # =================================================================
    # SECTION 5 -- DVR limit analysis (numerical reference as truth)
    # Search A: sweep dx at fixed n → maximum safe Δx for NUM_STATES levels
    # Search B: sweep n at fixed grid → maximum trustworthy n for this Δx
    # Both searches compare against the numerical reference (Section 2).
    # =================================================================
    print("\n" + "="*60)
    print("  SECTION 5 — DVR limit analysis (vs numerical reference)")
    print("="*60)

    with SimpleTimer("Section 5: DVR limit searches"):
        limit_results = run_dvr_limit_analysis(
            potential_func=my_potential,
            system_name=SYSTEM_NAME,
            reference_energies=energies_ref,
            num_levels_for_grid_search=NUM_STATES,
            x_min=x_min, x_max=x_max,
            num_points_for_level_search=n_grid,
            tolerance=LIMIT_TOLERANCE,
            metric="max_abs",
            mass=MASS, hbar=HBAR,
        )

    # =================================================================
    # SECTION 6 -- Cv numerical benchmark
    # Runs the FULL Cv pipeline (quantum Cv + classical limit sweep)
    # on the reference energies, then compares both curves against
    # the base results from Section 4. Produces:
    #   Figure 1: quantum Cv(T) base vs reference + error panel
    #   Figure 2: classical limit Cv(T) base vs reference + error panel
    # =================================================================
    print("\n" + "="*60)
    print("  SECTION 6 — Cv numerical benchmark (base vs reference)")
    print("="*60)

    with SimpleTimer("Section 6: reference Cv sweep"):
        cv_benchmark_results = run_cv_numerical_benchmark(
            base_cv_results=base_cv_results,
            reference_energies=energies_ref,
            beta_arr=base_cv_results["beta_arr"],
            system_name=SYSTEM_NAME,
            reference_label=ref_label,
            xi_start=XI_START, tol_xi=TOL_XI,
            min_stable_xi=MIN_STABLE_XI,
            xi_multiplier=XI_MULT, max_xi_steps=MAX_XI_STEPS,
            tol_cv=TOL_CV, min_stable_n=MIN_STABLE_N,
            T_units_label=T_UNITS_LABEL,
        )
