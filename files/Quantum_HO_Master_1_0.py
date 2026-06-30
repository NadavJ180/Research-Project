"""
Quantum_HO_Master_1_0.py
=====================================================================
WHAT THIS FILE DOES
---------------------------------------------------------------------
Single entry-point driver for the Harmonic Oscillator benchmark.
This file does NOT define any new physics or numerics itself -- it
only imports and calls functions from the other project files, in
clearly separated sections, so each piece can keep evolving
independently without this file needing structural changes:

    SECTION 1 -- DVR setup & diagonalization      (DVR_Algorithm_1_3)
    SECTION 2 -- Analytic HO energy levels         (HO_Analytical_1_0)
    SECTION 3 -- Energy-level accuracy check       (HO_Energy_Level_Error_1_0)
    SECTION 4 -- General Cv pipeline (numerical)   (Quantum_Classical_Combined_1_9)
    SECTION 5 -- Numerical vs analytical benchmark (HO_Benchmark_1_0)

Run this file directly (`python Quantum_HO_Master_1_0.py`) to execute
the full pipeline end-to-end and produce every plot described in the
project spec: numerical-vs-analytic HO energy levels (with a zoom on
the largest-error region), the energy-level error curve, the general
quantum/classical Cv(T) diagnostics, and the final numerical-vs-
analytical Cv(T) + classical-limit benchmark with its own error curve.

CHANGELOG (NEW FILE, v1.0)
---------------------------------------------------------------------
- New file. Replaces the old pattern of hard-coding systems at the
  bottom of Quantum_Classical_Combined_*.py's __main__ block. The HO
  system now lives entirely here, as a thin orchestration layer over
  the dedicated modules, leaving every other file free of
  system-specific wiring.
=====================================================================
"""

import multiprocessing

from DVR_Algorithm_1_3 import auto_configure_dvr, get_fully_converged_energy_levels
from HO_Analytical_1_0 import analytic_energy_levels_HO, analytic_cv_HO_classical
from HO_Energy_Level_Error_1_0 import (
    compute_energy_level_errors,
    plot_energy_level_comparison,
    plot_energy_level_error,
    print_accuracy_summary,
)
from Quantum_Classical_Combined_1_9 import run as run_general_cv_pipeline
from HO_Benchmark_1_0 import run_ho_benchmark


if __name__ == "__main__":
    # Required for safe multiprocessing behavior across OS environments (Windows/macOS).
    multiprocessing.freeze_support()

    # =================================================================
    # SECTION 0 -- Global configuration
    # =================================================================
    # Physical constants for the Harmonic Oscillator (dimensionless units).
    MASS, HBAR, OMEGA, KB = 1.0, 1.0, 1.0, 1.0

    # Number of energy levels to compute. Kept modest here so the full
    # benchmark runs quickly; raise it for a more thorough DVR accuracy
    # sweep (the highest states will show the most truncation error).
    NUM_STATES = 1000

    # Temperature/beta sweep range for the Cv pipeline.
    BETA_MIN, BETA_MAX, N_BETA = 0.05, 50.0, 1000

    # xi/n convergence-search parameters for the numerical classical limit.
    XI_START, TOL_XI, MIN_STABLE_XI, XI_MULT, MAX_XI_STEPS = 3.0, 5e-3, 3, 1.1, 80
    TOL_CV, MIN_STABLE_N = 1e-4, 3

    # The Harmonic Oscillator potential itself: V(x) = 0.5 * m * omega^2 * x^2.
    def ho_potential(x):
        return 0.5 * MASS * (OMEGA**2) * x**2

    # =================================================================
    # SECTION 1 -- DVR setup & diagonalization (numerical energy levels)
    # =================================================================
    x_min, x_max, n_grid = auto_configure_dvr(ho_potential, NUM_STATES, mass=MASS, hbar=HBAR)
    energies_numeric = get_fully_converged_energy_levels(
        potential_func=ho_potential, num_levels=NUM_STATES,
        x_min=x_min, x_max=x_max, num_points=n_grid,
        mass=MASS, hbar=HBAR,
    )

    # =================================================================
    # SECTION 2 -- Analytic HO energy levels (ground truth)
    # =================================================================
    energies_analytic = analytic_energy_levels_HO(NUM_STATES, hbar=HBAR, omega=OMEGA)

    # =================================================================
    # SECTION 3 -- Energy-level accuracy check (DVR vs analytic)
    # =================================================================
    energy_error = compute_energy_level_errors(energies_numeric, energies_analytic)
    print_accuracy_summary(energy_error, NUM_STATES, system_name="Harmonic Oscillator")
    plot_energy_level_comparison(energies_numeric, energies_analytic, error_result=energy_error,
                                  zoom=True, system_name="Harmonic Oscillator")
    plot_energy_level_error(energy_error, system_name="Harmonic Oscillator")

    # =================================================================
    # SECTION 4 -- General Cv pipeline (numerical quantum Cv + numerical
    #              classical limit + xi/n convergence diagnostics)
    # =================================================================
    numeric_results = run_general_cv_pipeline(
        energies=energies_numeric, system_name="1-D Harmonic Oscillator",
        beta_min=BETA_MIN, beta_max=BETA_MAX, n_beta=N_BETA,
        xi_start=XI_START, tol_xi=TOL_XI, min_stable_xi=MIN_STABLE_XI,
        xi_multiplier=XI_MULT, max_xi_steps=MAX_XI_STEPS,
        tol_cv=TOL_CV, min_stable_n=MIN_STABLE_N,
        cv_analytic=analytic_cv_HO_classical(kB=KB),
        T_units_label=r"$k_B T / \hbar\omega$",
    )

    # =================================================================
    # SECTION 5 -- Numerical vs analytical benchmark (Cv(T) + classical
    #              limit overlay, with quantitative error curve)
    # =================================================================
    benchmark_results = run_ho_benchmark(numeric_results, hbar=HBAR, omega=OMEGA, kB=KB)
