"""
config.py
=====================================================================
Single source of truth for the current run's physical setup: the
potential, its constants, and the numerical control parameters.

This is exactly "Section 0" of Quantum_HO_Master.py, pulled out into
its own module so that OTHER scripts (the figures/ tools, tests, a
notebook, ...) can import the potential / bounds / parameters WITHOUT
importing Quantum_HO_Master.py itself -- which would otherwise run
the full six-section pipeline (DVR solve, reference generation, Cv
sweep, ...) as a side effect of the import.

To switch systems: edit `my_potential`, `SYSTEM_NAME`, and
`T_UNITS_LABEL` below (swap which block is commented out), exactly as
before. Nothing else in the repository needs to change.
=====================================================================
"""

# --- Physical constants (dimensionless: ℏ = m = ω = k_B = 1) ---
MASS, HBAR, OMEGA = 1.0, 1.0, 1.0

# --- Potential function (swap this for any smooth V(x)) ---
r'''
def my_potential(x):
    """1-D harmonic oscillator: V(x) = ½ m ω² x²."""
    return 0.5 * MASS * (OMEGA**2) * x**2

SYSTEM_NAME   = "1-D Harmonic Oscillator"
T_UNITS_LABEL = r"$k_B T \,/\, \hbar\omega$"
'''

def my_potential(x):
        """1-D symmetric double well: V(x) = 1/4 x^4 - 1/2 x^2."""
        a, b, c, d = 0.25, -0.5, -0.5, 0
        return a * (x ** 4) + b * (x ** 3) + c * (x **2) + d * x

SYSTEM_NAME   = "1-D symmetric double well"
T_UNITS_LABEL = r"$k_B T \,/\, \hbar\omega$"

# --- DVR base grid: number of energy levels ---
# The n-convergence diagnostic (Section 4) will confirm the exact
# number needed; 500 gives comfortable headroom for this system.
NUM_STATES = 500

# --- Temperature sweep ---
BETA_MIN, BETA_MAX, N_BETA = 0.01, 50.0, 1000

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