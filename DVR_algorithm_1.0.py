import numpy as np
import scipy.linalg as la
import warnings

"""
This module implements the 1D Colbert-Miller DVR algorithm to numerically compute 
quantum energy eigenvalues for arbitrary potentials.

Exportable Functions:
- get_fully_converged_energy_levels: Takes a potential function, grid bounds, points, 
  and desired levels. Outputs a validated eigenvalue array.
- compute_quantum_heat_capacity: Computes partition functions and heat capacity (C_p).

Error Limits:
Smooth potentials (e.g., Harmonic Oscillator) reach near-machine precision (~10^-12). 
Hard-wall bounds (e.g., Particle-in-a-Box) exhibit higher error (~10^-5) due to 
Gibbs-like high-frequency oscillations induced by abrupt potential steps. Dual-axis 
verification is included to validate grid resolution and spatial span.
"""

def colbert_miller_dvr_1d(potential_func, num_levels, x_min, x_max, num_points, mass=1.0, hbar=1.0):
    """
    Core implementation of the 1D Colbert-Miller Discrete Variable Representation (DVR).
    """
    if num_points <= num_levels:
        raise ValueError(f"Number of grid points ({num_points}) must be greater than requested levels ({num_levels}).")
    
    # Generate the uniform coordinate grid
    x = np.linspace(x_min, x_max, num_points)
    dx = x[1] - x[0]
    
    # Kinetic energy factor: hbar^2 / (2 * m * dx^2)
    k_factor = (hbar**2) / (2.0 * mass * dx**2)
    
    # Vectorized construction of index distances (i - j)
    idx = np.arange(num_points)
    diff = idx[:, None] - idx[None, :]
    
    # Compute off-diagonal elements safely, ignoring the temporary division by zero on diagonal
    with np.errstate(divide='ignore', invalid='ignore'):
        T = k_factor * 2.0 * ((-1.0)**diff) / (diff**2)
        
    # Overwrite diagonal elements with analytical limit
    np.fill_diagonal(T, k_factor * (np.pi**2) / 3.0)
    
    # Construct Potential Energy Matrix and screen for infinite values
    v_diag = potential_func(x)
    if np.any(np.isinf(v_diag)):
        raise ValueError(
            "Potential array contains infinite values on the grid coordinates. "
            "For hard-wall potentials like Particle-in-a-Box, shift your grid boundaries "
            "slightly inside the box to avoid evaluation on the wall edges."
        )
        
    H = T + np.diag(v_diag)
    
    # Solve symmetric eigenvalue problem
    eigenvalues = la.eigh(H, eigvals_only=True)
    
    return eigenvalues[:num_levels]


def get_fully_converged_energy_levels(potential_func, num_levels, x_min, x_max, num_points, 
                                      mass=1.0, hbar=1.0, tolerance=1e-5):
    """
    Validates DVR convergence across BOTH physical criteria:
    1. Grid Resolution (dx) -> Evaluated via a 30% finer grid density.
    2. Grid Span (Box Size)  -> Evaluated via a 20% wider coordinate range.
    """
    # 1. Baseline calculation
    E_base = colbert_miller_dvr_1d(potential_func, num_levels, x_min, x_max, num_points, mass, hbar)
    
    # 2. Test Grid Resolution (Density check)
    num_points_finer = int(num_points * 1.3)
    E_res = colbert_miller_dvr_1d(potential_func, num_levels, x_min, x_max, num_points_finer, mass, hbar)
    res_error = np.max(np.abs(E_base - E_res))
    
    # 3. Test Grid Span (Box size check)
    span = x_max - x_min
    x_min_wider = x_min - 0.1 * span
    x_max_wider = x_max + 0.1 * span
    num_points_wider = int(num_points * 1.2) # Keeps grid density roughly equal
    
    E_span = colbert_miller_dvr_1d(potential_func, num_levels, x_min_wider, x_max_wider, num_points_wider, mass, hbar)
    span_error = np.max(np.abs(E_base - E_span))
    
    # Enforce strict checks on both sources of error
    if res_error > tolerance or span_error > tolerance:
        error_msg = (
            f"\n"
            f"=========================================================================\n"
            f"DVR CONVERGENCE ERROR: The calculation failed physical validation limits.\n"
            f"=========================================================================\n"
            f"Grid Resolution Error (dx stability): {res_error:.2e} (Allowed Tol: {tolerance:.2e})\n"
            f"Grid Span Error (Box edge clipping):   {span_error:.2e} (Allowed Tol: {tolerance:.2e})\n\n"
            f"Fixes:\n"
            f"1. If Resolution Error is high -> Increase your initial 'num_points'.\n"
            f"2. If Grid Span Error is high   -> Expand your physical range [x_min, x_max]\n"
            f"   further into the classically forbidden region.\n"
            f"=========================================================================\n"
        )
        raise RuntimeError(error_msg)
        
    return E_base


def compute_quantum_heat_capacity(E, beta, k_B=1.0):
    """
    Computes Cp = k_B * beta^2 * Var(E) while screening for truncation errors.
    """
    # Energy shift avoids numerical overflow issues in np.exp()
    E_shifted = E - E[0]
    weights = np.exp(-beta * E_shifted)
    Z = np.sum(weights)
    
    # Verify if the highest calculated level is significantly populated
    highest_state_occupancy = weights[-1] / Z
    if highest_state_occupancy > 1e-4:
        warnings.warn(
            f"Thermodynamic Truncation Threat: The highest calculated state holds {highest_state_occupancy:.2e} "
            f"of the ensemble population. At this high temperature (beta={beta}), your heat capacity "
            f"will artificially collapse toward zero! Increase 'num_levels' for this run.",
            UserWarning
        )
        
    mean_E = np.sum(E * weights) / Z
    mean_E_sq = np.sum((E**2) * weights) / Z
    variance_E = mean_E_sq - (mean_E**2)
    
    C_p = k_B * (beta**2) * variance_E
    return C_p