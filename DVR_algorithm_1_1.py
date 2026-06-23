import numpy as np
import scipy.linalg as la
import warnings
import sys
import time
import threading

"""
CHANGELOG (v1.1):
- Refactored DisappearingTimer to allow dynamic text updates for multi-pass tracking.
- Moved the timer out of the core DVR function and into the convergence verification suite.
- Replaced the costly (-1.0)**idx calculation with a much faster O(N) array slicing operation.
"""

class DisappearingTimer:
    """A background threaded timer that prints elapsed time and updates its message dynamically."""
    def __init__(self, message="Running..."):
        self.message = message.ljust(50)
        self.start_time = None
        self._stop_event = threading.Event()
        self._thread = None

    def update_text(self, new_text):
        self.message = new_text.ljust(50)

    def _run(self):
        while not self._stop_event.is_set():
            elapsed = time.time() - self.start_time
            sys.stdout.write(f"\r{self.message} [{elapsed:.1f}s]")
            sys.stdout.flush()
            self._stop_event.wait(0.1)
        # Clear the line cleanly when done
        sys.stdout.write("\r" + " " * (len(self.message) + 20) + "\r")
        sys.stdout.flush()

    def __enter__(self):
        self.start_time = time.time()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_event.set()
        if self._thread:
            self._thread.join()


def colbert_miller_dvr_1d(potential_func, num_levels, x_min, x_max, num_points, mass=1.0, hbar=1.0):
    if num_points <= num_levels:
        raise ValueError(f"Number of grid points ({num_points}) must be greater than requested levels ({num_levels}).")
    
    x = np.linspace(x_min, x_max, num_points)
    dx = x[1] - x[0]
    k_factor = (hbar**2) / (2.0 * mass * dx**2)
    
    # 1. Build only the first row/column 
    idx = np.arange(num_points, dtype=float)
    
    # [Optimization]: Generate alternating signs instantly without costly powers
    alter_sign = np.ones(num_points)
    alter_sign[1::2] = -1.0
    
    with np.errstate(divide='ignore', invalid='ignore'):
        first_row = k_factor * 2.0 * alter_sign / (idx**2)
    
    # Overwrite diagonal elements with analytical limit
    first_row[0] = k_factor * (np.pi**2) / 3.0
    
    # 2. Construct the full symmetric Toeplitz Matrix cleanly
    H = la.toeplitz(first_row)
    
    # 3. Apply potential strictly along the diagonal
    v_diag = potential_func(x)
    if np.any(np.isinf(v_diag)):
        raise ValueError("Potential contains infinite values on grid. Shift boundaries inward.")
    
    H[np.diag_indices(num_points)] += v_diag
    
    # 4. LAPACK Optimized solve using MRRR
    eigenvalues = la.eigvalsh(H, overwrite_a=True, subset_by_index=[0, num_levels - 1])
    
    return eigenvalues


def get_fully_converged_energy_levels(potential_func, num_levels, x_min, x_max, num_points, 
                                      mass=1.0, hbar=1.0, tolerance=1e-5):
    
    with DisappearingTimer("  [DVR] Pass 1/3: Calculating Base Grid...") as timer:
        E_base = colbert_miller_dvr_1d(potential_func, num_levels, x_min, x_max, num_points, mass, hbar)
        
        timer.update_text("  [DVR] Pass 2/3: Checking Resolution Limit...")
        num_points_finer = int(num_points * 1.3)
        E_res = colbert_miller_dvr_1d(potential_func, num_levels, x_min, x_max, num_points_finer, mass, hbar)
        res_error = np.max(np.abs(E_base - E_res))
        
        timer.update_text("  [DVR] Pass 3/3: Checking Boundary Span...")
        span = x_max - x_min
        x_min_wider = x_min - 0.1 * span
        x_max_wider = x_max + 0.1 * span
        num_points_wider = int(num_points * 1.2)
        
        E_span = colbert_miller_dvr_1d(potential_func, num_levels, x_min_wider, x_max_wider, num_points_wider, mass, hbar)
        span_error = np.max(np.abs(E_base - E_span))
    
    if res_error > tolerance or span_error > tolerance:
        raise RuntimeError(f"DVR CONVERGENCE ERROR\nResolution Error: {res_error:.2e}\nSpan Error: {span_error:.2e}")
        
    return E_base


def compute_quantum_heat_capacity(E, beta, k_B=1.0):
    E_shifted = E - E[0]
    weights = np.exp(-beta * E_shifted)
    Z = np.sum(weights)
    
    highest_state_occupancy = weights[-1] / Z
    if highest_state_occupancy > 1e-4:
        warnings.warn("Thermodynamic Truncation Threat: Heat capacity will collapse toward zero!", UserWarning)
        
    mean_E = np.sum(E * weights) / Z
    mean_E_sq = np.sum((E**2) * weights) / Z
    variance_E = mean_E_sq - (mean_E**2)
    
    return k_B * (beta**2) * variance_E