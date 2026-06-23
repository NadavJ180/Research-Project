import numpy as np
import scipy.linalg as la
import warnings
import sys
import time
import multiprocessing

"""
CHANGELOG (v1.2):
- Upgraded DisappearingTimer from threading.Thread to multiprocessing.Process.
- This isolates the visual terminal ticker into its own OS scheduling container, 
  preventing it from getting starved/frozen by intensive underlying LAPACK operations.
- Implemented a multiprocessing.Queue to cleanly handle cross-process text updates.
"""

class DisappearingTimer:
    """A background process-based timer that prints elapsed time and updates its message dynamically,
    preventing starvation from heavy multi-threaded underlying LAPACK/C-extension routines."""
    def __init__(self, message="Running..."):
        self.initial_message = message
        self._process = None
        self._queue = None
        self._stop_event = None

    def update_text(self, new_text):
        if self._queue and self._process and self._process.is_alive():
            self._queue.put(new_text)

    @staticmethod
    def _run(message, stop_event, queue):
        start_time = time.time()
        current_message = message.ljust(50)
        while not stop_event.is_set():
            # Check for incoming cross-process text updates
            while not queue.empty():
                try:
                    current_message = queue.get_nowait().ljust(50)
                except:
                    pass
            
            elapsed = time.time() - start_time
            sys.stdout.write(f"\r{current_message} [{elapsed:.1f}s]")
            sys.stdout.flush()
            time.sleep(0.1)
            
        # Clear the terminal line cleanly upon completion
        sys.stdout.write("\r" + " " * (len(current_message) + 20) + "\r")
        sys.stdout.flush()

    def __enter__(self):
        # Using multiprocessing context to instantiate safe cross-process primitives
        ctx = multiprocessing.get_context()
        self._queue = ctx.Queue()
        self._stop_event = ctx.Event()
        self._process = ctx.Process(
            target=self._run, 
            args=(self.initial_message, self._stop_event, self._queue)
        )
        self._process.daemon = True
        self._process.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._stop_event:
            self._stop_event.set()
        if self._process:
            self._process.join(timeout=1.0)
            if self._process.is_alive():
                self._process.terminate()


def colbert_miller_dvr_1d(potential_func, num_levels, x_min, x_max, num_points, mass=1.0, hbar=1.0):
    if num_points <= num_levels:
        raise ValueError(f"Number of grid points ({num_points}) must be greater than requested levels ({num_levels}).")
    
    x = np.linspace(x_min, x_max, num_points)
    dx = x[1] - x[0]
    k_factor = (hbar**2) / (2.0 * mass * dx**2)
    
    # 1. Build only the first row/column 
    idx = np.arange(num_points, dtype=float)
    
    # Generate alternating signs instantly without costly powers
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