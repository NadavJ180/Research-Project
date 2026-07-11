"""
Classical_Limit_Numerical_1_0.py
=====================================================================
WHAT THIS FILE DOES
---------------------------------------------------------------------
General-purpose (system-agnostic) numerics for finding the CLASSICAL
LIMIT of the heat capacity Cv purely from a finite list of quantum
energy levels -- no analytic formula required. Works for any system
(box, harmonic oscillator, double well, ...) as long as you can hand
it an array of energy eigenvalues.

The core idea: take the discrete spectrum {E_n} and introduce a
scaling factor xi that effectively stretches/compresses the spectrum
relative to k_B*T. As xi grows, the levels become "denser" relative
to thermal energy, mimicking what an infinite, continuous spectrum
would do -- which is exactly the classical limit. We scan xi upward,
watch Cv(xi) plateau, and call that plateau value the numerical
classical limit at that temperature. A companion scan over how many
levels n are actually needed (n-convergence) checks that the
plateau isn't an artifact of having too few states available.

This file contains NO plotting and NO knowledge of any particular
potential -- it is purely the numerical engine. Plotting and
system-specific wiring live in the files that import this one
(Quantum_Classical_Combined_1_9.py, HO_Benchmark_1_0.py, etc.).

CHANGELOG (NEW FILE, v1.0)
---------------------------------------------------------------------
- Extracted from Quantum_Classical_Combined_1_8_Opti.py and given its
  own file so the classical-limit-finding numerics can be reused by
  any system/driver without dragging along plotting or DVR-specific
  code. No change to the underlying math versus v1.8 -- this is a
  pure code-organization split.
- Added full module/function-level docstrings per project
  documentation conventions.
=====================================================================
"""

import numpy as np
from tqdm import tqdm


# =====================================================================
# Core Cv formula (also used directly for the "real" quantum Cv(T))
# =====================================================================
def compute_cv(energies, beta, xi=1.0):
    """
    Compute Cv/k_B at a single inverse temperature `beta` from a
    discrete energy spectrum, optionally rescaled by a factor `xi`.

    With xi=1.0 this is the literal, physical quantum heat capacity
    for the given (finite) spectrum. With xi != 1.0, the spectrum is
    effectively probed at an energy scale stretched by xi^2 -- this
    is the building block used by `converge_xi` to find the
    classical-limit plateau.

    Parameters
    ----------
    energies : array_like
        Energy eigenvalues (any order; need not be pre-sorted).
    beta : float
        Inverse temperature, 1 / (k_B * T).
    xi : float, optional
        Spectrum scaling factor (default 1.0 = no scaling, i.e. the
        true quantum Cv for this spectrum).

    Returns
    -------
    Cv : float
        Heat capacity (in units of k_B) at this beta and xi. Returns
        np.nan if the partition function underflows to zero or is
        otherwise non-finite (can happen at extreme beta*xi combos).
    """
    energies = np.asarray(energies, dtype=float)
    a = beta * energies / xi**2
    # Subtract the minimum exponent before exponentiating to avoid
    # overflow/underflow in the Boltzmann weights.
    w = np.exp(-(a - a.min()))
    Z = w.sum()
    if Z == 0 or not np.isfinite(Z):
        return np.nan
    avg_E = np.dot(w, energies) / Z
    avg_E2 = np.dot(w, energies**2) / Z
    return float((beta**2 / xi**4) * (avg_E2 - avg_E**2))


# =====================================================================
# xi-convergence: scan the scaling factor until Cv(xi) plateaus
# =====================================================================
def converge_xi(energies, beta, xi_start, tol_xi, min_stable, xi_mult, max_steps):
    """
    Scan xi upward (geometrically, by xi_mult each step) starting
    from xi_start, tracking Cv(xi), and detect the plateau where Cv
    stops changing -- that plateau value is the numerical classical
    limit at this beta.

    The scan also detects the failure mode where Cv eventually
    collapses toward zero (the spectrum has run out of resolvable
    states at large xi, i.e. a finite-N truncation artifact) and
    reports that distinctly from genuine convergence.

    Parameters
    ----------
    energies : array_like
        Energy eigenvalues for this system.
    beta : float
        Inverse temperature being probed.
    xi_start : float
        Initial scaling factor to start scanning from.
    tol_xi : float
        Maximum |Cv(xi_i) - Cv(xi_{i-1})| allowed for consecutive
        steps to be considered "stable".
    min_stable : int
        Minimum number of consecutive stable steps required before
        declaring convergence (guards against accidentally flat
        single steps).
    xi_mult : float
        Geometric growth factor applied to xi each step (xi *= xi_mult).
    max_steps : int
        Maximum number of xi steps to try before giving up.

    Returns
    -------
    dict with keys:
        xi_converged, cv_converged : float or None
            The plateau xi and Cv value, if convergence was found.
        converged : bool
            True if a genuine plateau (not a finite-N collapse) was found.
        stop_reason : str
            One of "converged", "finite_n", or "max_steps".
        xi_values, cv_values, deltas : list
            Full trace of the scan, useful for diagnostic plotting.
    """
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

            # Detect a finite-N collapse: Cv falling toward ~zero relative to its peak.
            if (peak_cv > 1e-4 and (last3 < 0.02 * peak_cv).all()) or np.isclose(last3, 0.0, atol=1e-5).all():
                pre = cvs[:-3]
                if len(pre) >= min_stable + 1:
                    pre_d = [abs(pre[i] - pre[i - 1]) for i in range(1, len(pre))]
                    streak, found = 0, False
                    for d in pre_d:
                        if d < tol_xi:
                            streak += 1
                            if streak >= min_stable:
                                found = True
                                break
                        else:
                            streak = 0
                    stop_reason = "converged" if found else "finite_n"
                else:
                    stop_reason = "finite_n"
                break
        xi *= xi_mult

    deltas = [None] + [abs(cvs[i] - cvs[i - 1]) for i in range(1, len(cvs))]

    xi_converged = cv_converged = None
    if stop_reason == "converged":
        # Locate the actual flat plateau region (excluding the trailing
        # collapse-detection window) and take its midpoint as the answer.
        pre_cvs = cvs[:-3]
        pre_d = [None] + [abs(pre_cvs[i] - pre_cvs[i - 1]) for i in range(1, len(pre_cvs))]
        plat_end = None
        for i in range(len(pre_d) - 1, 0, -1):
            if pre_d[i] is not None and pre_d[i] < tol_xi:
                plat_end = i
                break
        if plat_end is not None:
            plat_start = plat_end
            while plat_start > 1 and pre_d[plat_start - 1] is not None and pre_d[plat_start - 1] < tol_xi:
                plat_start -= 1
            plat_len = plat_end - plat_start + 1
            mid = plat_start + plat_len // 2
            n_take = min(3, plat_len)
            idx_start = max(plat_start, mid - n_take // 2)
            xi_converged = float(np.mean(xis[idx_start:idx_start + n_take]))
            cv_converged = float(np.mean(cvs[idx_start:idx_start + n_take]))
        else:
            xi_converged = float(np.mean(xis[-6:-3]))
            cv_converged = float(np.mean(cvs[-6:-3]))

    return {
        "xi_converged": xi_converged, "cv_converged": cv_converged,
        "converged": stop_reason == "converged", "stop_reason": stop_reason,
        "xi_values": xis, "cv_values": cvs, "deltas": deltas,
    }


# =====================================================================
# n-convergence: how many energy levels are actually needed?
# =====================================================================
def converge_n(energies, beta, xi, tol_cv, min_stable):
    """
    Sweep how many of the (ascending) energy levels are included in
    the partition sum, n = 2, 3, ..., N, tracking Cv(n) at a fixed
    (beta, xi), and find the smallest n after which Cv stays stable
    for the rest of the available spectrum.

    This answers "did we include enough states for this answer to be
    trustworthy at this temperature?" -- independent of (but used
    alongside) the xi-convergence check.

    Parameters
    ----------
    energies : array_like
        Full available energy spectrum (ascending order expected).
    beta : float
        Inverse temperature being probed.
    xi : float
        Scaling factor to use while sweeping n (typically the
        xi_converged value found by `converge_xi`, or 1.0 for the
        physical/unscaled quantum Cv).
    tol_cv : float
        Maximum |Cv(n) - Cv(n-1)| for two consecutive n to count as stable.
    min_stable : int
        Minimum run length of stable steps, extending all the way to
        n=N, required to declare convergence.

    Returns
    -------
    dict with keys:
        n_converged, cv_converged : int/float or None
            Smallest converged n and its Cv value (None if not converged).
        converged : bool
        n_values, cv_values, deltas : list
            Full trace, useful for diagnostic plotting.
    """
    N = len(energies)
    n_values = list(range(2, N + 1))

    energies_arr = np.asarray(energies, dtype=float)
    a = beta * energies_arr / (xi**2)
    w = np.exp(-(a - a.min()))

    # Cumulative sums let us compute Cv(n) for every n in one vectorized pass
    # instead of recomputing the partition sum from scratch each time.
    Z_n = np.cumsum(w)
    E_w_n = np.cumsum(w * energies_arr)
    E2_w_n = np.cumsum(w * (energies_arr**2))

    with np.errstate(divide='ignore', invalid='ignore'):
        avg_E = E_w_n / Z_n
        avg_E2 = E2_w_n / Z_n
        cv_array = (beta**2 / xi**4) * (avg_E2 - avg_E**2)

    cv_values = cv_array[1:].tolist()
    deltas = [None] + [abs(cv_values[i] - cv_values[i - 1]) for i in range(1, len(cv_values))]
    stable = [False] + [(d < tol_cv) for d in deltas[1:]]

    # Find runs of consecutive stable steps at least min_stable long.
    runs = []
    i = 0
    while i < len(stable):
        if stable[i]:
            j = i
            while j < len(stable) and stable[j]:
                j += 1
            if j - i >= min_stable:
                runs.append((i, j - 1))
            i = j
        else:
            i += 1

    # Convergence only counts if the stable run extends all the way to
    # the end of the available spectrum (otherwise it might just be a
    # coincidental flat stretch followed by renewed drift).
    last_idx = len(n_values) - 1
    n_converged = cv_conv = None
    for start_idx, end_idx in runs:
        if end_idx == last_idx:
            n_converged = n_values[start_idx]
            cv_conv = cv_values[start_idx]
            break

    return {
        "n_converged": n_converged, "cv_converged": cv_conv,
        "converged": n_converged is not None,
        "n_values": n_values, "cv_values": cv_values, "deltas": deltas,
    }


# =====================================================================
# Sweep both convergence checks across a full temperature range
# =====================================================================
def sweep_temperature_range(energies, beta_arr,
                             xi_start, tol_xi, min_stable_xi, xi_mult, max_xi_steps,
                             tol_cv, min_stable_n, verbose=True):
    """
    Run `converge_xi` followed by `converge_n` at every beta in
    `beta_arr`, assembling the numerical classical-limit curve
    Cv_classical(T) plus the convergence diagnostics needed to know
    how trustworthy each point is.

    At each temperature: find xi_converged via `converge_xi`; if
    found, use it (otherwise fall back to xi=1.0) to run `converge_n`
    and double check level-count convergence. The reported
    Cv_classical at that temperature prefers the xi-convergence
    result, falling back to the n-convergence result only if xi
    convergence itself failed.

    Parameters
    ----------
    energies : array_like
        Full available energy spectrum for this system.
    beta_arr : array_like
        Array of inverse temperatures to sweep over.
    xi_start, tol_xi, min_stable_xi, xi_mult, max_xi_steps :
        Passed through to `converge_xi` at every beta.
    tol_cv, min_stable_n :
        Passed through to `converge_n` at every beta.
    verbose : bool, optional
        If True, shows a tqdm progress bar and prints a one-line
        convergence summary at the end.

    Returns
    -------
    dict with keys:
        cv_classical, xi_conv, n_conv : ndarray, shape (len(beta_arr),)
            Per-temperature classical-limit Cv, converged xi, and
            converged n (np.nan where convergence failed).
        xi_results, n_results : list of dict
            Full per-temperature convergence traces (for diagnostic plotting).
        xi_fail_mask, n_fail_mask : ndarray of bool
    """
    n_T = len(beta_arr)
    cv_classical = np.full(n_T, np.nan)
    xi_conv_arr = np.full(n_T, np.nan)
    n_conv_arr = np.full(n_T, np.nan)
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
        n_n_fail = np.isnan(n_conv_arr).sum()
        if n_xi_fail:
            print(f"  \u26a0  \u03be-convergence failed at {n_xi_fail}/{n_T} temperatures.")
        if n_n_fail:
            print(f"  \u26a0  n-convergence failed at {n_n_fail}/{n_T} temperatures.")
        if not n_xi_fail and not n_n_fail:
            print(f"  \u2713  Both \u03be and n converged at all {n_T} temperatures.")

    return {
        "cv_classical": cv_classical, "xi_conv": xi_conv_arr, "n_conv": n_conv_arr,
        "xi_results": xi_results, "n_results": n_results,
        "xi_fail_mask": np.isnan(xi_conv_arr), "n_fail_mask": np.isnan(n_conv_arr),
    }
