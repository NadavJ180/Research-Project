# Findings

Technical summary of the physics behind this pipeline and how to read its diagnostic output. For the full derivations and validation argument, see [`docs/summaries/IEEE_Summary.tex`](docs/summaries/IEEE_Summary.tex); this document is a practical companion to it.

## Physics Background

### Heat Capacity from the Partition Function

Given energy eigenvalues $\{E_n\}$ at inverse temperature $\beta=1/(k_BT)$, the canonical-ensemble heat capacity is

$$C_v(T) = k_B \beta^2 \left[\langle E^2 \rangle - \langle E \rangle^2\right] = k_B\beta^2\,\mathrm{Var}(E), \qquad \langle E^k \rangle = \frac{\sum_n E_n^k\, e^{-\beta E_n}}{Z},\ \ Z = \sum_n e^{-\beta E_n}.$$

This holds regardless of whether $\{E_n\}$ comes from a closed-form expression or a numerical diagonalization — the physics doesn't care how the spectrum was obtained. Throughout, the system is assumed to be in thermal equilibrium with a bath at temperature $T$ (the canonical ensemble above); this is the standard basis for a quasi-static engine-cycle picture, not a finite-time or non-equilibrium treatment.

### The DVR Method

The Discrete Variable Representation (DVR) is a grid-based method for solving the 1-D time-independent Schrödinger equation $\hat H\psi=E\psi$, $\hat H=\hat T+V(x)$, by representing the wavefunction by its values on an evenly spaced grid rather than by coefficients in an analytical basis.

This implementation uses the **Colbert–Miller sinc-DVR** (1992), whose kinetic-energy matrix on a grid of spacing $\Delta x$ is exact (not an approximation) for band-limited functions sampled on that grid:

$$T_{ij} = \frac{\hbar^2}{2m\,\Delta x^2}\times\begin{cases}\dfrac{\pi^2}{3}, & i=j,\\[4pt]\dfrac{2\,(-1)^{i-j}}{(i-j)^2}, & i\neq j.\end{cases}$$

The potential energy is diagonal, $V_{ij}=V(x_i)\,\delta_{ij}$. Because $T_{ij}$ depends only on $|i-j|$, the matrix is Toeplitz and is built from a single row rather than by evaluating the formula at all $N^2$ entries independently. The Hamiltonian is diagonalized with `scipy.linalg.eigvalsh` using `subset_by_index`, which extracts only the lowest $N_{\text{lev}}$ eigenvalues directly via LAPACK's MRRR algorithm rather than resolving the full spectrum.

**Scope restriction:** the solver requires $V(x)$ to be finite everywhere on the grid. Hard-wall potentials (infinite square well, etc.) are explicitly rejected — a discontinuous derivative has unbounded momentum content that no finite grid spacing can resolve without aliasing.

**Grid limitations:** only roughly the lower half of any requested spectrum is trustworthy (the highest-index states are the first whose local momentum exceeds what a fixed spacing can resolve — see Search A/B below), and the dense $N\times N$ diagonalization scales as $\mathcal{O}(N^2)$ in memory and $\mathcal{O}(N^3)$ in time, the practical ceiling on how far $N_{\text{lev}}$ can be pushed.

### The Classical Limit and the $\xi$-Scaling Trick

The classical limit is $C_v$ as $\hbar\to0$ at fixed $T$ — for the HO this is exactly the equipartition value $k_B$ (two quadratic degrees of freedom, $\tfrac12k_B$ each), independent of $T$. Since $\hbar$ can't actually be dialed down, the pipeline scans a dimensionless factor $\xi$ that plays the role of $1/\hbar$: scaling both temperature and the spectrum by $\xi^2$,

$$a_n(\xi) = \frac{\beta E_n}{\xi^2},$$

is mathematically equivalent to sending $\hbar\to\hbar/\xi$. As $\xi\to\infty$ the spectrum looks continuous and $C_v(\xi)\to C_v^{\text{classical}}$. Any computed spectrum is truncated at a finite $E_{\max}$, so the plateau can only be trusted inside a window,

$$\sqrt{\beta\,\Delta E} \;\ll\; \xi \;\ll\; \sqrt{\beta\,E_{\max}},$$

set below by the level spacing needing to look continuous, and above by the top computed level needing to stay thermally inaccessible. The code scans $\xi$ upward geometrically from `XI_START`, detects the plateau (consecutive stable steps), and records that value as the classical limit at that temperature — while separately detecting the "finite-$N$ collapse" that occurs once $\xi$ overshoots the window and the spectrum runs out of resolvable states, and handling it as a rejected result rather than a false answer. A companion $n$-convergence scan (how many of the computed levels were actually needed to reach the plateau) gives an independent check that the result isn't an artifact of truncation.

**On the HO and equipartition:** the HO's classical $C_v$ is exactly $k_B$ at *every* temperature, not just asymptotically at high $T$ — so the numerically found plateau matches the equipartition prediction across the whole sweep. That exact temperature-independence is a special feature of the HO (a purely quadratic potential); it isn't guaranteed to hold as cleanly for an anharmonic potential like the double well.

## Findings So Far

- **HO validation:** base-grid energy levels agree with the numerical reference to within machine precision across the full computed spectrum. The resulting quantum and classical-limit $C_v(T)$ curves agree with the exact analytic (Einstein-oscillator) formula to machine precision as well — evidence that the pipeline's internal, closed-form-free validation methodology (base grid vs. numerical reference) actually tracks the true answer, not just a shared artifact of the method.
- **DVR resolution/level limits:** characterized directly for the HO — confirms the standard rule of thumb that only the lower half of a requested spectrum should be trusted.
- **Double well:** first anharmonic system in the pipeline ($V(x)=\tfrac14x^4+bx^3-\tfrac12x^2$); its levels scale as $E_n\propto n^{4/3}$ rather than linearly, narrowing the valid $\xi$-window relative to the HO. Run in progress — see `docs/summaries/IEEE_Summary.tex` §Systems and §Future Work for the current status and next steps.

## Reading the Diagnostic Plots

| Plot | What to look for |
|---|---|
| **Energy-level comparison** (base vs. reference, full range + zoom near largest error) | Curves visually indistinguishable at full scale; the zoom panel shows the worst-case disagreement in context. |
| **Relative error vs. state index $n$** | A smooth, gently rising curve — low-lying states are most accurate (longest de Broglie wavelength), highest states least. An abrupt spike at some $n^*$ flags where grid resolution first becomes insufficient. |
| **$\xi$-convergence diagnostic** (at the hardest temperature in the sweep) | Rising flank → flat plateau (stable region) → falling collapse (finite-$N$ region). A clear separation between plateau and collapse indicates robust convergence; a short or absent plateau means too few levels for that temperature. |
| **$n$-convergence diagnostic** | Near-zero at low $n$, smooth rise, flat tail. The converged point should sit well before the right edge — if it's at the very last level, there's no safety margin and `NUM_STATES` should be increased. |
| **$C_v(T)$ summary** (quantum + classical limit + $\xi_{\text{conv}}(T)$/$n_{\text{conv}}(T)$) | Quantum curve rises smoothly from ~0 to the classical plateau; the secondary axis shows which temperatures were hardest to converge. |
| **DVR resolution/level-count limit plots** | Long machine-precision floor, then an abrupt cliff once $\Delta x$ (or requested $n$) crosses the solver's breakdown point. |
| **Cv benchmark plots** (base vs. reference, or numerical vs. analytic) | Flat, featureless error curve well below the convergence tolerance across the full temperature range; structure in the error is diagnostic (see Troubleshooting below). |

## Key Parameters (`src/config.py`)

| Parameter | Effect |
|-----------|--------|
| `NUM_STATES` | More levels → higher computational cost, but wider temperature coverage and a higher trustworthy-$n$ ceiling in the DVR limit analysis. |
| `BETA_MIN` / `BETA_MAX` | Temperature sweep window. Too cold a `BETA_MAX` shrinks the range where the classical limit converges. |
| `XI_START` | Higher start → classical limit locatable at colder temperatures; `3.0` is a good default. |
| `TOL_XI` / `TOL_CV` | Tighter tolerances → more accurate classical limit, at the cost of a slower sweep and more scan steps. |
| `LIMIT_TOLERANCE` | Pass/fail threshold used by the DVR resolution/level-count limit searches. |
| `REFERENCE_SPAN_FACTOR` / `REFERENCE_DX_FACTOR` | How much wider/finer the numerical reference grid is than the base grid; `2.0`/`2.0` (span doubled, spacing halved) is the default. |

## Troubleshooting

- **Quantum $C_v$ cuts off abruptly:** extend the temperature sweep further in the relevant direction (smaller $\beta$ for higher $T$, larger $\beta$ for lower $T$) — you likely haven't reached the plateau yet.
- **Classical limit drops at low $T$:** increase `XI_START`.
- **Classical limit drops at high $T$ / Cv benchmark error rises at high $T$:** the partition sum may be truncating thermally-accessible levels — increase `NUM_STATES`.
- **Cv benchmark error rises at low $T$:** the lowest eigenvalues are inaccurate — check the base grid's resolution or boundary span.
- **Relative-error metrics blowing up at very low $T$:** both the numerator and denominator are underflowing toward zero there; check the *absolute* error instead, which stays meaningful throughout.
