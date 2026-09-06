# History

How this pipeline got to its current state. For a meeting-by-meeting account (including the reasoning behind each change), see [`docs/summaries/Meetings_Summary.tex`](docs/summaries/Meetings_Summary.tex); this is a condensed summary. Full detail is always in `git log`.

## Narrative

The project began with the two textbook systems that have closed-form solutions — the **harmonic oscillator (HO)** and the **box potential** — used to validate the variance-based $C_v = k_B\beta^2\mathrm{Var}(E)$ formula and the $\xi$-scaling route to the classical limit against known analytic answers.

A **Colbert–Miller sinc-DVR solver** was then built to obtain energy levels numerically rather than analytically, first tested against the HO (whose exact spectrum makes it a validation case, not a target). Early iterations manually tuned the grid; this was replaced by an **automatic grid configurator** (turning-point root-finding plus a local Nyquist criterion) that removed the need for per-potential guesswork.

Moving beyond the HO exposed several issues specific to anharmonic potentials, addressed in order:

- **Insufficient tail padding.** A fixed padding heuristic, calibrated for the HO, underestimated how far anharmonic wavefunction tails extend. Replaced with iterative span widening that stops once eigenvalues stop changing.
- **Narrow $\xi$-plateau window.** Because the double well's levels scale as $E_n\propto n^{4/3}$ rather than linearly, its valid $\xi$-window is narrower than the HO's. The $\xi$-step size was made more granular (multiplier reduced from 1.3 to 1.1) so the scan reliably lands inside the window before the finite-$N$ collapse.
- **Hard-wall potentials dropped.** The DVR core was restricted to smooth, everywhere-finite potentials only — a discontinuous potential's unbounded momentum content can't be represented on any finite grid, so supporting it safely would require a separate formulation entirely.

The pipeline's validation strategy also evolved: rather than relying on the HO's analytic formula as ground truth for every run, a **numerical reference generator** (an independently wider/finer DVR solve) was introduced to play that role instead — the same role the analytic formula plays for the HO, but one that works for any smooth potential, including ones with no closed form. The analytic HO comparison was kept, but demoted to a one-time external certification of the numerical-reference methodology itself, run in a separate benchmarking module rather than baked into the main pipeline.

Along the way the codebase was reorganized from a monolithic script into the current modular `src/` layout (`DVR/`, `analytical/`, `error/`, `figures/`), with version numbers dropped from filenames once the module boundaries stabilized, and diagnostic timers relocated out of the tight solver loops to avoid distorting runtime-critical code paths.

**Current state:** the HO benchmark (Sections 1–6 of the pipeline) is fully validated, agreeing with the exact analytic solution to machine precision. The same pipeline, unmodified beyond the potential definition, is now running on a quartic double well — the first anharmonic system, and the first candidate for a genuine physical Schottky anomaly (see `docs/summaries/IEEE_Summary.tex`).

## Version History

Pre-reorganization filenames carried explicit version suffixes (e.g. `DVR_Algorithm_1_4.py`); current files under `src/` no longer do (see [`README.md`](README.md) for the current structure). This table is kept as a historical record of the module-level changes that shaped the current design.

| File | Version | Key change |
|------|---------|-----------|
| `DVR_Algorithm` | 1.3 | Hard-wall support removed; smooth-only |
| `DVR_Algorithm` | 1.4 | Multiprocessing timer removed; replaced with inline per-pass timing |
| `DVR_Algorithm` | 1.5 | Adaptive span-expansion loop added to `auto_configure_dvr`; `E_ceiling` reverted from an oversized temporary hack back to `1.5 * num_levels` |
| `Classical_Limit_Numerical` | 1.0 | Extracted from the combined file; fully general |
| `Quantum_Classical_Combined` | 1.9 | System-agnostic Cv pipeline; xi/n engine extracted |
| `DVR_Reference_Generator` | 1.0 | New: numerical reference grid generation, replacing the analytic formula as ground truth |
| `DVR_Limit_Finder` | 1.0 | New: resolution and level-count limit searches |
| `DVR_Limit_Finder` | 1.1 | $\Delta x$ replaces point count as the resolution-search axis |
| `DVR_Limit_Finder` | 1.2 | Point annotations removed from the $\Delta x$ plot |
| `Cv_Numerical_Benchmark` | 1.0 | New: base vs. reference Cv comparison (quantum + classical) |
| `HO_Energy_Level_Error` | 1.1 | Docstring clarified — function is fully generic |
| `Quantum_HO_Master` | 1.5 | Analytical sections removed; fully numerical pipeline (numerical reference is now the sole ground truth for Sections 3, 5, 6) |
