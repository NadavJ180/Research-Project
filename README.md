# Quantum-Classical Heat Capacity

A modular, system-agnostic numerical pipeline for computing the heat capacity $C_v(T)$ of a quantum particle in a smooth 1-D potential, and locating its classical limit, using the Discrete Variable Representation (DVR) method. The pipeline validates itself without ever depending on a closed-form solution: every result is checked against an independently generated, higher-resolution numerical reference rather than an analytic formula — the same check that will be needed once the project moves to potentials that have no closed-form answer at all.

**Status:** The harmonic oscillator (HO) is fully validated, reproducing the exact analytic result to machine precision. The pipeline is now being extended to a quartic double well.

For the full write-up — theory, validation strategy, results, and derivations — see [`docs/summaries/IEEE_Summary.tex`](docs/summaries/IEEE_Summary.tex) (the project's IEEE-style report). For a technical summary of the physics and how to read the diagnostic plots, see [`FINDINGS.md`](FINDINGS.md). For how the pipeline evolved over the semester, see [`HISTORY.md`](HISTORY.md). Meeting-by-meeting notes and worked derivations are in [`docs/summaries/Meetings_Summary.tex`](docs/summaries/Meetings_Summary.tex).

---

## Repository Structure

```
src/
├── config.py                     Single source of truth: potential, constants, control parameters
├── Quantum_HO_Master.py          Master driver — entry point, runs the full 6-section pipeline
├── Quantum_Classical_Combined.py General Cv pipeline (quantum Cv(T) + classical-limit scan)
├── Classical_Limit_Numerical.py  xi/n convergence engine (classical-limit search)
├── Cv_Numerical_Benchmark.py     Cv comparison: base grid vs. numerical reference
├── DVR/
│   ├── DVR_Algorithm.py          Core DVR solver and automatic grid configuration
│   ├── DVR_Reference_Generator.py  Numerical reference grid generator
│   └── DVR_Limit_Finder.py       DVR accuracy limit searches (resolution and level-count)
├── analytical/
│   ├── HO_Analytical.py          Closed-form HO energy levels and Cv(T)
│   └── HO_Benchmark.py           External benchmark: numerical pipeline vs. analytic HO
├── error/
│   └── error_energylevels.py     Energy-level comparison (base vs. reference)
└── figures/
    ├── plot_potential.py         Potential-shape figure generator
    └── pipeline_diagram.py       Workflow-diagram generator

figures/            Generated plots (HO/, SymmetricDoubleWell/, plus standalone figures)
docs/summaries/      IEEE_Summary.tex (main report), Meetings_Summary.tex (meeting notes + derivations)
results/             Scratch/intermediate run outputs (not tracked)
```

## Pipeline Overview

`Quantum_HO_Master.py` runs six sequential sections, all driven by the single potential defined in `config.py`:

1. **DVR base solve** — lowest `NUM_STATES` energy levels on an automatically configured grid.
2. **Numerical reference solve** — the same spectrum on an independently wider/finer grid, generated once and shared by every later step as the ground truth.
3. **Energy-level accuracy** — base vs. reference eigenvalues, absolute and relative error.
4. **Cv pipeline** — quantum $C_v(T)$ from the base spectrum, plus the numerical classical limit via the $\xi$/$n$-convergence scan.
5. **DVR limit analysis** — the DVR solver's own resolution ($\Delta x$) and level-count ($n$) breakdown points, measured against the reference.
6. **Cv numerical benchmark** — the full Cv pipeline re-run on the reference spectrum, closing the loop between eigenvalue accuracy and the final thermodynamic observable.

Only Section 0 of `config.py` needs editing to run on a new potential — no other file changes.

## Generalising to a New Potential

1. Edit `my_potential(x)` in `src/config.py` to the new $V(x)$ (must be finite everywhere — no hard walls).
2. Update `SYSTEM_NAME` and `T_UNITS_LABEL` for plot labelling.
3. Adjust `NUM_STATES`, `BETA_MIN`/`BETA_MAX`, and `XI_START` for the new system's energy scale.
4. Run `src/Quantum_HO_Master.py` — the grid auto-configurator and reference generator adapt automatically.

## Acknowledgements

**Supervisor:** Dr. David Gelbwaser-Klimovsky
**Student:** Nadav Jean
