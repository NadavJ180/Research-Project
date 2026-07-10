Quantum Heat Machines Research Project

This repository contains the simulation code and mathematical derivations for studying the convergence of quantum systems to their classical limits, specifically focusing on the Heat Capacity ($C_v$) of various potential energy systems.


Project Overview

This project investigates the quantum-to-classical transition using a scaling factor, $\xi$, which effectively scales $\hbar \rightarrow 0$ by scaling the temperature and potential energy. The core objective is to identify a stable $\xi$-plateau where the calculated heat capacity converges to the classical limit predicted by the Equipartition Theorem.


Key Features & Algorithms

- DVR Algorithm: A Discrete Variable Representation (DVR) solver used to compute energy eigenvalues. It has been optimized for smooth potentials, enforcing strict computational contracts to ensure stability.

- Phase-Space Auto-Scanner (v1.6+): Dynamically optimizes grid density ($N$) and spatial bounds ($[x_{min}, x_{max}]$) based on the Nyquist-Shannon sampling theorem to prevent aliasing.

- $\xi$-Convergence Module: An automated search that navigates the narrow stability window defined by:
$$\sqrt{\beta \Delta E} \ll \xi \ll \sqrt{\beta E_{max}}$$
This ensures the system reaches the classical continuum limit without suffering from finite-$N$ truncation.

- Performance Optimization: Includes decoupled diagnostic timers, grid padding refinements, and matrix-free approaches to overcome $\mathcal{O}(N^3)$ complexity limitations.


Mathematical Foundations

The project explores two primary systems:

1. Harmonic Oscillator (HO)
- Quantum Limit: Analytical derivation of $Z^{HO}$ and $C_v^{HO}$ using the geometric series of energy levels $E_n = \hbar\omega(n+1/2)$.
- Classical Limit: Confirms $C_v \rightarrow k_B$ via the limit $\xi \rightarrow \infty$.

2. Box Potential
- Quantum Limit: Employs numerical approximation of the partition function via truncation of the infinite sum, deriving $C_v$ from the variance of the energy: $C_v = k_B \beta^2 \cdot Var(E)$.
- Classical Limit: Confirms $C_v \rightarrow \frac{1}{2}k_B$.


Troubleshooting & Best Practices

- Performance: For intensive linear algebra operations, ensure diagnostic timers are removed from the main solver loops.

- Convergence:If $C_v$ drops to zero at high temperatures, check the $\xi$ stability window.
If the classical limit drops at low temperatures, increase the starting $\xi$ value.
Always verify convergence by checking both quantum and classical $C_v$ plateaus.


Future Work

- FFT-Accelerated Methods: Transitioning to matrix-free operators using scipy.sparse.linalg.eigsh and the Convolution Theorem to reduce complexity to $\mathcal{O}(N \log N)$.
- Adaptive Mesh: Implementing spatial mapping transformations $x = f(u)$ to satisfy the Nyquist limit locally, potentially reducing $N$ by a factor of 3 to 4.

Supervisor: Dr. David Gelbwaser-Klimovsky
Student: Nadav Jean