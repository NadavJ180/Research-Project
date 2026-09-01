"""
plot_potential.py  (src/figures/)
=====================================================================
Plots the potential V(x) for whatever system is currently configured
in config.py, with the ACTUAL computed energy levels overlaid -- not
a placeholder count. Every input (potential, mass, hbar, bounds,
number of levels) is read from config.py and DVR_Algorithm.py, the
same modules Quantum_HO_Master.py itself uses for Section 1. That
means this script always plots exactly the system currently being
researched: change the potential in config.py and re-run, nothing
here needs to be touched.

Because it calls the real grid auto-configuration and the real
3-pass converged DVR solve for all NUM_STATES levels, this script
does real (if modest) computation -- it is not instantaneous the way
a purely illustrative plot would be, but the eigenvalues it draws are
the same ones the rest of the pipeline is using.

Run from anywhere; paths are resolved relative to this file.

Usage:
    python plot_potential.py
Output:
    figures/fig_potential_<system>.png   (repo-root figures/ folder)
"""

import os
import re
import sys

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Make `src/` (for config.py) and `src/DVR/` (via the `DVR` package)
# importable regardless of the current working directory this script
# is launched from.
# ---------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(THIS_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import MASS, HBAR, NUM_STATES, SYSTEM_NAME, my_potential  # noqa: E402
from DVR.DVR_Algorithm import (                                        # noqa: E402
    auto_configure_dvr,
    get_fully_converged_energy_levels,
)

# ============================== USER CONFIG ===============================
# Number of levels to draw. Defaults to ALL of NUM_STATES (i.e. exactly the
# scope currently being researched, per config.py) -- override to an int
# only if you want a deliberately reduced, less visually dense subset.
LEVELS_TO_DRAW = NUM_STATES 

OUTPUT_DIR = os.path.join(REPO_ROOT, "figures")
# ============================================================================


def _slugify(name):
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip()).strip("_").lower()
    return slug or "system"


def  classical_turning_points(V, E, x):
    """Leftmost/rightmost x where V(x) <= E, for drawing a level segment
    that spans the classically allowed region at that energy."""
    allowed = x[V(x) <= E]
    if allowed.size == 0:
        return None
    return allowed.min(), allowed.max()


def main():
    print(f"Configuring grid for '{SYSTEM_NAME}', {NUM_STATES} levels ...")
    x_min, x_max, n_grid = auto_configure_dvr(
        my_potential, NUM_STATES, mass=MASS, hbar=HBAR
    )

    print("Solving (3-pass converged) for the real energy levels "
          "-- this reuses the same call as Section 1, so it can take "
          "a little while for large NUM_STATES ...")
    energies = get_fully_converged_energy_levels(
        potential_func=my_potential,
        num_levels=NUM_STATES,
        x_min=x_min, x_max=x_max, num_points=n_grid,
        mass=MASS, hbar=HBAR,
    )

    levels = np.sort(np.asarray(energies))[:LEVELS_TO_DRAW]

    x = np.linspace(x_min, x_max, 4000)
    V = my_potential(x)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, V, color="#1f4e8c", linewidth=2.2, label="$V(x)$", zorder=3)

    # Many levels (NUM_STATES can be in the hundreds) -> thin, semi-
    # transparent, colormap-graded lines rather than individually labelled
    # ones. This renders as a density gradient that is still informative
    # (dense near the bottom, sparser as E grows) instead of a solid block.
    n_levels = len(levels)
    cmap = plt.cm.autumn_r
    label_every = max(1, n_levels // 10)  # label ~10 levels even if N is large
    for i, E in enumerate(levels):
        tp = classical_turning_points(my_potential, E, x)
        if tp is None:
            continue
        xl, xr = tp
        color = cmap(0.15 + 0.8 * i / max(n_levels - 1, 1))
        if i % 10 == 0:
            ax.hlines(E, xl, xr, color=color, linewidth=1.0,
                    alpha=0.55 if n_levels > 40 else 1.0, zorder=2)
        if n_levels <= 40 or i % label_every == 0 or i == n_levels - 1:
            ax.text(xr + 0.015 * (x_max - x_min), E, f"$n={i}$",
                     va="center", fontsize=7, color="#555555")

    ax.set_xlabel("$x$")
    ax.set_ylabel("Energy")
    ax.set_title(f"{SYSTEM_NAME} -- potential and computed spectrum\n"
                 f"({n_levels} levels shown)", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend(loc="upper center")
    fig.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"fig_potential_{_slugify(SYSTEM_NAME)}.png")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()