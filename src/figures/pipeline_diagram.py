"""
pipeline_diagram.py
--------------------
Generates a figure summarizing the numerical workflow of the DVR heat-capacity
pipeline (potential -> spectrum -> convergence checks -> Cv(T) -> classical
limit -> benchmark).

Run this once to (re)produce `fig_pipeline.png`, e.g. after changing the
number/labels of the pipeline stages. No project-specific imports are needed;
the diagram is descriptive, not a call graph.

Usage:
    python pipeline_diagram.py
Output:
    figures/fig_pipeline.png   (repo-root figures/ folder)
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path

# Resolve the repo-root figures/ folder relative to this file's location
# (src/figures/pipeline_diagram.py), so the script works regardless of the
# current working directory it's launched from.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "..", "figures"))

# ---------------------------------------------------------------- Stages ---
# Each stage: (label, x, y, width, height, facecolor)
# Laid out as a 2-row "S-curve" so the figure is roughly landscape (nice for
# a two-column LaTeX figure* placement) instead of one long vertical strip.
STAGES = [
    ("1. Define potential $V(x)$\nand mass $m$",                          0.5, 3.3, "#dbe9f6"),
    ("2. Auto-configure grid\n(turning points + Nyquist $\\Delta x$)",     2.6, 3.3, "#dbe9f6"),
    ("3. Solve base DVR\nspectrum $\\{E_n\\}$",                            4.7, 3.3, "#cfe8cf"),
    ("4. Generate high-res.\nreference spectrum",                         6.8, 3.3, "#cfe8cf"),
    ("5. Verify resolution /\nboundary convergence",                      6.8, 1.1, "#f6e6c9"),
    ("6. Compute $C_v(T)$;\nscan $\\xi$ for classical\nplateau + $n$-check", 4.7, 1.1, "#f6d3c9"),
    ("7. Benchmark base vs.\nreference $C_v(T)$",                          2.6, 1.1, "#e3d6f0"),
]

BOX_W, BOX_H = 1.9, 1.15

fig, ax = plt.subplots(figsize=(11, 5.2))
ax.set_xlim(-0.3, 9.1)
ax.set_ylim(-0.15, 4.9)
ax.axis("off")

centers = []
for label, x, y, color in STAGES:
    box = FancyBboxPatch(
        (x, y), BOX_W, BOX_H,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.3, edgecolor="#333333", facecolor=color, zorder=2,
    )
    ax.add_patch(box)
    ax.text(x + BOX_W / 2, y + BOX_H / 2, label,
             ha="center", va="center", fontsize=10.5, zorder=3)
    centers.append((x + BOX_W / 2, y + BOX_H / 2, x, y))

def arrow(i, j, connectionstyle="arc3,rad=0.0"):
    x1, y1, bx1, by1 = centers[i]
    x2, y2, bx2, by2 = centers[j]
    # connect edge-to-edge rather than center-to-center
    if abs(y1 - y2) < 1e-6:  # same row -> horizontal edges
        if x1 < x2:
            p1, p2 = (bx1 + BOX_W, y1), (bx2, y2)
        else:
            p1, p2 = (bx1, y1), (bx2 + BOX_W, y2)
    else:  # different row -> vertical edges
        if y1 > y2:
            p1, p2 = (x1, by1), (x2, by2 + BOX_H)
        else:
            p1, p2 = (x1, by1 + BOX_H), (x2, by2)
    arr = FancyArrowPatch(
        p1, p2, arrowstyle="-|>", mutation_scale=16,
        connectionstyle=connectionstyle, linewidth=1.4,
        color="#333333", zorder=1,
    )
    ax.add_patch(arr)

# main forward chain: 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7
arrow(0, 1)
arrow(1, 2)
arrow(2, 3)
arrow(3, 4)
arrow(4, 5)
arrow(5, 6)

# feedback loop: stage 5 can send you back to stage 2 (widen/refine grid).
# Routed as a clean right-angle path underneath row 2 so it never crosses
# any box, rather than a diagonal arc cutting through stage 3.
x5, y5, bx5, by5 = centers[4]   # stage 5 (verify convergence)
x2, y2, bx2, by2 = centers[1]   # stage 2 (auto-configure grid)
y_drop = 0.55
verts = [
    (bx5 + BOX_W * 0.25, by5),          # leave from bottom of box 5
    (bx5 + BOX_W * 0.25, y_drop),       # drop down
    (bx2 + BOX_W * 0.75, y_drop),       # travel left, under everything
    (bx2 + BOX_W * 0.75, by2),          # rise up into bottom of box 2
]
loop_path = Path(verts, [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO])
loop = FancyArrowPatch(
    path=loop_path, arrowstyle="-|>", mutation_scale=14, linewidth=1.1,
    linestyle="dashed", color="#a33", zorder=1,
)
ax.add_patch(loop)
ax.text((bx5 + bx2) / 2 + BOX_W * 0.5, y_drop - 0.32,
        "refine grid if not converged", fontsize=8.5,
        color="#a33", ha="center", style="italic")

ax.set_title("Numerical workflow of the DVR heat-capacity pipeline",
              fontsize=13, fontweight="bold", pad=14)

plt.tight_layout()
os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, "fig_pipeline.png")
plt.savefig(out_path, dpi=220, bbox_inches="tight")
print(f"Saved {out_path}")