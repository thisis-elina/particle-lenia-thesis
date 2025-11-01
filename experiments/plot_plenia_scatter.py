"""
Plot Particle‑Lenia: stability vs diversity scatter (color = avg_energy),
with an approximate Pareto frontier overlay.

Usage (from repo root):
  python experiments\plot_plenia_scatter.py --in results\plenia_sweep.csv --out results\plenia_scatter.png

Notes:
- Expects a CSV produced by experiments/random_search.py with columns:
  [simulation, seed, steps, config, avg_energy, stability, diversity, goal_completion, elapsed_sec]
- Filters to simulation == "particle-lenia".
- If your CSV mixes modes, it's fine; it will auto-filter.
- Requires matplotlib; install if needed:
    python -m pip install matplotlib
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from typing import List, Tuple

# Lazy import to provide a clear error if matplotlib is missing
try:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
except Exception as e:  # pragma: no cover
    plt = None
    mpl = None
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None


def load_particle_lenia_rows(path: str):
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("simulation") != "particle-lenia":
                continue
            try:
                row["stability"] = float(row["stability"]) if row["stability"] != "" else float("nan")
                row["diversity"] = float(row["diversity"]) if row["diversity"] != "" else float("nan")
                row["avg_energy"] = float(row["avg_energy"]) if row["avg_energy"] != "" else float("nan")
            except ValueError:
                continue
            rows.append(row)
    return rows


def pareto_front(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """Return a simple Pareto frontier for maximizing both x (stability) and y (diversity).

    Implementation: sort by stability asc, then keep points whose diversity is a new max.
    """
    pts = sorted(points, key=lambda t: (t[0], t[1]))
    frontier = []
    best_y = -math.inf
    for x, y in pts:
        if y > best_y:
            frontier.append((x, y))
            best_y = y
    return frontier


def plot_scatter(rows, out_path: str, title_suffix: str = ""):
    if _IMPORT_ERR is not None:
        raise SystemExit(
            "matplotlib is required for plotting. Install with: python -m pip install matplotlib\n"
            f"Import error: {_IMPORT_ERR}"
        )

    x = [r["stability"] for r in rows if math.isfinite(r["stability"]) and math.isfinite(r["diversity"]) and math.isfinite(r["avg_energy"]) ]
    y = [r["diversity"] for r in rows if math.isfinite(r["stability"]) and math.isfinite(r["diversity"]) and math.isfinite(r["avg_energy"]) ]
    c = [r["avg_energy"] for r in rows if math.isfinite(r["stability"]) and math.isfinite(r["diversity"]) and math.isfinite(r["avg_energy"]) ]

    if len(x) == 0:
        raise SystemExit("No valid Particle‑Lenia rows with finite stability/diversity/avg_energy found.")

    # Pareto
    frontier = pareto_front(list(zip(x, y)))

    plt.figure(figsize=(7.5, 5.5), dpi=140)
    sc = plt.scatter(x, y, c=c, cmap="viridis", s=24, alpha=0.85, edgecolors='none')
    plt.colorbar(sc, label="average energy")
    # Overlay frontier as a line
    fx, fy = zip(*frontier)
    plt.plot(fx, fy, color="#ff7f0e", lw=2.0, label="Pareto frontier (approx.)")

    plt.xlabel("Stability (higher = steadier)")
    plt.ylabel("Diversity (det(cov))")
    ttl = "Particle‑Lenia: Stability vs Diversity"
    if title_suffix:
        ttl += f" — {title_suffix}"
    plt.title(ttl)
    plt.legend(loc="lower right")
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"Wrote figure: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Plot Particle‑Lenia stability vs diversity scatter with Pareto frontier.")
    ap.add_argument("--in", dest="in_csv", required=True, help="Input CSV from random_search")
    ap.add_argument("--out", dest="out_png", default="results/plenia_scatter.png", help="Output PNG path")
    ap.add_argument("--title", dest="title", default="", help="Optional title suffix")
    args = ap.parse_args()

    rows = load_particle_lenia_rows(args.in_csv)
    plot_scatter(rows, args.out_png, title_suffix=args.title)


if __name__ == "__main__":
    main()
