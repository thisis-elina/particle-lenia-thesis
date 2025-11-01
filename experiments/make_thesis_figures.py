"""
Convenience script: generate core thesis figures from sweep CSVs.

It will:
- Read Particle‑Lenia sweep CSV and produce a stability vs diversity scatter with a Pareto frontier.
- Read Food‑Hunt sweep CSV and produce a goal completion histogram.

Usage (from repo root):
  python experiments\make_thesis_figures.py \
      --plenia results\plenia_sweep.csv \
      --food  results\food_sweep.csv \
      --outdir results

Outputs (defaults):
  results/plenia_scatter.png
  results/food_goal_hist.png

Requires matplotlib:
  python -m pip install matplotlib
"""
from __future__ import annotations

import argparse
import os
import sys

# Ensure project root on path (in case run from experiments/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from experiments.plot_plenia_scatter import load_particle_lenia_rows, plot_scatter
from experiments.plot_food_goal_hist import load_food_rows, plot_hist


def main():
    ap = argparse.ArgumentParser(description="Generate thesis figures from sweep CSVs.")
    ap.add_argument("--plenia", dest="plenia_csv", required=True, help="Particle‑Lenia sweep CSV path")
    ap.add_argument("--food", dest="food_csv", required=True, help="Food‑Hunt sweep CSV path")
    ap.add_argument("--outdir", dest="outdir", default="results", help="Directory to write figures")
    ap.add_argument("--title", dest="title", default="", help="Optional title suffix for plots")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Particle‑Lenia scatter
    pl_rows = load_particle_lenia_rows(args.plenia_csv)
    pl_out = os.path.join(args.outdir, "plenia_scatter.png")
    plot_scatter(pl_rows, pl_out, title_suffix=args.title)

    # Food‑Hunt histogram
    fh_rows = load_food_rows(args.food_csv)
    fh_out = os.path.join(args.outdir, "food_goal_hist.png")
    plot_hist(fh_rows, fh_out, bins=20, title_suffix=args.title)

    print("All figures written:")
    print(" -", pl_out)
    print(" -", fh_out)


if __name__ == "__main__":
    main()
