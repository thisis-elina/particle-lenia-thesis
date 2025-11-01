"""
Thesis figure generator (visualizations wrapper).

Purpose:
- Provide a clean entry point to render thesis plots into results/figures/ by default.
- Wrap existing plotters so you don't need to remember individual scripts.

Generates:
- Particle‑Lenia scatter: stability vs diversity, color = avg_energy (PNG)
- Food‑Hunt histogram: goal_completion distribution (PNG)

Usage examples (from repo root, Windows PowerShell):

# Default output directory: results/figures/
python experiments/visualizations/make_figures.py --plenia results/plenia_sweep.csv --food results/food_no_respawn_p64_steps1800.csv

# Custom output directory
python experiments/visualizations/make_figures.py --plenia results/plenia_sweep.csv --food results/food_sweep.csv --outdir results/figures_exp2

Notes:
- This does not alter your CSVs. It only reads them and writes PNGs.
- For Food‑Hunt, choose the evaluation CSV you want to present (e.g., no‑respawn, tuned settings).
"""
from __future__ import annotations

import argparse
import os
import sys
import subprocess

# Ensure project root on sys.path when run from experiments/visualizations/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_OUTDIR = os.path.join("results", "figures")


def main():
    ap = argparse.ArgumentParser(description="Generate thesis figures (wrapper).")
    ap.add_argument("--plenia", required=True, help="Input CSV for Particle‑Lenia sweep")
    ap.add_argument("--food", required=True, help="Input CSV for Food‑Hunt sweep/evaluation")
    ap.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Directory to write PNGs (default results/figures)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Build output subfolders
    plenia_dir = os.path.join(args.outdir, "plenia")
    food_dir = os.path.join(args.outdir, "food")
    os.makedirs(plenia_dir, exist_ok=True)
    os.makedirs(food_dir, exist_ok=True)
    plenia_png = os.path.join(plenia_dir, "plenia_scatter.png")
    food_png = os.path.join(food_dir, "food_goal_hist.png")

    # Call existing plotters so behavior stays consistent
    # plenia scatter
    subprocess.check_call([
        sys.executable,
        os.path.join("experiments", "plot_plenia_scatter.py"),
        "--in", args.plenia,
        "--out", plenia_png,
        "--title", "Particle‑Lenia Stability vs Diversity"
    ])

    # food histogram
    subprocess.check_call([
        sys.executable,
        os.path.join("experiments", "plot_food_goal_hist.py"),
        "--in", args.food,
        "--out", food_png,
        "--bins", "20",
        "--title", "Food‑Hunt Goal Completion"
    ])

    print(f"Wrote: {plenia_png}\nWrote: {food_png}")


if __name__ == "__main__":
    main()
