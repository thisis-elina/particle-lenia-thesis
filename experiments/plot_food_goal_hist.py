"""
Plot Food‑Hunt: goal completion histogram (and optional stability/diversity scatter).

Usage (from repo root):
  python experiments\plot_food_goal_hist.py --in results\food_sweep.csv --out results\food_goal_hist.png

Notes:
- Expects a CSV produced by experiments/random_search.py with columns:
  [simulation, seed, steps, config, avg_energy, stability, diversity, goal_completion, elapsed_sec]
- Filters to simulation == "food-hunt".
- Requires matplotlib; install if needed:
    python -m pip install matplotlib
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from typing import List

# Lazy import to provide a clear error if matplotlib is missing
try:
    import matplotlib.pyplot as plt
except Exception as e:  # pragma: no cover
    plt = None
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None


def load_food_rows(path: str):
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("simulation") != "food-hunt":
                continue
            try:
                row["goal_completion"] = float(row["goal_completion"]) if row["goal_completion"] != "" else float("nan")
            except ValueError:
                continue
            rows.append(row)
    return rows


def plot_hist(rows, out_path: str, bins: int = 20, title_suffix: str = ""):
    if _IMPORT_ERR is not None:
        raise SystemExit(
            "matplotlib is required for plotting. Install with: python -m pip install matplotlib\n"
            f"Import error: {_IMPORT_ERR}"
        )

    vals = [r["goal_completion"] for r in rows if math.isfinite(r["goal_completion"]) ]
    if len(vals) == 0:
        raise SystemExit("No valid Food‑Hunt rows with finite goal_completion found.")

    plt.figure(figsize=(7.5, 5.0), dpi=140)
    plt.hist(vals, bins=bins, color="#1f77b4", alpha=0.9, edgecolor="white")
    plt.xlabel("Goal completion (fraction in [0,1])")
    plt.ylabel("Count")
    ttl = "Food‑Hunt: Goal completion distribution"
    if title_suffix:
        ttl += f" — {title_suffix}"
    plt.title(ttl)
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    print(f"Wrote figure: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Plot Food‑Hunt goal completion histogram.")
    ap.add_argument("--in", dest="in_csv", required=True, help="Input CSV from random_search")
    ap.add_argument("--out", dest="out_png", default="results/food_goal_hist.png", help="Output PNG path")
    ap.add_argument("--bins", dest="bins", type=int, default=20, help="Number of histogram bins")
    ap.add_argument("--title", dest="title", default="", help="Optional title suffix")
    args = ap.parse_args()

    rows = load_food_rows(args.in_csv)
    plot_hist(rows, args.out_png, bins=args.bins, title_suffix=args.title)


if __name__ == "__main__":
    main()
