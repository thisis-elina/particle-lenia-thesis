"""
Plot Food‑Hunt time‑series metrics from replay CSV.

Input CSV is produced by experiments/replay_food_timeseries.py and must contain
columns: step, com_dist, frac_within_radius, goal_x, goal_y

Example usage (from repo root):

python experiments/plot_food_timeseries.py \
  --in results/artifacts/food_row2_timeseries.csv \
  --out results/figures/food_row2_timeseries.png \
  --title "Row 2 (r=1.0)" --show-dist

Notes:
- The plotted fraction corresponds to the radius used during replay.
- Use --show-dist to overlay center‑of‑mass distance (second y‑axis).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

# Ensure project root on sys.path (safe no‑op if already present)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


def load_timeseries(path: str):
    steps, frac, dist = [], [], []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            steps.append(int(row["step"]))
            frac.append(float(row["frac_within_radius"]))
            # com_dist may be absent in older files, so guard
            dval = row.get("com_dist", "")
            dist.append(float(dval) if dval not in (None, "", "NaN", "nan") else float("nan"))
    return steps, frac, dist


def main():
    ap = argparse.ArgumentParser(description="Plot Food‑Hunt time‑series from replay CSV.")
    ap.add_argument("--in", dest="in_csv", required=True, help="Input timeseries CSV from replay")
    ap.add_argument("--out", dest="out_png", required=True, help="Output PNG path")
    ap.add_argument("--title", dest="title", default="Food‑Hunt time series", help="Figure title")
    ap.add_argument("--show-dist", dest="show_dist", action="store_true", help="Overlay COM distance")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)

    steps, frac, dist = load_timeseries(args.in_csv)

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(steps, frac, color="tab:blue", label="fraction within radius")
    ax1.set_xlabel("step")
    ax1.set_ylabel("fraction within radius", color="tab:blue")
    ax1.set_ylim(0.0, 1.0)
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    if args.show_dist:
        ax2 = ax1.twinx()
        ax2.plot(steps, dist, color="tab:red", alpha=0.6, label="COM distance")
        ax2.set_ylabel("COM distance", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")

    fig.suptitle(args.title)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=150)
    print(f"Wrote: {args.out_png}")


if __name__ == "__main__":
    main()
