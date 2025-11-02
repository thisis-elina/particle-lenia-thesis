"""
Plot Particle‑Lenia Top‑K robustness (error bars) from aggregated CSV.

Input CSV must be produced by experiments/replay_topk.py and contain columns:
rank, base_seed, steps, n_seeds, mean_energy, std_energy, mean_stability, std_stability, mean_diversity, std_diversity, config

This script will create a single PNG with three subplots showing error bars for:
- stability (mean ± std)
- diversity (mean ± std)
- energy (mean ± std)

Usage (from repo root):

python experiments/plot_pl_robustness.py \
  --in results/artifacts/plenia_topk_robust_s8_s2000.csv \
  --out results/figures/plenia/plenia_robustness.png \
  --title "Particle‑Lenia Top‑K Robustness (8 seeds, 2000 steps)"

Notes:
- The x‑axis is Top‑K rank; points include error bars.
- Use different input CSVs if you ran multiple robustness passes (e.g., s5_s1500 vs s8_s2000).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import List, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_rows(path: str) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                rows.append({
                    "rank": int(row["rank"]),
                    "mean_stability": float(row["mean_stability"]),
                    "std_stability": float(row["std_stability"]),
                    "mean_diversity": float(row["mean_diversity"]),
                    "std_diversity": float(row["std_diversity"]),
                    "mean_energy": float(row["mean_energy"]),
                    "std_energy": float(row["std_energy"]),
                })
            except Exception:
                # Skip malformed rows
                continue
    # Sort by rank ascending
    rows.sort(key=lambda d: d["rank"])
    return rows


def main():
    ap = argparse.ArgumentParser(description="Plot PL Top‑K robustness (error bars) from aggregated CSV.")
    ap.add_argument("--in", dest="in_csv", required=True, help="Input CSV from replay_topk.py")
    ap.add_argument("--out", dest="out_png", required=True, help="Output PNG path")
    ap.add_argument("--title", dest="title", default="Particle‑Lenia Top‑K Robustness", help="Figure title")
    args = ap.parse_args()

    rows = load_rows(args.in_csv)
    if not rows:
        raise SystemExit(f"No valid rows in: {args.in_csv}")

    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)

    ranks = [r["rank"] for r in rows]
    stab_m = [r["mean_stability"] for r in rows]
    stab_s = [r["std_stability"] for r in rows]
    div_m = [r["mean_diversity"] for r in rows]
    div_s = [r["std_diversity"] for r in rows]
    eng_m = [r["mean_energy"] for r in rows]
    eng_s = [r["std_energy"] for r in rows]

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    # Stability
    ax = axes[0]
    ax.errorbar(ranks, stab_m, yerr=stab_s, fmt='o-', color='tab:blue', ecolor='lightblue', capsize=3)
    ax.set_ylabel('stability (mean ± std)')
    ax.grid(True, linestyle=':', alpha=0.5)

    # Diversity (use log scale if values are large)
    ax = axes[1]
    ax.errorbar(ranks, div_m, yerr=div_s, fmt='o-', color='tab:green', ecolor='lightgreen', capsize=3)
    ax.set_ylabel('diversity (mean ± std)')
    ax.grid(True, linestyle=':', alpha=0.5)

    # Energy
    ax = axes[2]
    ax.errorbar(ranks, eng_m, yerr=eng_s, fmt='o-', color='tab:purple', ecolor='#d7b3ff', capsize=3)
    ax.set_ylabel('energy (mean ± std)')
    ax.set_xlabel('Top‑K rank')
    ax.grid(True, linestyle=':', alpha=0.5)

    fig.suptitle(args.title)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(args.out_png, dpi=150)
    print(f"Wrote: {args.out_png}")


if __name__ == "__main__":
    main()
