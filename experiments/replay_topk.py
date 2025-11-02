"""
Replay Particle‑Lenia Top‑K configs across multiple seeds and longer steps,
then write an aggregated robustness table (mean ± std) for thesis.

Input: Top‑K JSON produced by experiments/rank_results.py (plenia_topk.json)
Each entry should include: rank, simulation, seed, steps, config.

Output: CSV with aggregate stats per Top‑K row, including:
- rank, base_seed, steps, n_seeds
- mean_energy, std_energy
- mean_stability, std_stability
- mean_diversity, std_diversity
- config (JSON string)

Usage examples (run from repo root):

# 5 seeds, 1500 steps, write to artifacts
python experiments/replay_topk.py \
  --topk results/artifacts/plenia_topk.json \
  --out  results/artifacts/plenia_topk_robust.csv \
  --seeds 5 --steps 1500

# 10 seeds, keep original per‑row steps (from Top‑K), offset seeds by +1000
python experiments/replay_topk.py \
  --topk results/artifacts/plenia_topk.json \
  --out  results/artifacts/plenia_topk_robust_s10.csv \
  --seeds 10 --seed-offset 1000
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from statistics import mean, pstdev
from typing import Any, Dict, List

# Ensure project root on sys.path when run from experiments/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from simulations.particle_lenia_headless import ParticleLeniaSimulation
from utils.metrics import stability_score, diversity_score


def load_topk(path: str) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        data = json.load(f)
    # Normalize entries (some rankers may store config under different keys)
    norm = []
    for e in data:
        cfg = e.get("config") or e.get("config_json")
        if isinstance(cfg, str):
            try:
                cfg = json.loads(cfg)
            except Exception:
                pass
        norm.append({
            "rank": int(e.get("rank", 0)),
            "simulation": e.get("simulation", "particle-lenia"),
            "seed": int(e.get("seed", 0)),
            "steps": int(e.get("steps", 500)),
            "config": cfg,
        })
    return norm


def run_plenia(cfg: Dict[str, Any], seed: int, steps: int) -> Dict[str, float]:
    sim = ParticleLeniaSimulation(cfg, seed=seed)
    avg_E = sim.run_headless(steps=steps)
    pts = sim.get_state()
    return {
        "avg_energy": float(avg_E),
        "stability": float(stability_score(pts)),
        "diversity": float(diversity_score(pts)),
    }


def aggregate(rows: List[Dict[str, float]]) -> Dict[str, float]:
    def m(values):
        return float(mean(values)) if values else float("nan")
    def s(values):
        return float(pstdev(values)) if values and len(values) > 1 else 0.0
    E = [r["avg_energy"] for r in rows]
    S = [r["stability"] for r in rows]
    D = [r["diversity"] for r in rows]
    return {
        "mean_energy": m(E), "std_energy": s(E),
        "mean_stability": m(S), "std_stability": s(S),
        "mean_diversity": m(D), "std_diversity": s(D),
    }


def main():
    ap = argparse.ArgumentParser(description="Replay PL Top‑K across seeds and steps to compute mean±std.")
    ap.add_argument("--topk", required=True, help="Path to Top‑K JSON (from rank_results.py)")
    ap.add_argument("--out", required=True, help="Output CSV path for aggregated stats")
    ap.add_argument("--seeds", type=int, default=5, help="Number of seeds per Top‑K row")
    ap.add_argument("--steps", type=int, default=0, help="Override steps; 0 = use each row's steps")
    ap.add_argument("--seed-offset", type=int, default=0, help="Offset added to base seed for replay")
    args = ap.parse_args()

    topk = load_topk(args.topk)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "rank", "base_seed", "steps", "n_seeds",
            "mean_energy", "std_energy",
            "mean_stability", "std_stability",
            "mean_diversity", "std_diversity",
            "config",
        ])
        for entry in topk:
            if entry.get("simulation") != "particle-lenia":
                # Skip non‑PL entries silently
                continue
            cfg = entry["config"]
            base_seed = entry["seed"] + args.seed_offset
            steps = args.steps if args.steps > 0 else int(entry["steps"])

            per_seed = []
            for sidx in range(args.seeds):
                seed = base_seed + sidx
                metrics = run_plenia(cfg, seed=seed, steps=steps)
                per_seed.append(metrics)
            agg = aggregate(per_seed)
            w.writerow([
                entry["rank"], base_seed, steps, args.seeds,
                agg["mean_energy"], agg["std_energy"],
                agg["mean_stability"], agg["std_stability"],
                agg["mean_diversity"], agg["std_diversity"],
                json.dumps(cfg),
            ])
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
