"""
Replay a single Food-Hunt configuration and log time-series goal metrics.

Purpose:
- Compute time-averaged goal occupancy over steps: the average fraction of particles
  within a given radius of the goal at each step (default radius = 1.0).
- Optionally export a per-step CSV with:
  step, com_dist, frac_within_radius, goal_x, goal_y

Input options (one of):
1) --from-csv <path> --row-index <i>
   - Reads the i-th data row (0-based among data rows, excluding header) from a random_search CSV.
   - Uses its config JSON, seed, and steps unless overridden.
2) --config-path <json> [--seed ...] [--steps ...]
   - Uses a standalone config JSON file (same schema/aliases as other tools).

Notes:
- Works regardless of whether respawn_on_reach is true/false; the goal used per step is the
  simulation's current food_pos at that step.
- The final-step goal completion (fraction within radius at the last step) is also printed
  to compare with the sweep's snapshot metric.

Usage examples:

# Replay a row from a Food-Hunt evaluation CSV (no-respawn) for 1000 steps
python experiments/replay_food_timeseries.py --from-csv results/food_no_respawn_steps1000.csv --row-index 3 --out results/food_row3_timeseries.csv

# Replay using a standalone config JSON, overriding steps and seed
python experiments/replay_food_timeseries.py --config-path configs/fixed_food_eval.json --steps 1200 --seed 777 --out results/food_eval_timeseries.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from typing import Any, Dict, Optional
import numpy as np

# Ensure project root on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from simulations.food_hunt_cell_headless import FoodHuntSimulation


def load_row_from_csv(path: str, row_index: int) -> Dict[str, Any]:
    """Load a specific data row (0-based among data rows) from a random_search CSV.

    Returns a dict with keys: config_json (dict), seed (int), steps (int)
    """
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
    if not rows:
        raise SystemExit(f"No rows found in CSV: {path}")
    if row_index < 0 or row_index >= len(rows):
        raise SystemExit(f"row-index {row_index} out of range [0, {len(rows)-1}]")
    row = rows[row_index]
    try:
        cfg = json.loads(row["config"]) if isinstance(row["config"], str) else row["config_json"]
    except Exception as e:
        raise SystemExit(f"Failed to parse config JSON from CSV row {row_index}: {e}")
    try:
        seed = int(row.get("seed", 0))
    except Exception:
        seed = 0
    try:
        steps = int(row.get("steps", 500))
    except Exception:
        steps = 500
    return {"config_json": cfg, "seed": seed, "steps": steps}


def metric_fraction_within_radius(points: np.ndarray, goal: np.ndarray, radius: float = 1.0) -> float:
    d = np.linalg.norm(points - goal, axis=1)
    return float(np.mean(d < radius))


def run_timeseries(config: Dict[str, Any], seed: Optional[int], steps: int, radius: float, out_csv: Optional[str]) -> Dict[str, Any]:
    sim = FoodHuntSimulation(config, seed=seed)

    # Prepare CSV writer if requested
    writer = None
    f = None
    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        f = open(out_csv, "w", newline="")
        writer = csv.writer(f)
        writer.writerow(["step", "com_dist", "frac_within_radius", "goal_x", "goal_y"])    

    fractions = []
    for t in range(steps):
        # One step
        sim.step()
        pts = sim.get_state()
        goal = sim.food_pos.copy()
        # Center of mass distance to current goal
        com_x = float(np.mean(pts[:, 0]))
        com_y = float(np.mean(pts[:, 1]))
        com_dist = math.hypot(goal[0]-com_x, goal[1]-com_y)
        # Fraction of particles within radius at this step
        frac = metric_fraction_within_radius(pts, goal, radius=radius)
        fractions.append(frac)
        if writer:
            writer.writerow([t+1, com_dist, frac, goal[0], goal[1]])

    if f:
        f.close()

    avg_occupancy = float(np.mean(fractions)) if fractions else 0.0
    final_frac = fractions[-1] if fractions else 0.0
    return {
        "avg_occupancy": avg_occupancy,
        "final_step_fraction": final_frac,
        "steps": steps,
        "radius": radius,
    }


def main():
    ap = argparse.ArgumentParser(description="Replay Food-Hunt and log time-series goal occupancy.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-csv", dest="from_csv", help="Input sweep CSV (random_search output)")
    src.add_argument("--config-path", dest="config_path", help="Standalone config JSON path")

    ap.add_argument("--row-index", dest="row_index", type=int, default=0, help="Row index (0-based) when using --from-csv")
    ap.add_argument("--steps", dest="steps", type=int, default=0, help="Override steps; 0 = use CSV/1000 default")
    ap.add_argument("--seed", dest="seed", type=int, default=None, help="Override seed; None = use CSV or default")
    ap.add_argument("--radius", dest="radius", type=float, default=1.0, help="Metric radius for occupancy (default 1.0)")
    ap.add_argument("--out", dest="out_csv", default="", help="Optional per-step CSV to write")

    args = ap.parse_args()

    if args.from_csv:
        row = load_row_from_csv(args.from_csv, args.row_index)
        config = row["config_json"]
        seed = args.seed if args.seed is not None else row["seed"]
        steps = args.steps if args.steps > 0 else (row["steps"] or 1000)
    else:
        with open(args.config_path, "r") as f:
            config = json.load(f)
        seed = args.seed
        steps = args.steps if args.steps > 0 else 1000

    # Run and optionally write per-step CSV
    res = run_timeseries(config=config, seed=seed, steps=steps, radius=args.radius, out_csv=(args.out_csv or None))

    print(
        f"avg_occupancy={res['avg_occupancy']:.4f} over {res['steps']} steps at radius={res['radius']}, "
        f"final_step_fraction={res['final_step_fraction']:.4f}"
    )


if __name__ == "__main__":
    main()
