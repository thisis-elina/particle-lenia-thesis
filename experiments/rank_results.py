"""
Ranking utility for sweep results produced by experiments/random_search.py.

Expected input CSV schema:
[simulation, seed, steps, config, avg_energy, stability, diversity, goal_completion, elapsed_sec]

Objectives:
- composite: J = z(diversity) + λ · z(stability)
- constrained: same composite but rows failing stability/diversity thresholds are heavily penalized
- food-goal: prioritize goal_completion (Food-Hunt) with a diversity floor

Outputs:
- Ranked CSV with score and original metrics
- Optional Top-K JSON with exact configs for re-runs
"""

import argparse
import csv
import json
import math
import os
from statistics import mean, stdev

# This script ranks results produced by experiments/random_search.py or a similar CSV
# Schema expected: [simulation, seed, steps, config, avg_energy, stability, diversity, goal_completion, elapsed_sec]


def load_rows(path):
    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            # Cast numeric fields
            for k in ["seed", "steps"]:
                row[k] = int(row[k])
            for k in ["avg_energy", "stability", "diversity", "elapsed_sec"]:
                row[k] = float(row[k]) if row[k] != "" else float("nan")
            # goal_completion may be NaN for particle-lenia
            try:
                row["goal_completion"] = float(row["goal_completion"]) if row["goal_completion"] != "" else float("nan")
            except ValueError:
                row["goal_completion"] = float("nan")
            # Parse config JSON
            try:
                row["config_json"] = json.loads(row["config"])
            except Exception:
                row["config_json"] = None
            rows.append(row)
    return rows


def zscore(values):
    vals = [v for v in values if not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))]
    if len(vals) < 2:
        # Avoid div-by-zero; return zeros
        return [0.0 for _ in values]
    mu = mean(vals)
    sd = stdev(vals) if len(vals) > 1 else 1.0
    if sd == 0:
        return [0.0 for _ in values]
    return [(0.0 if (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else (v - mu) / sd) for v in values]


def rank(rows, mode: str, objective: str, lam: float, stab_min: float, diversity_floor: float):
    # Filter to selected simulation mode if provided
    if mode:
        rows = [r for r in rows if r["simulation"] == mode]
    if not rows:
        return []

    # Compute z-scores on the filtered set
    stab_z = zscore([r["stability"] for r in rows])
    div_z = zscore([r["diversity"] for r in rows])

    scores = []
    for i, r in enumerate(rows):
        s = r["stability"]
        d = r["diversity"]
        gc = r.get("goal_completion", float("nan"))

        if objective == "composite":
            j = div_z[i] + lam * stab_z[i]
        elif objective == "constrained":
            # Penalize if constraints not met
            ok = True
            if s < stab_min:
                ok = False
            if d < diversity_floor:
                ok = False
            j = (div_z[i] + lam * stab_z[i]) if ok else -1e9
        elif objective == "food-goal":
            # For FoodHunt: prioritize goal completion, then diversity floor
            if math.isnan(gc):
                j = -1e9
            else:
                j = gc if d >= diversity_floor else -1e9
        else:
            raise ValueError(f"Unknown objective: {objective}")
        scores.append(j)

    # Attach score and sort
    for r, j in zip(rows, scores):
        r["score"] = j

    rows_sorted = sorted(rows, key=lambda r: r["score"], reverse=True)
    return rows_sorted


def write_ranked(rows_sorted, out_csv, topk_json=None, top_k=10):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    fieldnames = [
        "rank", "simulation", "seed", "steps", "score", "avg_energy", "stability", "diversity", "goal_completion", "elapsed_sec", "config"
    ]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, r in enumerate(rows_sorted, start=1):
            w.writerow({
                "rank": i,
                "simulation": r["simulation"],
                "seed": r["seed"],
                "steps": r["steps"],
                "score": r["score"],
                "avg_energy": r["avg_energy"],
                "stability": r["stability"],
                "diversity": r["diversity"],
                "goal_completion": r.get("goal_completion", float("nan")),
                "elapsed_sec": r["elapsed_sec"],
                "config": r["config"],
            })

    if topk_json:
        top = [{
            "rank": i+1,
            "simulation": r["simulation"],
            "seed": r["seed"],
            "steps": r["steps"],
            "score": r["score"],
            "config": r.get("config_json", r.get("config")),
        } for i, r in enumerate(rows_sorted[:top_k])]
        with open(topk_json, "w") as f:
            json.dump(top, f, indent=2)


def main():
    ap = argparse.ArgumentParser(description="Rank random search results by objective.")
    ap.add_argument("--in", dest="in_csv", required=True, help="Input CSV from random_search")
    ap.add_argument("--out", dest="out_csv", default="results/ranked.csv", help="Output ranked CSV path")
    ap.add_argument("--mode", choices=["", "particle-lenia", "food-hunt"], default="", help="Filter by simulation mode")
    ap.add_argument("--objective", choices=["composite", "constrained", "food-goal"], default="composite")
    ap.add_argument("--lambda", dest="lam", type=float, default=0.6, help="Weight for stability in composite score")
    ap.add_argument("--stab-min", dest="stab_min", type=float, default=0.1, help="Minimum stability for constrained objective")
    ap.add_argument("--div-floor", dest="div_floor", type=float, default=1e-6, help="Diversity floor for constraints")
    ap.add_argument("--topk-json", dest="topk_json", default="", help="Optional path to write top-K configs JSON")
    ap.add_argument("--top-k", dest="top_k", type=int, default=10, help="How many configs to export to JSON")
    args = ap.parse_args()

    rows = load_rows(args.in_csv)
    rows_sorted = rank(rows, mode=args.mode or None, objective=args.objective,
                       lam=args.lam, stab_min=args.stab_min, diversity_floor=args.div_floor)

    write_ranked(rows_sorted, out_csv=args.out_csv,
                 topk_json=(args.topk_json or None), top_k=args.top_k)

    print(f"Ranked {len(rows_sorted)} rows. Wrote: {args.out_csv}" + (f" and {args.topk_json}" if args.topk_json else ""))


if __name__ == "__main__":
    main()
