"""
Animate Particle‑Lenia or Food‑Hunt as moving circles and export to GIF/MP4.

Two input modes (pick one):
1) --from-csv <random_search_csv> --row-index <i>
   - Replays a config from a sweep CSV (uses its config JSON + seed; steps can be overridden).
2) --config-path <json> [--seed ...] [--steps ...]
   - Replays a standalone config JSON (aliases supported by random_search.py are fine; here we just pass dict through).

Examples (run from repo root):
# Replay a Food‑Hunt eval row to GIF (every 4th step, 600 frames):
python experiments/visualizations/animate_simulation.py \
  --mode food-hunt \
  --from-csv results/food_no_respawn_p64_steps1800.csv --row-index 2 \
  --steps 1200 --frame-stride 4 --fps 20 \
  --out results/figures/food_row2_anim.gif

# Replay Particle‑Lenia from a JSON config, save MP4 (requires ffmpeg installed):
python experiments/visualizations/animate_simulation.py \
  --mode particle-lenia \
  --config-path configs/fixed.json --seed 123 --steps 1000 --frame-stride 3 --fps 25 \
  --out results/figures/plenia_demo.mp4

Notes:
- For GIF export we use PillowWriter (no external dependencies). MP4 export attempts to use ffmpeg; install if needed or use GIF.
- To keep files small, increase --frame-stride (e.g., 3–6) or reduce --steps.
- Axis limits are auto‑scaled based on initial spread; adjust via --extent if desired.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from typing import Any, Dict, Optional, Tuple

# Ensure project root on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
from matplotlib import animation

from simulations.particle_lenia_headless import ParticleLeniaSimulation
from simulations.food_hunt_cell_headless import FoodHuntSimulation

SIMS = {
    "particle-lenia": ParticleLeniaSimulation,
    "food-hunt": FoodHuntSimulation,
}


def load_row(path: str, row_index: int) -> Tuple[Dict[str, Any], int, int]:
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
    if not rows:
        raise SystemExit(f"No rows in CSV: {path}")
    if row_index < 0 or row_index >= len(rows):
        raise SystemExit(f"row-index {row_index} out of range [0, {len(rows)-1}]")
    row = rows[row_index]
    try:
        cfg = json.loads(row["config"]) if isinstance(row["config"], str) else row["config_json"]
    except Exception as e:
        raise SystemExit(f"Failed to parse config JSON from row {row_index}: {e}")
    seed = int(row.get("seed", 0))
    steps = int(row.get("steps", 500))
    return cfg, seed, steps


def auto_extent(points: np.ndarray, goal: Optional[np.ndarray] = None, margin: float = 2.0) -> Tuple[float, float, float, float]:
    xs = points[:, 0]
    ys = points[:, 1]
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    if goal is not None and np.isfinite(goal).all():
        gx, gy = float(goal[0]), float(goal[1])
        xmin = min(xmin, gx)
        xmax = max(xmax, gx)
        ymin = min(ymin, gy)
        ymax = max(ymax, gy)
    # Add margin
    dx = xmax - xmin
    dy = ymax - ymin
    if dx == 0: dx = 1.0
    if dy == 0: dy = 1.0
    return xmin - margin, xmax + margin, ymin - margin, ymax + margin


def main():
    ap = argparse.ArgumentParser(description="Animate Particle‑Lenia / Food‑Hunt and export GIF/MP4.")
    ap.add_argument("--mode", choices=list(SIMS.keys()), required=True)

    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-csv", dest="from_csv", help="Input random_search CSV for config replay")
    src.add_argument("--config-path", dest="config_path", help="Standalone config JSON path")

    ap.add_argument("--row-index", type=int, default=0, help="Row index when using --from-csv")
    ap.add_argument("--seed", type=int, default=None, help="Override seed")
    ap.add_argument("--steps", type=int, default=0, help="Override steps; 0 = use CSV/1000 default")
    ap.add_argument("--frame-stride", type=int, default=3, help="Sim steps per video frame (>=1)")
    ap.add_argument("--fps", type=int, default=20, help="Output frames per second")
    ap.add_argument("--extent", type=float, nargs=4, default=None, help="Axis limits xmin xmax ymin ymax (optional)")
    ap.add_argument("--lock-extent", action="store_true", help="Do not adjust axes during animation; otherwise we ensure goal stays visible")
    ap.add_argument("--out", required=True, help="Output file path (.gif or .mp4)")

    args = ap.parse_args()

    # Load config/seed/steps
    if args.from_csv:
        cfg, seed0, steps0 = load_row(args.from_csv, args.row_index)
        seed = args.seed if args.seed is not None else seed0
        steps = args.steps if args.steps > 0 else steps0
    else:
        with open(args.config_path, "r") as f:
            cfg = json.load(f)
        seed = args.seed
        steps = args.steps if args.steps > 0 else 1000

    Sim = SIMS[args.mode]
    sim = Sim(cfg, seed=seed)

    # Initialize and determine plot extent from current points
    pts = sim.get_state()
    if args.extent and len(args.extent) == 4:
        xmin, xmax, ymin, ymax = args.extent
    else:
        goal0 = sim.food_pos if args.mode == "food-hunt" else None
        xmin, xmax, ymin, ymax = auto_extent(pts, goal0)

    # Figure setup
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    scat = ax.scatter(pts[:, 0], pts[:, 1], s=10, c="tab:blue", alpha=0.9)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f"{args.mode} — seed={seed} steps={steps}")

    # (Optional) mark goal for Food‑Hunt
    goal_artist = None
    if args.mode == "food-hunt":
        goal_artist = ax.scatter([sim.food_pos[0]], [sim.food_pos[1]], s=50, c="tab:red", marker="x", label="goal")
        ax.legend(loc="upper right")

    # Animation update
    stride = max(1, int(args.frame_stride))
    total_frames = max(1, steps // stride)

    def update(frame_idx: int):
        # advance simulation by stride steps
        for _ in range(stride):
            sim.step()
        p = sim.get_state()
        scat.set_offsets(p)
        if goal_artist is not None:
            goal_artist.set_offsets([sim.food_pos])
        # Keep goal visible by adjusting axes to include current goal (unless locked)
        if not args.lock_extent:
            gx, gy = (sim.food_pos[0], sim.food_pos[1]) if args.mode == "food-hunt" else (None, None)
            if gx is not None:
                xmin2, xmax2, ymin2, ymax2 = auto_extent(p, np.array([gx, gy]))
            else:
                xmin2, xmax2, ymin2, ymax2 = auto_extent(p, None)
            ax.set_xlim(xmin2, xmax2)
            ax.set_ylim(ymin2, ymax2)
        return (scat,) if goal_artist is None else (scat, goal_artist)

    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=1000/args.fps, blit=True)

    # Save
    out_lower = args.out.lower()
    try:
        if out_lower.endswith('.gif'):
            from matplotlib.animation import PillowWriter
            ani.save(args.out, writer=PillowWriter(fps=args.fps))
        elif out_lower.endswith('.mp4'):
            ani.save(args.out, writer='ffmpeg', fps=args.fps)
        else:
            # default to GIF if extension is unknown
            from matplotlib.animation import PillowWriter
            ani.save(args.out + '.gif', writer=PillowWriter(fps=args.fps))
            print(f"Unknown extension, wrote GIF: {args.out + '.gif'}")
    finally:
        plt.close(fig)

    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()
