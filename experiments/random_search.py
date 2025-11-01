"""
Random search CLI for Particle‑Lenia and Food‑Hunt (headless) simulations.

Key features:
- JSON‑driven sweeps via --fixed-config and --sweep (supports nested keys using dot notation)
- Multi‑model runs via --modes (comma‑separated)
- Deterministic sampling given a base --seed
- Writes tidy CSV with config JSON + metrics for ranking/analysis

Usage examples are in README.md.
"""

import os
import sys
import csv
import json
import time
import argparse
import random
import math
import numpy as np
from typing import Any, Dict

# Ensure project root is on sys.path when running as a script from experiments/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from simulations.particle_lenia_headless import ParticleLeniaSimulation
from simulations.food_hunt_cell_headless import FoodHuntSimulation
from utils.metrics import stability_score, diversity_score, goal_completion

SIMULATION_CLASSES = {
    "particle-lenia": ParticleLeniaSimulation,
    "food-hunt": FoodHuntSimulation,
}

# Default ranges for sampler when no JSON sweep is provided
PARTICLE_LENIA_RANGES = {
    "mu_k": (3.0, 5.0),
    "sigma_k": (0.6, 1.4),
    "w_k": (0.01, 0.04),
    "c_rep": (0.8, 1.5),
    "mu_g": (0.4, 0.8),
    "sigma_g": (0.10, 0.25),
    "dt": (0.05, 0.15),
}

FH_RANGES = {
    **PARTICLE_LENIA_RANGES,
    "food_params.food_attraction_strength": (0.03, 0.12),
    "food_params.food_radius": (1.5, 3.0),
    "food_params.food_spawn_min_dist": (4.0, 8.0),
}

# User-friendly alias keys → canonical simulation keys.
# Dot-notation is supported for nested fields.
ALIASES = {
    # core dynamics
    "kernel_mu": "mu_k",
    "interaction_kernel_mu": "mu_k",
    "kernel_sigma": "sigma_k",
    "kernel_weight": "w_k",
    "repulsion_coefficient": "c_rep",
    "growth_mu": "mu_g",
    "growth_sigma": "sigma_g",
    "time_step": "dt",
    "point_count": "point_n",
    # food hunt nested under food_params
    "food_params.attraction_strength": "food_params.food_attraction_strength",
    "food_params.goal_radius": "food_params.food_radius",
    "food_params.spawn_min_distance": "food_params.food_spawn_min_dist",
}


def _uniform(a: float, b: float) -> float:
    """Sample a float uniformly in [a, b].

    Parameters
    ----------
    a, b : float
        Lower and upper bounds.
    """
    return random.uniform(a, b)


def _midpoint(a: float, b: float) -> float:
    """Midpoint helper used to derive deterministic default values."""
    return (a + b) / 2.0


def _deep_set(d: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a value in a (possibly) nested dict using dot notation.

    Example: _deep_set(cfg, "food_params.food_radius", 2.0)
    will ensure cfg["food_params"]["food_radius"] == 2.0
    """
    parts = dotted_key.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _deep_has(d: Dict[str, Any], dotted_key: str) -> bool:
    """Check if a dotted path exists in a nested dict.

    Returns True if the final key is present; does not validate value type.
    """
    parts = dotted_key.split(".")
    cur = d
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            return False
        cur = cur[p]
    return parts[-1] in cur


def _apply_fixed(base: Dict[str, Any], fixed: Dict[str, Any]) -> None:
    """Apply fixed overrides onto a base config.

    Supports either nested dicts (e.g., {"food_params": {"food_radius": 2.0}})
    or dotted keys (e.g., {"food_params.food_radius": 2.0}).
    """
    # fixed can have nested dicts or dotted keys
    for k, v in fixed.items():
        if isinstance(v, dict) and not any(ch in k for ch in ["."]):
            # nested dict; recurse into keys
            for kk, vv in v.items():
                _deep_set(base, f"{k}.{kk}", vv)
        else:
            _deep_set(base, k, v)


def _iter_dotted_items(mapping: Dict[str, Any], prefix: str = ""):
    """Yield (dotted_key, value) pairs for leaves in a possibly nested dict.

    For sweep specs, a dict that contains any of {"uniform","choice","const"}
    is treated as a leaf (spec dict). For fixed configs, any non-dict is a leaf.
    """
    for k, v in mapping.items():
        dotted = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict) and not set(v.keys()) & {"uniform", "choice", "const"}:
            # structural nesting (e.g., food_params)
            yield from _iter_dotted_items(v, dotted)
        else:
            yield dotted, v


def _normalize_alias_keys(mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Return a new flat dict with dotted canonical keys from a mapping that
    may use alias keys (nested or dotted). If both alias and canonical forms
    are provided, the canonical key wins; a note is printed to console.
    """
    flat: Dict[str, Any] = {}
    if not mapping:
        return flat
    # Flatten and translate aliases
    for dotted, value in _iter_dotted_items(mapping):
        canonical = ALIASES.get(dotted, dotted)
        if canonical in flat and canonical != dotted:
            # Prefer existing canonical; only warn if alias conflicts
            print(f"[aliases] Both alias '{dotted}' and canonical '{canonical}' provided; using canonical value.")
            continue
        flat[canonical] = value
    return flat


def _sample_from_spec(spec: Dict[str, Any]) -> Any:
    """Sample a value from a sweep specification.

    Supported forms:
    - {"uniform": [a, b]} → float in [a, b]
    - {"choice": [v1, v2, ...]} → one element from the list
    - {"const": v} → always v
    If a raw value is passed instead of a dict, it is treated as const.
    """
    if "uniform" in spec:
        a, b = spec["uniform"]
        return _uniform(float(a), float(b))
    if "choice" in spec:
        choices = spec["choice"]
        return random.choice(choices)
    if "const" in spec:
        return spec["const"]
    # Fallback: treat as const if raw value passed
    # e.g., spec = 0.1
    if not isinstance(spec, dict):
        return spec
    raise ValueError(f"Unrecognized sweep spec: {spec}")


def _default_config(sim_name: str) -> Dict[str, Any]:
    """Build a deterministic default config using midpoints of ranges.

    For Food‑Hunt, includes default values for food_params.* as well.
    """
    cfg: Dict[str, Any] = {}
    ranges = PARTICLE_LENIA_RANGES
    for k, (a, b) in ranges.items():
        _deep_set(cfg, k, _midpoint(a, b))
    if sim_name == "food-hunt":
        for k, (a, b) in FH_RANGES.items():
            if k.startswith("food_params."):
                _deep_set(cfg, k, _midpoint(a, b))
    return cfg


def build_config(sim_name: str,
                 point_n_choices,
                 fixed_config: Dict[str, Any] | None,
                 sweep_spec: Dict[str, Any] | None) -> Dict[str, Any]:
    """Construct a single config for a given simulation mode.

    Priority order for values:
    1) fixed_config (explicit overrides, held constant)
    2) sweep_spec (sampled per trial for keys present)
    3) deterministic defaults (midpoints of built-in ranges)

    point_n is handled specially: if not provided by fixed/sweep, it is drawn
    from the provided `point_n_choices`.
    """
    # Start from defaults (midpoints) to have deterministic base
    cfg = _default_config(sim_name)

    # point_n handled specially: remove any default so we can control it below
    cfg.pop("point_n", None)

    # Apply fixed overrides if provided
    if fixed_config:
        _apply_fixed(cfg, fixed_config)

    # Apply sweep sampling if provided
    if sweep_spec:
        for dotted_key, rule in sweep_spec.items():
            sampled = _sample_from_spec(rule)
            _deep_set(cfg, dotted_key, sampled)
    else:
        # If no sweep provided, use legacy randomized sampling across defaults
        for k, (a, b) in PARTICLE_LENIA_RANGES.items():
            if k == "point_n":
                continue
            if _deep_has(cfg, k):
                _deep_set(cfg, k, _uniform(a, b))
        if sim_name == "food-hunt":
            for k, (a, b) in FH_RANGES.items():
                if k.startswith("food_params."):
                    _deep_set(cfg, k, _uniform(a, b))

    # Ensure point_n is set: fixed > sweep > fallback choices
    if not _deep_has(cfg, "point_n"):
        # Allow sweep spec to set point_n (if provided)
        if sweep_spec and "point_n" in sweep_spec:
            _deep_set(cfg, "point_n", _sample_from_spec(sweep_spec["point_n"]))
        else:
            _deep_set(cfg, "point_n", int(random.choice(point_n_choices)))

    # Clean up: if no food_params for particle-lenia, ensure it's absent
    if sim_name != "food-hunt":
        cfg.pop("food_params", None)

    return cfg


def main():
    ap = argparse.ArgumentParser(description="Random search over Particle‑Lenia / Food‑Hunt with JSON‑driven sweeps.")
    # Backward compatible single-mode; new plural form supports multiple modes
    ap.add_argument("--mode", choices=list(SIMULATION_CLASSES.keys()), help="Single simulation mode to run (backward compatible)")
    ap.add_argument("--modes", type=str, default="", help="Comma‑separated list of modes to run (e.g., particle-lenia,food-hunt)")
    ap.add_argument("--trials", type=int, default=50, help="Number of random configurations per mode")
    ap.add_argument("--steps", type=int, default=300, help="Headless steps per trial")
    ap.add_argument("--seed", type=int, default=123, help="Base seed; per‑trial uses seed+t")
    ap.add_argument("--out", type=str, default="results/random_search.csv", help="Output CSV path")
    ap.add_argument("--pointn", type=str, default="64,100,150,200", help="Comma‑separated point_n choices (used if not set by fixed/sweep)")
    ap.add_argument("--fixed-config", dest="fixed_path", type=str, default="", help="Path to JSON with fixed overrides (held constant)")
    ap.add_argument("--sweep", dest="sweep_path", type=str, default="", help="Path to JSON with sweep spec (uniform/choice/const)")
    args = ap.parse_args()

    # Determine modes list
    modes = []
    if args.modes:
        modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    elif args.mode:
        modes = [args.mode]
    else:
        raise SystemExit("Please specify --mode or --modes")

    # Load JSONs if provided
    fixed_cfg: Dict[str, Any] | None = None
    sweep_spec: Dict[str, Any] | None = None
    if args.fixed_path:
        with open(args.fixed_path, "r") as f:
            user_fixed = json.load(f)
            fixed_cfg = _normalize_alias_keys(user_fixed)
    if args.sweep_path:
        with open(args.sweep_path, "r") as f:
            user_sweep = json.load(f)
            sweep_spec = _normalize_alias_keys(user_sweep)

    if fixed_cfg:
        print("[aliases] Normalized fixed-config keys:", sorted(list(fixed_cfg.keys())))
    if sweep_spec:
        print("[aliases] Normalized sweep keys:", sorted(list(sweep_spec.keys())))

    point_n_choices = tuple(int(x) for x in args.pointn.split(",") if x)

    # Determinism for sampler and any numpy usage within sampling scope.
    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "simulation", "seed", "steps", "config",
            "avg_energy", "stability", "diversity", "goal_completion", "elapsed_sec"
        ])

        for mode in modes:
            Sim = SIMULATION_CLASSES[mode]
            for t in range(args.trials):
                cfg = build_config(mode, point_n_choices, fixed_cfg, sweep_spec)
                run_seed = args.seed + t
                sim = Sim(cfg, seed=run_seed)

                t0 = time.time()
                avg_E = sim.run_headless(steps=args.steps)
                pts = sim.get_state()
                stab = stability_score(pts)
                div = diversity_score(pts)
                if mode == "food-hunt":
                    goal = goal_completion(pts, goal_pos=sim.food_pos)
                else:
                    goal = float("nan")
                elapsed = time.time() - t0

                w.writerow([
                    mode, run_seed, args.steps, json.dumps(cfg),
                    avg_E, stab, div, goal, elapsed
                ])
                print(f"[{mode}] [{t+1}/{args.trials}] seed={run_seed} E={avg_E:.4f} stab={stab:.3f} div={div:.3f} goal={goal} dt={elapsed:.2f}s")


if __name__ == "__main__":
    main()
