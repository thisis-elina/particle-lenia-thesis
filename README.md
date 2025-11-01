# Particle Lenia – Thesis Black‑Box Experiments

This repository contains Python simulations inspired by Particle‑Lenia with both interactive and headless (non‑visual) modes. It now includes a modular, JSON‑driven "black‑box" for automated parameter sweeps and a ranking utility for selecting top configurations.

- Interactive sims (pygame): `simulations/interactive/*.py`
- Headless sims (programmatic): `simulations/*_headless.py`
- Baseline experiments: `experiments/experiment_runner.py`
- Random search CLI (JSON sweeps, multi‑model): `experiments/random_search.py`
- Result ranking: `experiments/rank_results.py`
- Metrics: `utils/metrics.py`
- Quick tests: `tests/*.py`

---

## What’s new (black‑box features)
- JSON‑driven configuration and multi‑model runs in `experiments/random_search.py`:
  - `--modes` to run multiple simulations in one command (e.g., `particle-lenia,food-hunt`). Backward‑compatible `--mode` still works.
  - `--fixed-config <path.json>` for keys held constant (supports nested keys and dot notation like `food_params.food_radius`).
  - `--sweep <path.json>` for variables that change per trial with simple schemas: `{ "uniform": [a, b] }`, `{ "choice": [...] }`, or `{ "const": v }`.
  - Deterministic defaults: if `--sweep` is omitted, behavior matches the previous random search (uniform over default ranges). If `--sweep` is provided, only those keys vary; others come from `--fixed-config` or deterministic midpoints of defaults. `point_n` is set by fixed/sweep or chosen from `--pointn`.
- Result ranking utility `experiments/rank_results.py`:
  - Composite score (z‑scored diversity and stability), constrained objective (stability/diversity floors), and Food‑Hunt goal objective.
  - Exports ranked CSV and optional Top‑K JSON for re‑runs.
- Quick validation scripts in `tests/` for smoke, determinism, and metric bounds.

These additions do not change core simulation logic.

---

## Naming conventions and terminology
To keep code and thesis language aligned, the simulations and tests use descriptive names:
- Potentials (how much interaction a particle accumulates): `repulsion_potential`, `attraction_potential`.
- Direction sums (which way to move): `repulsion_direction_sum`, `attraction_direction_sum`.
- Common script variables: `config`, `simulation`, `average_energy`, `points`, `stability`, `diversity`, `goal`.

In the headless update step, velocity is computed as `v = dG * attraction_direction_sum - repulsion_direction_sum`,
where `dG` comes from the global growth kernel applied to the `attraction_potential`.

---

## Setup (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```
If you run in a true headless Linux CI and SDL complains, set `SDL_VIDEODRIVER=dummy`.

### **Experiments** (`experiments/experiment_runner.py`)
- Run multiple parameter configurations without Pygame.
- Automatically logs metrics and average energy to CSV.
- Example output:
```sh
config_index,avg_energy,stability,diversity,goal_completion
0,-0.08210806,0.954,1.234,0.120
1,0.02232304,0.872,0.987,0.050
```
---

## Run interactive simulations (pygame)
From the repo root:
```powershell
python simulations\interactive\multi_particle_simulation.py
```
Replace with any file under `simulations/interactive/` (any file not suffixed `_headless.py`).

---

## Baseline headless experiments
Runs predefined configs and appends metrics to `results/experiment_results.csv`.
```powershell
python experiments\experiment_runner.py
```

---

## Random search (JSON‑driven sweeps)
Run automated sweeps that log full configs + metrics to CSV.

### 1) Prepare JSONs (you may use either canonical keys or human‑friendly aliases)
Both forms below are accepted. The tool normalizes aliases to canonical keys internally, so your CSVs remain consistent.

- fixed.json (held constant) — canonical keys:
```json
{ "mu_k": 4.0, "sigma_k": 1.0, "w_k": 0.022, "c_rep": 1.0, "mu_g": 0.6, "sigma_g": 0.15, "dt": 0.1, "point_n": 100 }
```
- fixed.json — alias keys (equivalent to the above):
```json
{ "kernel_mu": 4.0, "kernel_sigma": 1.0, "kernel_weight": 0.022, "repulsion_coefficient": 1.0, "growth_mu": 0.6, "growth_sigma": 0.15, "time_step": 0.1, "point_count": 100 }
```
- sweep.json (variables to change) — canonical keys:
```json
{
  "mu_k": {"uniform": [3.0, 5.0]},
  "sigma_k": {"uniform": [0.6, 1.4]},
  "w_k": {"uniform": [0.01, 0.04]},
  "c_rep": {"uniform": [0.8, 1.5]},
  "mu_g": {"uniform": [0.4, 0.8]},
  "sigma_g": {"uniform": [0.10, 0.25]},
  "dt": {"uniform": [0.05, 0.15]},
  "point_n": {"choice": [64, 100, 150, 200]}
}
```
- sweep.json — alias keys (equivalent):
```json
{
  "kernel_mu": {"uniform": [3.0, 5.0]},
  "kernel_sigma": {"uniform": [0.6, 1.4]},
  "kernel_weight": {"uniform": [0.01, 0.04]},
  "repulsion_coefficient": {"uniform": [0.8, 1.5]},
  "growth_mu": {"uniform": [0.4, 0.8]},
  "growth_sigma": {"uniform": [0.10, 0.25]},
  "time_step": {"uniform": [0.05, 0.15]},
  "point_count": {"choice": [64, 100, 150, 200]}
}
```
- Food‑Hunt additions (only when running `food-hunt`) — canonical keys:
```json
{
  "food_params.food_attraction_strength": {"uniform": [0.03, 0.12]},
  "food_params.food_radius": {"uniform": [1.5, 3.0]},
  "food_params.food_spawn_min_dist": {"uniform": [4.0, 8.0]}
}
```
- Food‑Hunt additions — alias keys (equivalent):
```json
{
  "food_params.attraction_strength": {"uniform": [0.03, 0.12]},
  "food_params.goal_radius": {"uniform": [1.5, 3.0]},
  "food_params.spawn_min_distance": {"uniform": [4.0, 8.0]}
}
```

### 2) Run sweeps
- Particle‑Lenia only:
```powershell
python experiments\random_search.py --mode particle-lenia --trials 60 --steps 300 ^
  --fixed-config fixed.json --sweep sweep.json --out results\plenia_sweep.csv
```
- Food‑Hunt only:
```powershell
python experiments\random_search.py --mode food-hunt --trials 60 --steps 300 ^
  --fixed-config fixed.json --sweep sweep.json --out results\food_sweep.csv
```
- Both models in one go:
```powershell
python experiments\random_search.py --modes particle-lenia,food-hunt --trials 60 --steps 300 ^
  --fixed-config fixed.json --sweep sweep.json --out results\both_sweep.csv
```
Notes:
- If `--sweep` is omitted, the script samples over built‑in ranges (backward compatible).
- If `--fixed-config` is omitted, base values are set to midpoints of default ranges (deterministic).

---

## Rank results by objective
Create a ranked CSV and optional Top‑K JSON for re‑runs.

- Particle‑Lenia composite objective (`J = z(diversity) + λ·z(stability)`, λ=0.6 default):
```powershell
python experiments\rank_results.py --in results\plenia_sweep.csv --mode particle-lenia ^
  --objective composite --lambda 0.6 --out results\plenia_ranked.csv --topk-json results\plenia_topk.json
```
- Constrained objective (require stability ≥ 0.10 and diversity floor):
```powershell
python experiments\rank_results.py --in results\plenia_sweep.csv --mode particle-lenia ^
  --objective constrained --stab-min 0.10 --div-floor 1e-6 --lambda 0.6 ^
  --out results\plenia_ranked_c.csv --topk-json results\plenia_topk_c.json
```
- Food‑Hunt (prioritize goal completion with diversity floor):
```powershell
python experiments\rank_results.py --in results\food_sweep.csv --mode food-hunt ^
  --objective food-goal --div-floor 1e-6 --out results\food_ranked.csv --topk-json results\food_topk.json
```

Artifacts:
- Ranked CSV: `rank, simulation, seed, steps, score, avg_energy, stability, diversity, goal_completion, elapsed_sec, config`.
- Top‑K JSON: exact configs + seeds to reproduce.

---

## Quick tests (optional but recommended)
Run from repo root:
```powershell
python tests\smoke_test.py
python tests\determinism_test.py
python tests\metrics_bounds_test.py
```
What they check:
- Smoke: both headless sims run, shapes and metrics are sane.
- Determinism: same config + seed → identical results.
- Metric bounds: diversity ≥ 0, goal completion in [0,1], stability finite.

## 🚀 Running Experiments (Headless)
- We provide an **ExperimentRunner** to run multiple simulations without visualization and log metrics for analysis.
- 
```sh
python experiments/experiment_runner.py
```
- This will generate:
```sh
results/experiment_results.csv
```
with following columns:
```sh
| simulation    | config\_index | avg\_energy | stability | diversity | goal\_completion |
| ------------- | ------------- | ----------- | --------- | --------- | ---------------- |
| ParticleLenia | 0             | -0.0821     | 0.965     | 1.23      | NaN              |
| ParticleLenia | 1             | 0.0223      | 0.872     | 1.45      | NaN              |
| FoodHunt      | 0             | -0.0501     | 0.912     | 1.12      | 0.87             |
| FoodHunt      | 1             | 0.0102      | 0.893     | 1.18      | 0.91             |
```
- Metrics explained:

**avg_energy:** Mean system energy over the simulation.

**stability:** Measures how stable particle positions are over time (lower movement = higher stability).

**diversity:** Spatial spread of particles.

**goal_completion:** Fraction of particles that reach the target (only for FoodHunt).

---

## 🛠️ Adding New Configurations

- You can extend experiments by modifying experiment_sets in ExperimentRunner.py:
```sh
experiment_sets = [
    {
        "name": "ParticleLenia",
        "sim_class": ParticleLeniaSimulation,
        "configs": [
            {"mu_k": 4.0, "sigma_k": 1.0, "w_k": 0.022, "c_rep": 1.0,
             "mu_g":0.6, "sigma_g":0.15, "dt":0.1, "point_n":200},
            # Add more parameter sets here
        ]
    },
    {
        "name": "FoodHunt",
        "sim_class": FoodHuntSimulation,
        "configs": [
            {"mu_k": 3.5, "sigma_k":0.8, "w_k":0.03, "c_rep":1.2,
             "mu_g":0.6, "sigma_g":0.15, "dt":0.1, "point_n":100,
             "food_params":{"food_attraction_strength":0.1,
                            "food_radius":2.0,
                            "food_spawn_min_dist":5.0}},
        ]
    }
]

```
- Simply add new parameter sets to run experiments in batch mode.
  
---

## Repeatability: how it works
- Deterministic seeding: the Random Search CLI uses a base `--seed`; trial `t` runs with `seed + t` so sampling and simulation are deterministic.
- Logged configuration: every row in the CSV includes the exact `config` JSON used (canonical keys), plus `simulation`, `seed`, `steps`, and `elapsed_sec`.
- Fixed steps and mode: `--steps` and `--mode/--modes` are stored with results to pin down the execution context.
- Stable API: headless sims expose `run_headless(steps)` and `get_state()` so trials can be reproduced programmatically.

## Why repeatability matters
- Scientific validity: claims and figures in your thesis can be reproduced on another machine.
- Comparison and re‑analysis: you can re‑rank the same runs with a new objective or regenerate plots without rerunning sweeps.
- Debugging: surprising runs can be replayed exactly to inspect behavior.

## How to perform a repeatable sweep here
1) Define what varies vs. what stays fixed using JSON (aliases allowed):
- fixed.json (held constant)
```json
{ "kernel_mu": 4.0, "kernel_sigma": 1.0, "kernel_weight": 0.022, "repulsion_coefficient": 1.0, "growth_mu": 0.6, "growth_sigma": 0.15, "time_step": 0.1, "point_count": 100 }
```
- sweep.json (varies per trial)
```json
{
  "kernel_mu": {"uniform": [3.0, 5.0]},
  "kernel_sigma": {"uniform": [0.6, 1.4]},
  "kernel_weight": {"uniform": [0.01, 0.04]},
  "repulsion_coefficient": {"uniform": [0.8, 1.5]},
  "growth_mu": {"uniform": [0.4, 0.8]},
  "growth_sigma": {"uniform": [0.10, 0.25]},
  "time_step": {"uniform": [0.05, 0.15]},
  "point_count": {"choice": [64, 100, 150, 200]}
}
```
2) Run with an explicit seed and steps
```powershell
python experiments\random_search.py --mode particle-lenia --trials 60 --steps 300 --seed 123 ^
  --fixed-config fixed.json --sweep sweep.json --out results\plenia_sweep.csv
```
3) (Optional) Rank by objective
```powershell
python experiments\rank_results.py --in results\plenia_sweep.csv --mode particle-lenia ^
  --objective composite --lambda 0.6 --out results\plenia_ranked.csv --topk-json results\plenia_topk.json
```

## How to reproduce a specific trial later
Given a CSV row with `simulation`, `seed`, `steps`, and `config` (JSON), you can replay it:
```python
import json
from simulations.particle_lenia_headless import ParticleLeniaSimulation

# Paste values from one CSV row
row_config_json = "{""mu_k"": 4.0, ""sigma_k"": 1.0, ""w_k"": 0.022, ""c_rep"": 1.0, ""mu_g"": 0.6, ""sigma_g"": 0.15, ""dt"": 0.1, ""point_n"": 100}"
config = json.loads(row_config_json)
seed = 137
steps = 300

sim = ParticleLeniaSimulation(config, seed=seed)
avg_E = sim.run_headless(steps=steps)
pts = sim.get_state()
print(avg_E, pts.shape)
```
For Food‑Hunt, import `FoodHuntSimulation` and pass the same `config`, `seed`, and `steps` from the CSV.

## Reproducibility tips
- Record your environment next to CSVs (e.g., save a `RUN_INFO.txt` with `python --version` and `pip freeze`).
- Commit your `fixed.json` and `sweep.json` to version control.
- Use the same `--seed` and `--steps` when re‑running.
- Run commands from the repo root so imports resolve (`simulations/...`).

---

## Troubleshooting
- `ModuleNotFoundError: simulations` → run from the repo root or add the root to `sys.path` (see `tests/smoke_test.py`).
- SDL/pygame errors on headless Linux CI → set `SDL_VIDEODRIVER=dummy`.
- Numerical issues (NaN/Inf) → reduce `dt` or increase `sigma*`; avoid degenerate initial states.

---

## Acknowledgments
- Particle‑Lenia inspiration by @znah (ObservableHQ): https://observablehq.com/@znah/particle-lenia-from-scratch

## License
MIT License.


---

## Visualizations (figures, time‑series, animations)

Figures output directories (created automatically):
- `results/figures/plenia/` — Particle‑Lenia plots (e.g., `plenia_scatter.png`)
- `results/figures/food/` — Food‑Hunt plots (e.g., `food_goal_hist.png`)
- `results/figures/timeseries/` — time‑series plots from replays
- `results/figures/animations/` — GIF/MP4 animations

Generate the two main thesis figures (scatter + histogram) in one command:
```powershell
python experiments\visualizations\make_figures.py --plenia results\plenia_sweep.csv --food results\food_no_respawn_p64_steps1800.csv
```
Outputs:
- `results/figures/plenia/plenia_scatter.png`
- `results/figures/food/food_goal_hist.png`

Time‑series (Food‑Hunt)
1) Replay a row to produce a per‑step CSV:
```powershell
python experiments\replay_food_timeseries.py --from-csv results\food_no_respawn_p64_steps1800.csv --row-index 2 --out results\artifacts\food_row2_timeseries.csv
```
2) Plot the time‑series into the figures folder:
```powershell
python experiments\plot_food_timeseries.py --in results\artifacts\food_row2_timeseries.csv --out results\figures\timeseries\food_row2_timeseries.png --title "Food-Hunt row 2" --show-dist
```

Animations (moving circles)
- Food‑Hunt (goal always kept in frame by default):
```powershell
python experiments\visualizations\animate_simulation.py --mode food-hunt --from-csv results\food_no_respawn_p64_steps1800.csv --row-index 2 --steps 1200 --frame-stride 4 --fps 20 --out results\figures\animations\food_row2_anim.gif
```
Notes:
- Add `--lock-extent` to keep a fixed camera; optionally pair with `--extent xmin xmax ymin ymax`.
- Use `.gif` (no extra deps) or `.mp4` (requires ffmpeg installed).

Tips:
- Keep commands single‑line in PowerShell (avoid `^`); for multi‑line use backticks (`` ` ``) as the last character.
- Always run from the repo root so relative paths resolve.
