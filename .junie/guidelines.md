# Particle‑Lenia Thesis — Development Guidelines (Project‑Specific)

This document captures verified, project‑specific details to accelerate future development. It focuses on how this repository actually behaves on a clean setup, how to run experiments headlessly, how to add and run quick tests, and what to watch out for.

Last verified on: 2025‑10‑26 (Windows, Python 3.13) by running a temporary smoke test (removed after verification).

---

## 1) Build / Configuration

- Python version: tested with Python 3.13; earlier 3.10–3.12 should work but weren’t re‑verified today.
- Dependencies: `requirements.txt` is authoritative. Install inside a virtual environment.
  - PowerShell (Windows):
    ```powershell
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install -r requirements.txt
    ```
- Display/SDL: `pygame` is used by the interactive scripts, but the headless simulations also import it indirectly (`particle_lenia_headless.py` imports `pygame`). No window is opened during headless runs; simply having `pygame` installed is sufficient. If you ever run on a true headless Linux CI and get SDL/display errors, set `SDL_VIDEODRIVER=dummy` in the environment.
- Project layout assumptions:
  - Imports expect the project root on `sys.path`. When running ad‑hoc scripts from a subfolder (e.g., `tests/`), add the project root to `sys.path` (see testing section) or invoke with `python -m` from the repo root.

---

## 2) Running The Simulations

- Interactive visualization (uses `pygame`): run any of the `simulations/*.py` files directly, e.g.:
  ```powershell
  python simulations\multi_particle_simulation.py
  ```
- Headless experiment pipeline: use `experiments\experiment_runner.py`. It will run predefined configurations for:
  - `ParticleLeniaSimulation` (headless)
  - `FoodHuntSimulation` (headless)
  and write a CSV to `results\experiment_results.csv`.
  ```powershell
  python experiments\experiment_runner.py
  ```

Notes:
- The experiment runner expects the `utils.metrics` functions (`stability_score`, `diversity_score`, `goal_completion`) and that `FoodHuntSimulation` exposes `food_pos` (both satisfied in this repo).
- Output directory `results/` is created automatically.

---

## 3) Testing Information

There is no dedicated test framework baked in; quick validation is easiest with small Python scripts. PyTest can be added optionally if needed, but is not required.

### 3.1 Minimal smoke test pattern (verified)

The following script was created and executed to verify the headless APIs and metrics. It was removed after verification, but you can recreate it anytime.

Create `tests/smoke_test.py` with:
```python
import sys, os
import numpy as np

# Ensure project root on path when running from tests/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from simulations.particle_lenia_headless import ParticleLeniaSimulation
from simulations.food_hunt_cell_headless import FoodHuntSimulation
from utils.metrics import stability_score, diversity_score, goal_completion


def run_particle_lenia_smoke():
    cfg = {"mu_k": 4.0, "sigma_k": 1.0, "w_k": 0.022, "c_rep": 1.0,
           "mu_g": 0.6, "sigma_g": 0.15, "dt": 0.1, "point_n": 64}
    sim = ParticleLeniaSimulation(cfg, seed=42)
    avg_E = sim.run_headless(steps=20)
    pts = sim.get_state()
    assert pts.shape == (cfg["point_n"], 2)
    s = stability_score(pts)
    d = diversity_score(pts)
    assert np.isfinite(avg_E) and np.isfinite(s) and np.isfinite(d)
    print(f"ParticleLenia OK: avg_E={avg_E:.4f}, stability={s:.4f}, diversity={d:.4f}")


def run_food_hunt_smoke():
    cfg = {"mu_k": 4.0, "sigma_k": 1.0, "w_k": 0.022, "c_rep": 1.0,
           "mu_g": 0.6, "sigma_g": 0.15, "dt": 0.1, "point_n": 32,
           "food_params": {"food_attraction_strength": 0.05,
                            "food_radius": 2.0,
                            "food_spawn_min_dist": 5.0}}
    sim = FoodHuntSimulation(cfg, seed=123)
    assert hasattr(sim, "food_pos")
    avg_E = sim.run_headless(steps=20)
    pts = sim.get_state()
    gc = goal_completion(pts, goal_pos=sim.food_pos)
    assert 0.0 <= gc <= 1.0
    print(f"FoodHunt OK: avg_E={avg_E:.4f}, goal_completion={gc:.3f}")


if __name__ == "__main__":
    run_particle_lenia_smoke()
    run_food_hunt_smoke()
    print("SMOKE TESTS PASSED")
```

Run from repo root:
```powershell
python tests\smoke_test.py
```

Observed (example) output during verification:
```
pygame 2.6.1 (SDL 2.28.4, Python 3.13.5)
Hello from the pygame community. https://www.pygame.org/contribute.html
ParticleLenia OK: avg_E=0.4710, stability=0.1322, diversity=148.5137
FoodHunt OK: avg_E=0.5062, goal_completion=0.000
SMOKE TESTS PASSED
```

You can safely increase `steps` for deeper checks. Keep `point_n` modest for quick runs.

### 3.2 Adding new tests

- Preferred approach for this repo is fast, self‑contained scripts that:
  - import a simulation class and `utils.metrics`
  - run a short `run_headless(steps=N)`
  - assert on shapes/finite values/metric ranges and any invariants (e.g., `goal_completion` in [0, 1])
- If you choose to use `pytest`, add it to your environment and place test files under `tests/`, ensuring the project root is discoverable (either run from repo root or add `sys.path` shim as shown above). Example command if pytest is installed:
  ```powershell
  pytest -q
  ```

---

## 4) Development Notes and Conventions

- Headless API surface (stable as of this snapshot):
  - `ParticleLeniaSimulation(config, seed=None)`
    - `run_headless(steps=...) -> float` returns average energy
    - `get_state() -> np.ndarray[(N, 2)]` particle positions
  - `FoodHuntSimulation(config, seed=None)`
    - `run_headless(steps=...) -> float` returns average energy
    - `get_state() -> np.ndarray[(N, 2)]`
    - `food_pos: np.ndarray[(2,)]` current food position (used for goal metrics)
- Configuration structure (keys that must be present):
  - Common: `mu_k`, `sigma_k`, `w_k`, `c_rep`, `mu_g`, `sigma_g`, `dt`, `point_n`
  - Food hunt specific (`food_params` dict): `food_attraction_strength`, `food_radius`, `food_spawn_min_dist`
- Determinism: both headless sims accept an optional `seed`; use it for reproducible runs.
- Numeric conventions:
  - Particle positions are stored as a flat `float32` array of length `2*N` and reshaped on demand. Utilities like `add_xy` assume this layout; keep it consistent when adding new forces.
  - `fast_exp` is a custom approximation; don’t replace casually without validating dynamics.
  - Metrics:
    - `stability_score(points)` uses mean stepwise displacement (lower movement → higher score via `1/(1+mean_norm)`)
    - `diversity_score(points)` returns `det(cov(points))`; may be sensitive if covariance is near‑singular — avoid degenerate states in tests.
    - `goal_completion(points, goal_pos)` = fraction of particles within radius 1.0 of `goal_pos`.
- Performance/scale tips:
  - Both headless sims currently compute pairwise interactions in O(N^2). Keep `point_n` moderate for experiments or consider spatial partitioning if you scale up.
- File I/O:
  - Experiment results are appended to `results\experiment_results.csv` by `experiments\experiment_runner.py` (folder auto‑created). Ensure the process has write permission.
- Imports and working directory:
  - When running from subdirectories, either:
    - add the repo root to `sys.path` (as shown in the smoke test), or
    - run with `python -m` from the repo root, or
    - configure your IDE run configuration’s working directory to the project root.

---

## 5) Troubleshooting

- `ModuleNotFoundError: No module named 'simulations'` when running a test from `tests/`:
  - Add the repo root to `sys.path` (see smoke test) or run from the root.
- SDL/pygame errors in CI:
  - Set `SDL_VIDEODRIVER=dummy` and avoid importing interactive modules; headless modules already avoid opening a window.
- Numerical blow‑ups (NaNs/inf in metrics):
  - Check for degenerate particle distributions; increase `sigma` values or reduce step size `dt`.

---

## 6) What was Verified Today

- Dependency install via `pip -r requirements.txt`.
- Headless runs for both `ParticleLeniaSimulation` and `FoodHuntSimulation` with small `point_n` and `steps`.
- Metrics functions integrate with the headless simulations.
- A working smoke test pattern; file removed after verification to keep the repo clean.
