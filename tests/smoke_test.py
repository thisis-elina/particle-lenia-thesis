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
    """Smoke test for Particle‑Lenia headless simulation.

    Checks: shape of state, finite metrics, and prints a short summary.
    """
    config = {"mu_k": 4.0, "sigma_k": 1.0, "w_k": 0.022, "c_rep": 1.0,
              "mu_g": 0.6, "sigma_g": 0.15, "dt": 0.1, "point_n": 64}
    simulation = ParticleLeniaSimulation(config, seed=42)
    average_energy = simulation.run_headless(steps=20)
    points = simulation.get_state()
    assert points.shape == (config["point_n"], 2), f"Unexpected shape: {points.shape}"
    stability = stability_score(points)
    diversity = diversity_score(points)
    assert np.isfinite(average_energy) and np.isfinite(stability) and np.isfinite(diversity)
    print(f"ParticleLenia OK: avg_energy={average_energy:.4f}, stability={stability:.4f}, diversity={diversity:.4f}")


def run_food_hunt_smoke():
    """Smoke test for Food‑Hunt headless simulation.

    Checks: exposes food_pos, finite metrics, and goal completion in [0, 1].
    """
    config = {"mu_k": 4.0, "sigma_k": 1.0, "w_k": 0.022, "c_rep": 1.0,
              "mu_g": 0.6, "sigma_g": 0.15, "dt": 0.1, "point_n": 32,
              "food_params": {"food_attraction_strength": 0.05,
                               "food_radius": 2.0,
                               "food_spawn_min_dist": 5.0}}
    simulation = FoodHuntSimulation(config, seed=123)
    assert hasattr(simulation, "food_pos")
    average_energy = simulation.run_headless(steps=20)
    points = simulation.get_state()
    goal = goal_completion(points, goal_pos=simulation.food_pos)
    assert 0.0 <= goal <= 1.0
    print(f"FoodHunt OK: avg_energy={average_energy:.4f}, goal_completion={goal:.3f}")


if __name__ == "__main__":
    run_particle_lenia_smoke()
    run_food_hunt_smoke()
    print("SMOKE TESTS PASSED")
