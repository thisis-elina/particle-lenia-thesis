"""
Baseline headless experiment runner.

Purpose:
- Provide a tiny, readable example of how to instantiate the headless
  simulations, run them for a fixed number of steps, compute metrics,
  and write a CSV.
- Kept as a simple baseline alongside the more flexible random_search CLI.

Outputs:
- results/experiment_results.csv with columns:
  [simulation, config_index, avg_energy, stability, diversity, goal_completion]
"""

import os
import sys
import csv
import numpy as np

# Ensure project root on sys.path when run as a script from experiments/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from simulations.particle_lenia_headless import ParticleLeniaSimulation
from simulations.food_hunt_cell_headless import FoodHuntSimulation
from utils.metrics import stability_score, diversity_score, goal_completion

# =============================
# Experiment configurations
# =============================
experiment_sets = [
    {
        "name": "ParticleLenia",
        "sim_class": ParticleLeniaSimulation,
        "configs": [
            {"mu_k": 4.0, "sigma_k": 1.0, "w_k": 0.022, "c_rep": 1.0,
             "mu_g": 0.6, "sigma_g": 0.15, "dt": 0.1, "point_n": 200},
            {"mu_k": 3.5, "sigma_k": 0.8, "w_k": 0.03, "c_rep": 1.2,
             "mu_g": 0.6, "sigma_g": 0.15, "dt": 0.1, "point_n": 200},
        ]
    },
    {
        "name": "FoodHunt",
        "sim_class": FoodHuntSimulation,
        "configs": [
            {"mu_k": 4.0, "sigma_k": 1.0, "w_k": 0.022, "c_rep": 1.0,
             "mu_g": 0.6, "sigma_g": 0.15, "dt": 0.1, "point_n": 100,
             "food_params": {"food_attraction_strength": 0.1,
                             "food_radius": 2.0,
                             "food_spawn_min_dist": 5.0}},
            {"mu_k": 3.5, "sigma_k": 0.8, "w_k": 0.03, "c_rep": 1.2,
             "mu_g": 0.6, "sigma_g": 0.15, "dt": 0.1, "point_n": 100,
             "food_params": {"food_attraction_strength": 0.1,
                             "food_radius": 2.0,
                             "food_spawn_min_dist": 5.0}},
        ]
    }
]

results_file = "results/experiment_results.csv"
os.makedirs(os.path.dirname(results_file), exist_ok=True)

# =============================
# Run experiments
# =============================
with open(results_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["simulation", "config_index", "avg_energy", "stability", "diversity", "goal_completion"])

    for exp in experiment_sets:
        sim_name = exp["name"]
        SimClass = exp["sim_class"]

        for idx, config in enumerate(exp["configs"]):
            print(f"Running {sim_name} config {idx}...")
            sim = SimClass(config)

            # Run headless simulation
            avg_energy = sim.run_headless(steps=500)
            points = sim.get_state()

            # Metrics
            stability = stability_score(points)
            diversity = diversity_score(points)

            # For FoodHuntSimulation, compute goal completion using the last food position
            if sim_name == "FoodHunt":
                goal = goal_completion(points, goal_pos=sim.food_pos)
            else:
                goal = np.nan  # Not applicable for ParticleLenia

            # Save results
            writer.writerow([sim_name, idx, avg_energy, stability, diversity, goal])
            print(f"Done: avg_energy={avg_energy:.5f}, stability={stability:.3f}, diversity={diversity:.3f}, goal={goal}")
