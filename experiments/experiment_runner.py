import csv
import os
import numpy as np
from simulations.particle_lenia import ParticleLeniaSimulation
from utils.metrics import stability_score, diversity_score, goal_completion

# Parameter sets
parameter_sets = [
    {"mu_k": 4.0, "sigma_k": 1.0, "w_k": 0.022, "c_rep": 1.0, "mu_g":0.6,"sigma_g":0.15,"dt":0.1,"point_n":200},
    {"mu_k": 3.5, "sigma_k": 0.8, "w_k": 0.03, "c_rep": 1.2, "mu_g":0.6,"sigma_g":0.15,"dt":0.1,"point_n":200},
]

results_file = "../results/experiment_results.csv"
os.makedirs(os.path.dirname(results_file), exist_ok=True)

with open(results_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["config_index", "avg_energy", "stability", "diversity", "goal_completion"])

    for idx, params in enumerate(parameter_sets):
        sim = ParticleLeniaSimulation(params)
        avg_energy = sim.run_headless(steps=500)
        points = sim.get_state()

        stability = stability_score(points)
        diversity = diversity_score(points)
        goal = goal_completion(points, goal_pos=np.array([0.0,0.0]))

        writer.writerow([idx, avg_energy, stability, diversity, goal])
        print(f"Config {idx} done, avg_energy={avg_energy:.5f}, stability={stability:.3f}, diversity={diversity:.3f}, goal={goal:.3f}")
