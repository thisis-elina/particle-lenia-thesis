# Particle Lenia Simulations (Thesis Work)

This repository contains Python-based simulations inspired by **Particle Lenia**, an artificial life model originally created by **@znah** on **ObservableHQ** ([source](https://observablehq.com/@znah/particle-lenia-from-scratch)).  

These simulations explore **multi-particle interactions**, using **NumPy** and **Pygame** for real-time visualization. This project also supports headless runs for metric logging and parameter exploration.

---

# Project Overview

This repository contains Python-based simulations inspired by **Particle Lenia**, an artificial life model originally created by @znah on ObservableHQ. The simulations explore multi-particle interactions and cell-like agents in 2D environments, with a focus on energy dynamics, stability, diversity, and goal-directed behaviors.

## 📂 Project Structure
```sh
particle-lenia-thesis/
├── simulations/
│ ├── particle_lenia.py # Particle Lenia simulation (continuous particle interactions)
│ ├── food_hunt_cell.py # Cell simulation that hunts food
│ ├── food_hunt_cell_headless.py # Headless version for experiments
├── utils/
│ └── metrics.py # Functions to compute stability, diversity, goal completion
├── ExperimentRunner.py # Run experiments with multiple configurations
├── results/ # CSV results of headless experiments
└── README.md
```

---

## 🔬 Features

### **Simulations**
- **ParticleLeniaSimulation** (`simulations/particle_lenia.py`)  
  Core simulation of autonomous particles interacting via **attraction** and **repulsion**. Supports:
  - Interactive visualization via Pygame.
  - Headless runs for logging energy and metrics.

### **Metrics** (`utils/metrics.py`)
- **Stability score:** Measures how stable the particle configuration is over time.  
- **Diversity score:** Measures the spatial spread of particles.  
- **Goal completion:** Fraction of particles near a predefined goal.  

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

## 🔬 Acknowledgment
This project is inspired by **Particle Lenia** by **@znah**.  
The original implementation can be found [here](https://observablehq.com/@znah/particle-lenia-from-scratch).

---

## 🚀 Installation & Setup
### **1. Clone the Repository**
```sh
git clone https://github.com/YOUR_USERNAME/particle-lenia-thesis.git
cd particle-lenia-thesis
```

### **2. Create a Virtual Environment**
If using **PyCharm**, it should automatically detect the virtual environment. Otherwise, run:
```sh
python -m venv .venv
```

### **3. Activate the Virtual Environment**
- **Windows:**
```sh
python .venv\Scripts\activate
```
- **Mac/Linux:**
```sh
python .venv/bin/activate
```

### **4. Install Dependencies**
```sh
python pip install -r requirements.txt
```

---

## 🎮 Running the Simulations

### Interactive Visualization
Run the main simulation with Pygame:

```sh
python simulations/particle_lenia.py
```

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

## 🔬 My Analysis & Thesis Use

- Run ExperimentRunner.py for multiple configurations.

- Collect CSV metrics (avg_energy, stability, diversity, goal_completion) for each experiment.

- Use the data for plots, tables, and analysis in the thesis.

- Optionally, run individual scripts with Pygame to visualize particle dynamics.

---
## 🖼️ Preview

Will add screenshots later :>

---

## 📜 License

This project is licensed under the MIT License.

---
