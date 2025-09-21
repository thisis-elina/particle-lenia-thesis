# Particle Lenia Simulations (Thesis Work)

This repository contains Python-based simulations inspired by **Particle Lenia**, an artificial life model originally created by **@znah** on **ObservableHQ** ([source](https://observablehq.com/@znah/particle-lenia-from-scratch)).  

These simulations explore **multi-particle interactions**, using **NumPy** and **Pygame** for real-time visualization. This project also supports headless runs for metric logging and parameter exploration.

---

# Project Overview

This repository contains simulations of different particle systems with various dynamics and behaviors. Each script demonstrates how particles interact based on different forces and energy fields. The simulations are visualized using `pygame` and allow for interactive exploration of particle dynamics.

particle-lenia-thesis/
│
├── simulations/
│   └── particle_lenia.py         # Core Particle Lenia simulation
│
├── utils/
│   └── metrics.py               # Stability, diversity, goal completion
│
├── experiments/
│   └── experiment_runner.py     # Runs headless experiments and logs metrics
│
├── results/                     # Stores experiment CSV outputs
│
├── README.md
├── requirements.txt
└── .gitignore

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

### Headless Experiments & Metrics
- Run the main simulation with Pygame:

```sh
python experiments/experiment_runner.py
```
- Results are saved in results/experiment_results.csv.
- Use this for thesis experiments and metric analysis.
- 
---

## 🖼️ Preview

Will add screenshots later :>

---

## 📜 License

This project is licensed under the MIT License.

---
