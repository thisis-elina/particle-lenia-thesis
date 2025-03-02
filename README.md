# Particle Lenia Simulations (Thesis Work)

This repository contains Python-based simulations inspired by **Particle Lenia**, an artificial life model originally created by **@znah** on **ObservableHQ** ([source](https://observablehq.com/@znah/particle-lenia-from-scratch)).  

These simulations explore **multi-particle interactions**, using **NumPy** and **Pygame** for real-time visualization.

---

# Project Overview

This repository contains simulations of different particle systems with various dynamics and behaviors. Each script demonstrates how particles interact based on different forces and energy fields. The simulations are visualized using `pygame` and allow for interactive exploration of particle dynamics.

## 📌 Features

### **`particle_lenia.py`**
This Python implementation of the **Particle Lenia** model simulates life-like particles in a continuous 2D environment. The particles exhibit autonomous behavior by interacting with each other based on dynamic rules for attraction and repulsion. This model is a direct adaptation of the **Particle Lenia** concept originally written in JavaScript on ObservableHQ (https://observablehq.com/@znah/particle-lenia-from-scratch). 

### **`multi_particle_simulation.py`**
Simulates the interactions of multiple particle types, each with different behaviors and dynamic field interactions. Particles exhibit unique forces that affect their motion and positioning over time.

### **`two_glider_thing.py`**
Models two interacting glider-like structures within a particle system, where the particles move according to repulsive forces and potential energy fields. The simulation showcases the behavior of two groups of particles in real-time.

### **`food_hunt_cell.py`**
Simulates a simple cell that moves in a 2D environment to hunt for food. The cell is attracted to food sources and repels when too close. The particles move based on attractive and repulsive forces, and the simulation continuously updates the cell's position to explore the environment. Food is spawned randomly, and once the cell reaches it, new food appears.

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
Each script uses **Pygame** for visualization. Run any script using:

```sh
python multi_particle_simulation.py
```
Replace with any script name.

---

## 🖼️ Preview

Will add screenshots later :>

---

## 📜 License

This project is licensed under the MIT License.