import numpy as np
import math
import random

# =============================
# Core Simulation: Food Hunt
# =============================

class FoodHuntSimulation:
    def __init__(self, config, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Particle & food parameters
        self.params = config
        self.point_n = config.get("point_n", 100)
        self.steps_per_frame = config.get("steps_per_frame", 10)
        self.world_width = config.get("world_width", 25.0)

        self.food_params = config.get("food_params", {
            "food_attraction_strength": 0.1,
            "food_radius": 2.0,
            "food_spawn_min_dist": 5.0
        })

        # Particle positions
        self.points = np.zeros(self.point_n * 2, dtype=np.float32)
        self.init_points()

        # Fields for forces
        self.fields = {
            "R_val": np.zeros(self.point_n, dtype=np.float32),
            "U_val": np.zeros(self.point_n, dtype=np.float32),
            "R_grad": np.zeros(self.point_n * 2, dtype=np.float32),
            "U_grad": np.zeros(self.point_n * 2, dtype=np.float32),
        }

        # Spawn initial food
        self.food_pos = self.spawn_food()

    # -----------------------------
    # Initialization
    # -----------------------------
    def init_points(self):
        for i in range(self.point_n):
            self.points[i*2] = (random.random() - 0.5) * 12
            self.points[i*2+1] = (random.random() - 0.5) * 12

    # -----------------------------
    # Math helpers
    # -----------------------------
    def add_xy(self, a, i, x, y, c):
        a[i*2] += x * c
        a[i*2+1] += y * c

    def repulsion_f(self, x, c_rep):
        t = max(1.0 - x, 0.0)
        return [0.5 * c_rep * t*t, -c_rep * t]

    def fast_exp(self, x):
        t = 1.0 + x/32.0
        t *= t; t *= t; t *= t; t *= t; t *= t
        return t

    def peak_f(self, x, mu, sigma, w=1.0):
        t = (x - mu)/sigma
        y = w/self.fast_exp(t*t)
        return [y, -2.0*t*y/sigma]

    # -----------------------------
    # Core simulation steps
    # -----------------------------
    def compute_fields(self):
        R_val, U_val, R_grad, U_grad = (
            self.fields["R_val"],
            self.fields["U_val"],
            self.fields["R_grad"],
            self.fields["U_grad"]
        )
        c_rep = self.params["c_rep"]
        mu_k = self.params["mu_k"]
        sigma_k = self.params["sigma_k"]
        w_k = self.params["w_k"]

        # Reset fields
        R_val.fill(self.repulsion_f(0.0, c_rep)[0])
        U_val.fill(self.peak_f(0.0, mu_k, sigma_k, w_k)[0])
        R_grad.fill(0)
        U_grad.fill(0)

        for i in range(self.point_n - 1):
            for j in range(i+1, self.point_n):
                rx = self.points[i*2] - self.points[j*2]
                ry = self.points[i*2+1] - self.points[j*2+1]
                r = math.sqrt(rx*rx + ry*ry) + 1e-20
                rx /= r
                ry /= r

                # Repulsion
                if r < 1.0:
                    R, dR = self.repulsion_f(r, c_rep)
                    self.add_xy(R_grad, i, rx, ry, dR)
                    self.add_xy(R_grad, j, rx, ry, -dR)
                    R_val[i] += R
                    R_val[j] += R

                # Attraction
                K, dK = self.peak_f(r, mu_k, sigma_k, w_k)
                self.add_xy(U_grad, i, rx, ry, dK)
                self.add_xy(U_grad, j, rx, ry, -dK)
                U_val[i] += K
                U_val[j] += K

    def step(self):
        R_val, U_val, R_grad, U_grad = (
            self.fields["R_val"],
            self.fields["U_val"],
            self.fields["R_grad"],
            self.fields["U_grad"]
        )
        mu_g = self.params["mu_g"]
        sigma_g = self.params["sigma_g"]
        dt = self.params["dt"]
        food_strength = self.food_params["food_attraction_strength"]
        food_radius = self.food_params["food_radius"]

        # Compute particle-particle interactions
        self.compute_fields()

        total_E = 0.0
        # Center of mass for food attraction
        center_x = np.mean(self.points[::2])
        center_y = np.mean(self.points[1::2])

        # Food attraction
        dx = self.food_pos[0] - center_x
        dy = self.food_pos[1] - center_y
        dist = math.sqrt(dx*dx + dy*dy)
        if dist > 1.0:
            dx /= dist
            dy /= dist
            for i in range(self.point_n):
                self.points[i*2] += dx * food_strength * dt
                self.points[i*2+1] += dy * food_strength * dt

        # Food repulsion if too close
        for i in range(self.point_n):
            px, py = self.points[i*2], self.points[i*2+1]
            d = math.sqrt((px - self.food_pos[0])**2 + (py - self.food_pos[1])**2)
            if d < food_radius:
                rep = food_strength * (food_radius - d)
                dx = px - self.food_pos[0]
                dy = py - self.food_pos[1]
                d = max(d, 1e-10)
                dx /= d
                dy /= d
                self.points[i*2] += dx * rep
                self.points[i*2+1] += dy * rep

        # Particle-particle movement
        for i in range(self.point_n):
            G, dG = self.peak_f(U_val[i], mu_g, sigma_g)
            vx = dG * U_grad[i*2] - R_grad[i*2]
            vy = dG * U_grad[i*2+1] - R_grad[i*2+1]
            self.add_xy(self.points, i, vx, vy, dt)
            total_E += R_val[i] - G

        return total_E / self.point_n

    # -----------------------------
    # Headless simulation for experiments
    # -----------------------------
    def run_headless(self, steps=500):
        total_energy = 0.0
        for _ in range(steps):
            energy = self.step()
            total_energy += energy
            # Respawn food if reached
            center_x = np.mean(self.points[::2])
            center_y = np.mean(self.points[1::2])
            if math.sqrt((self.food_pos[0]-center_x)**2 + (self.food_pos[1]-center_y)**2) < 1.0:
                self.food_pos = self.spawn_food()
        return total_energy / steps

    # -----------------------------
    # Utility functions
    # -----------------------------
    def get_state(self):
        return self.points.reshape(-1,2).copy()

    def spawn_food(self):
        while True:
            food_x = random.uniform(-self.world_width/2, self.world_width/2)
            food_y = random.uniform(-self.world_width/2, self.world_width/2)
            min_dist = self.food_params["food_spawn_min_dist"]
            overlap = any(math.sqrt((food_x - self.points[i*2])**2 + (food_y - self.points[i*2+1])**2) < min_dist
                          for i in range(self.point_n))
            if not overlap:
                return np.array([food_x, food_y])
