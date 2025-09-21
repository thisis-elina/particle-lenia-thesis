import pygame
import numpy as np
import math
import random

class ParticleLeniaSimulation:
    def __init__(self, config, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.params = config
        self.point_n = config.get("point_n", 200)
        self.points = np.zeros(self.point_n * 2, dtype=np.float32)
        self.init_points()

        # Internal per-particle buffers used during force aggregation:
        # - "R_val": scalar repulsion potential accumulated from close neighbors
        # - "U_val": scalar attraction (kernel) potential accumulated from all neighbors
        # - "R_grad": vector sum (x,y per particle) of repulsion directions scaled by dR/dr
        # - "U_grad": vector sum (x,y per particle) of attraction directions scaled by dK/dr
        # Keys are kept short for compatibility, but we use descriptive local names when reading them.
        self.fields = {
            "repulsion_potential": np.zeros(self.point_n, dtype=np.float32),
            "attraction_potential": np.zeros(self.point_n, dtype=np.float32),
            "repulsion_direction_sum": np.zeros(self.point_n * 2, dtype=np.float32),
            "attraction_direction_sum": np.zeros(self.point_n * 2, dtype=np.float32),
        }

    def init_points(self):
        for i in range(self.point_n):
            self.points[i*2] = (random.random() - 0.5) * 12
            self.points[i*2+1] = (random.random() - 0.5) * 12

    def add_xy(self, a, i, x, y, c):
        a[i*2] += x * c
        a[i*2+1] += y * c

    def repulsion_f(self, x, c_rep):
        t = max(1.0 - x, 0.0)
        return [0.5 * c_rep * t*t, -c_rep * t]

    def fast_exp(self, x):
        t = 1.0 + x/32.0
        for _ in range(5):
            t *= t
        return t

    def peak_f(self, x, mu, sigma, w=1.0):
        t = (x - mu)/sigma
        y = w / self.fast_exp(t*t)
        return [y, -2.0*t*y/sigma]

    def compute_fields(self):
        repulsion_potential, attraction_potential, repulsion_direction_sum, attraction_direction_sum = (
            self.fields["repulsion_potential"],
            self.fields["attraction_potential"],
            self.fields["repulsion_direction_sum"],
            self.fields["attraction_direction_sum"],
        )
        c_rep = self.params["c_rep"]
        mu_k = self.params["mu_k"]
        sigma_k = self.params["sigma_k"]
        w_k = self.params["w_k"]

        # Reset accumulators for this step
        repulsion_potential.fill(self.repulsion_f(0.0, c_rep)[0])
        attraction_potential.fill(self.peak_f(0.0, mu_k, sigma_k, w_k)[0])
        repulsion_direction_sum.fill(0)
        attraction_direction_sum.fill(0)

        # O(N^2) pairwise accumulation of potentials and direction sums
        for i in range(self.point_n-1):
            for j in range(i+1, self.point_n):
                rx = self.points[i*2] - self.points[j*2]
                ry = self.points[i*2+1] - self.points[j*2+1]
                r = math.sqrt(rx*rx + ry*ry) + 1e-20
                rx /= r
                ry /= r

                if r < 1.0:
                    R, dR = self.repulsion_f(r, c_rep)
                    self.add_xy(repulsion_direction_sum, i, rx, ry, dR)
                    self.add_xy(repulsion_direction_sum, j, rx, ry, -dR)
                    repulsion_potential[i] += R
                    repulsion_potential[j] += R

                K, dK = self.peak_f(r, mu_k, sigma_k, w_k)
                self.add_xy(attraction_direction_sum, i, rx, ry, dK)
                self.add_xy(attraction_direction_sum, j, rx, ry, -dK)
                attraction_potential[i] += K
                attraction_potential[j] += K

    def step(self):
        mu_g = self.params["mu_g"]
        sigma_g = self.params["sigma_g"]
        dt = self.params["dt"]

        self.compute_fields()
        repulsion_potential, attraction_potential, repulsion_direction_sum, attraction_direction_sum = (
            self.fields["repulsion_potential"],
            self.fields["attraction_potential"],
            self.fields["repulsion_direction_sum"],
            self.fields["attraction_direction_sum"],
        )

        total_E = 0.0
        for i in range(self.point_n):
            G, dG = self.peak_f(attraction_potential[i], mu_g, sigma_g)
            vx = dG * attraction_direction_sum[i*2] - repulsion_direction_sum[i*2]
            vy = dG * attraction_direction_sum[i*2+1] - repulsion_direction_sum[i*2+1]
            self.add_xy(self.points, i, vx, vy, dt)
            total_E += repulsion_potential[i] - G

        return total_E / self.point_n

    def get_state(self):
        return self.points.reshape(-1,2).copy()

    def run_headless(self, steps=500):
        total_energy = 0
        for _ in range(steps):
            total_energy += self.step()
        return total_energy / steps
