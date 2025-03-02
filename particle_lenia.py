import pygame
import numpy as np
import math
import random

# Parameters
params = {
    "mu_k": 4.0, "sigma_k": 1.0, "w_k": 0.022,
    "mu_g": 0.6, "sigma_g": 0.15, "c_rep": 1.0,
    "dt": 0.1
}

point_n = 200
steps_per_frame = 10
world_width = 25.0

# Initialize points
def init(points):
    for i in range(point_n):
        points[i * 2] = (random.random() - 0.5) * 12
        points[i * 2 + 1] = (random.random() - 0.5) * 12
    return points

points = init(np.zeros(point_n * 2, dtype=np.float32))

# Fields
fields = {
    "R_val": np.zeros(point_n, dtype=np.float32),
    "U_val": np.zeros(point_n, dtype=np.float32),
    "R_grad": np.zeros(point_n * 2, dtype=np.float32),
    "U_grad": np.zeros(point_n * 2, dtype=np.float32)
}

# Helper function to add to arrays
def add_xy(a, i, x, y, c):
    a[i * 2] += x * c
    a[i * 2 + 1] += y * c

# Compute fields
def compute_fields():
    R_val, U_val, R_grad, U_grad = fields["R_val"], fields["U_val"], fields["R_grad"], fields["U_grad"]
    c_rep, mu_k, sigma_k, w_k = params["c_rep"], params["mu_k"], params["sigma_k"], params["w_k"]

    # Initialize fields
    R_val.fill(repulsion_f(0.0, c_rep)[0])
    U_val.fill(peak_f(0.0, mu_k, sigma_k, w_k)[0])
    R_grad.fill(0)
    U_grad.fill(0)

    # Compute interactions
    for i in range(point_n - 1):
        for j in range(i + 1, point_n):
            rx = points[i * 2] - points[j * 2]
            ry = points[i * 2 + 1] - points[j * 2 + 1]
            r = math.sqrt(rx * rx + ry * ry) + 1e-20
            rx /= r
            ry /= r  # ∇r = [rx, ry]

            if r < 1.0:
                # ∇R = R'(r) ∇r
                R, dR = repulsion_f(r, c_rep)
                add_xy(R_grad, i, rx, ry, dR)
                add_xy(R_grad, j, rx, ry, -dR)
                R_val[i] += R
                R_val[j] += R

            # ∇K = K'(r) ∇r
            K, dK = peak_f(r, mu_k, sigma_k, w_k)
            add_xy(U_grad, i, rx, ry, dK)
            add_xy(U_grad, j, rx, ry, -dK)
            U_val[i] += K
            U_val[j] += K

# Repulsion function
def repulsion_f(x, c_rep):
    t = max(1.0 - x, 0.0)
    return [0.5 * c_rep * t * t, -c_rep * t]

# Fast exponential approximation
def fast_exp(x):
    t = 1.0 + x / 32.0
    t *= t
    t *= t
    t *= t
    t *= t
    t *= t  # t **= 32
    return t

# Peak function
def peak_f(x, mu, sigma, w=1.0):
    t = (x - mu) / sigma
    y = w / fast_exp(t * t)
    return [y, -2.0 * t * y / sigma]

# Step function
def step():
    R_val, U_val, R_grad, U_grad = fields["R_val"], fields["U_val"], fields["R_grad"], fields["U_grad"]
    mu_g, sigma_g, dt = params["mu_g"], params["sigma_g"], params["dt"]

    compute_fields()
    total_E = 0.0

    for i in range(point_n):
        G, dG = peak_f(U_val[i], mu_g, sigma_g)
        # [vx, vy] = -∇E = G'(U)∇U - ∇R
        vx = dG * U_grad[i * 2] - R_grad[i * 2]
        vy = dG * U_grad[i * 2 + 1] - R_grad[i * 2 + 1]
        add_xy(points, i, vx, vy, dt)
        total_E += R_val[i] - G

    return total_E / point_n

# Pygame setup
pygame.init()
width, height = 800, 800
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Particle Simulation")
clock = pygame.time.Clock()

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update simulation
    for _ in range(steps_per_frame):
        step()

    # Render particles
    screen.fill((0, 0, 0))
    s = width / world_width
    for i in range(point_n):
        x = points[i * 2] * s + width / 2
        y = points[i * 2 + 1] * s + height / 2
        r = params["c_rep"] / (fields["R_val"][i] * 5.0) * s
        pygame.draw.circle(screen, (255, 255, 255), (int(x), int(y)), max(1, int(r)))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()