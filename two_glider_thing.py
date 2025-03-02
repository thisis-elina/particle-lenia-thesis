import pygame
import numpy as np
import math
import random

# Parameters for two particle types
params_type1 = {
    "mu_k": 4.0, "sigma_k": 1.0, "w_k": 0.022,
    "mu_g": 0.6, "sigma_g": 0.15, "c_rep": 1.0,
    "c_rep_inter": 2.0,  # Repulsion strength between groups
    "dt": 0.1
}

params_type2 = {
    "mu_k": 2.0, "sigma_k": 0.5, "w_k": 0.1,
    "mu_g": 0.8, "sigma_g": 0.2, "c_rep": 1.2,
    "c_rep_inter": 2.0,  # Repulsion strength between groups
    "dt": 0.1
}

point_n = 200  # Total number of particles
steps_per_frame = 10
world_width = 25.0

# Initialize points, types, and groups
def init(points, types, groups):
    for i in range(point_n):
        points[i * 2] = (random.random() - 0.5) * 12
        points[i * 2 + 1] = (random.random() - 0.5) * 12
        types[i] = 0 if i < point_n // 2 else 1  # First half type 0, second half type 1
        groups[i] = 0 if i < point_n // 2 else 1  # First half group 0, second half group 1
    return points, types, groups

points = np.zeros(point_n * 2, dtype=np.float32)
types = np.zeros(point_n, dtype=int)
groups = np.zeros(point_n, dtype=int)  # Group identifier for each particle
points, types, groups = init(points, types, groups)

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
    c_rep, c_rep_inter, mu_k, sigma_k, w_k = params_type1["c_rep"], params_type1["c_rep_inter"], params_type1["mu_k"], params_type1["sigma_k"], params_type1["w_k"]

    # Initialize fields
    R_val.fill(repulsion_f(0.0, c_rep)[0])
    U_val.fill(peak_f(0.0, mu_k, sigma_k, w_k)[0])
    R_grad.fill(0)
    U_grad.fill(0)

    # Compute interactions
    for i in range(point_n - 1):
        for j in range(i + 1, point_n):
            # Get parameters for the pair of particles
            params_i = params_type1 if types[i] == 0 else params_type2
            params_j = params_type1 if types[j] == 0 else params_type2

            rx = points[i * 2] - points[j * 2]
            ry = points[i * 2 + 1] - points[j * 2 + 1]
            r = math.sqrt(rx * rx + ry * ry) + 1e-20
            rx /= r
            ry /= r  # ∇r = [rx, ry]

            # Intra-group interaction (same group)
            if groups[i] == groups[j]:
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

            # Inter-group interaction (different groups)
            else:
                if r < 1.5:  # Stronger repulsion between groups
                    # ∇R = R'(r) ∇r
                    R, dR = repulsion_f(r, c_rep_inter)
                    add_xy(R_grad, i, rx, ry, dR)
                    add_xy(R_grad, j, rx, ry, -dR)
                    R_val[i] += R
                    R_val[j] += R

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
    mu_g, sigma_g, dt = params_type1["mu_g"], params_type1["sigma_g"], params_type1["dt"]

    compute_fields()
    total_E = 0.0

    for i in range(point_n):
        G, dG = peak_f(U_val[i], mu_g, sigma_g)
        # [vx, vy] = -∇E = G'(U)∇U - ∇R
        vx = dG * U_grad[i * 2] - R_grad[i * 2]
        vy = dG * U_grad[i * 2 + 1] - R_grad[i * 2 + 1]

        # Add directional force based on group
        if groups[i] == 0:  # Group 1
            vx += 0.1  # Move right
            vy += 0.0  # No vertical movement
        else:  # Group 2
            vx += -0.1  # Move left
            vy += 0.0  # No vertical movement

        add_xy(points, i, vx, vy, dt)
        total_E += R_val[i] - G

    return total_E / point_n

# Pygame setup
pygame.init()
width, height = 800, 800
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Two Interacting Cells Simulation")
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

        # Dynamic size based on field interaction
        if groups[i] == 0:  # Group 1
            r = 5 + fields["R_val"][i] * 10  # Size based on repulsion
            color = (255, 0, 0)  # Red for group 1
        else:  # Group 2
            r = 7 + fields["R_val"][i] * 8  # Size based on repulsion
            color = (0, 0, 255)  # Blue for group 2

        pygame.draw.circle(screen, color, (int(x), int(y)), max(1, int(r)))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()