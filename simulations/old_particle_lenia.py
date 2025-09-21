import pygame
import numpy as np
import math
import random

# Parameters
params = {
    "mu_k": 4.0,       # Peak function parameter (controls 'attraction')
    "sigma_k": 1.0,     # Peak function parameter (controls 'spread')
    "w_k": 0.022,       # Peak function parameter (controls 'weight')
    "mu_g": 0.6,        # Peak function for movement (gravity/repulsion)
    "sigma_g": 0.15,    # Peak function spread for gravity
    "c_rep": 1.0,       # Repulsion strength
    "dt": 0.1,          # Time step (used to update positions)
}

point_n = 200            # Number of particles
steps_per_frame = 10     # Number of simulation steps per frame
world_width = 25.0       # World width (scaling factor for rendering)

# Initialize particle positions randomly
def init(points):
    for i in range(point_n):
        points[i * 2] = (random.random() - 0.5) * 12  # Random x-coordinate
        points[i * 2 + 1] = (random.random() - 0.5) * 12  # Random y-coordinate
    return points

points = init(np.zeros(point_n * 2, dtype=np.float32))  # Initialize particles

# Fields: Store calculations for each particle
fields = {
    "R_val": np.zeros(point_n, dtype=np.float32),   # Repulsion values
    "U_val": np.zeros(point_n, dtype=np.float32),   # Attraction values
    "R_grad": np.zeros(point_n * 2, dtype=np.float32),  # Repulsion gradients
    "U_grad": np.zeros(point_n * 2, dtype=np.float32),  # Attraction gradients
}

# Helper function to add values to gradient arrays (for position updates)
def add_xy(a, i, x, y, c):
    a[i * 2] += x * c
    a[i * 2 + 1] += y * c

# Compute fields: Calculate repulsion and attraction for all particles
def compute_fields():
    R_val, U_val, R_grad, U_grad = fields["R_val"], fields["U_val"], fields["R_grad"], fields["U_grad"]
    c_rep, mu_k, sigma_k, w_k = params["c_rep"], params["mu_k"], params["sigma_k"], params["w_k"]

    # Initialize fields with values at distance zero (default behavior)
    R_val.fill(repulsion_f(0.0, c_rep)[0])
    U_val.fill(peak_f(0.0, mu_k, sigma_k, w_k)[0])
    R_grad.fill(0)  # Reset gradients
    U_grad.fill(0)

    # Loop through all pairs of points to compute interactions
    for i in range(point_n - 1):
        for j in range(i + 1, point_n):
            # Calculate the vector difference between two particles
            rx = points[i * 2] - points[j * 2]
            ry = points[i * 2 + 1] - points[j * 2 + 1]
            r = math.sqrt(rx * rx + ry * ry) + 1e-20  # Avoid division by zero
            rx /= r  # Normalize direction vector
            ry /= r

            if r < 1.0:
                # Apply repulsion if particles are close
                R, dR = repulsion_f(r, c_rep)
                add_xy(R_grad, i, rx, ry, dR)
                add_xy(R_grad, j, rx, ry, -dR)
                R_val[i] += R
                R_val[j] += R

            # Apply attraction based on the peak function
            K, dK = peak_f(r, mu_k, sigma_k, w_k)
            add_xy(U_grad, i, rx, ry, dK)
            add_xy(U_grad, j, rx, ry, -dK)
            U_val[i] += K
            U_val[j] += K

# Repulsion function: Returns the force and its derivative
def repulsion_f(x, c_rep):
    t = max(1.0 - x, 0.0)  # Smooth transition for repulsion
    return [0.5 * c_rep * t * t, -c_rep * t]  # Repulsion force and its gradient

# Fast exponential approximation (used in peak function)
def fast_exp(x):
    t = 1.0 + x / 32.0
    t *= t  # Square to speed up exponentiation
    t *= t
    t *= t
    t *= t
    t *= t
    return t

# Peak function: Represents attraction or gravitational pull between particles
def peak_f(x, mu, sigma, w=1.0):
    t = (x - mu) / sigma
    y = w / fast_exp(t * t)  # Compute the peak value using approximation
    return [y, -2.0 * t * y / sigma]  # Return value and gradient

# Step function: Perform a single simulation step (position update)
def step():
    R_val, U_val, R_grad, U_grad = fields["R_val"], fields["U_val"], fields["R_grad"], fields["U_grad"]
    mu_g, sigma_g, dt = params["mu_g"], params["sigma_g"], params["dt"]

    # Compute all field values
    compute_fields()
    total_E = 0.0

    # Update positions based on gradients and forces
    for i in range(point_n):
        G, dG = peak_f(U_val[i], mu_g, sigma_g)  # Compute the gravity or repulsion effect
        # Update velocities based on gradients of potential energy
        vx = dG * U_grad[i * 2] - R_grad[i * 2]
        vy = dG * U_grad[i * 2 + 1] - R_grad[i * 2 + 1]
        add_xy(points, i, vx, vy, dt)  # Apply velocity update to position
        total_E += R_val[i] - G  # Energy calculation for debugging

    # Return the average energy for tracking purposes
    return total_E / point_n

# Pygame setup for rendering the simulation
pygame.init()
width, height = 800, 800
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Particle Simulation")
clock = pygame.time.Clock()

# Main loop to update the simulation and render particles
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False  # Quit when window is closed

    # Update simulation by taking multiple steps per frame
    for _ in range(steps_per_frame):
        step()

    # Render particles on the screen
    screen.fill((0, 0, 0))  # Clear screen with black
    s = width / world_width  # Scaling factor for the world size
    for i in range(point_n):
        # Calculate screen position for each particle
        x = points[i * 2] * s + width / 2
        y = points[i * 2 + 1] * s + height / 2
        r = params["c_rep"] / (fields["R_val"][i] * 5.0) * s  # Size of the particle
        pygame.draw.circle(screen, (255, 255, 255), (int(x), int(y)), max(1, int(r)))  # Draw particle

    pygame.display.flip()  # Update the screen
    clock.tick(60)  # Limit the frame rate to 60 FPS

pygame.quit()  # Close Pygame window when the loop ends
