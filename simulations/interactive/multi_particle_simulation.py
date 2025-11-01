import pygame
import numpy as np
import math
import random

# Parameters for the two different types of particles
params_type1 = {
    "mu_k": 4.0, "sigma_k": 1.0, "w_k": 0.022,  # Parameters for the first particle type
    "mu_g": 0.6, "sigma_g": 0.15, "c_rep": 1.0,  # Interaction parameters for type 1
    "dt": 0.1  # Time step for simulation
}

params_type2 = {
    "mu_k": 2.0, "sigma_k": 0.5, "w_k": 0.1,    # Parameters for the second particle type
    "mu_g": 0.8, "sigma_g": 0.2, "c_rep": 1.2,  # Interaction parameters for type 2
    "dt": 0.1  # Time step for simulation
}

point_n = 200  # Total number of particles
steps_per_frame = 10  # How many steps to simulate per frame for smoothness
world_width = 25.0  # The size of the world to scale to the screen size

# Function to initialize particle positions and types
def init(points, types):
    for i in range(point_n):
        # Randomly initialize positions within a certain range
        points[i * 2] = (random.random() - 0.5) * 12
        points[i * 2 + 1] = (random.random() - 0.5) * 12
        # Assign particle types: First half type 0, second half type 1
        types[i] = 0 if i < point_n // 2 else 1
    return points, types

points = np.zeros(point_n * 2, dtype=np.float32)  # Store positions for all particles
types = np.zeros(point_n, dtype=int)  # Store types for all particles (0 or 1)
points, types = init(points, types)  # Initialize the particles

# Arrays to store calculated field values
fields = {
    "R_val": np.zeros(point_n, dtype=np.float32),  # Repulsion values
    "U_val": np.zeros(point_n, dtype=np.float32),  # Potential energy values
    "R_grad": np.zeros(point_n * 2, dtype=np.float32),  # Repulsion gradients (force)
    "U_grad": np.zeros(point_n * 2, dtype=np.float32)   # Gradient of potential energy (force)
}

# Helper function to add to the x and y components of the forces
def add_xy(a, i, x, y, c):
    a[i * 2] += x * c
    a[i * 2 + 1] += y * c

# Compute the fields for all particles based on their interactions
def compute_fields():
    # Unpacking field values for easy reference
    R_val, U_val, R_grad, U_grad = fields["R_val"], fields["U_val"], fields["R_grad"], fields["U_grad"]
    c_rep, mu_k, sigma_k, w_k = params_type1["c_rep"], params_type1["mu_k"], params_type1["sigma_k"], params_type1["w_k"]

    # Initialize field values
    R_val.fill(repulsion_f(0.0, c_rep)[0])  # Start with the repulsion value
    U_val.fill(peak_f(0.0, mu_k, sigma_k, w_k)[0])  # Start with potential energy value
    R_grad.fill(0)  # Initialize repulsion gradient to zero
    U_grad.fill(0)  # Initialize potential energy gradient to zero

    # Compute pairwise interactions for all particles
    for i in range(point_n - 1):
        for j in range(i + 1, point_n):
            # Get parameters for the pair of particles (based on their type)
            params_i = params_type1 if types[i] == 0 else params_type2
            params_j = params_type1 if types[j] == 0 else params_type2

            # Calculate the distance between the two particles
            rx = points[i * 2] - points[j * 2]
            ry = points[i * 2 + 1] - points[j * 2 + 1]
            r = math.sqrt(rx * rx + ry * ry) + 1e-20  # Add a small value to avoid division by zero
            rx /= r
            ry /= r  # Normalize the direction vector

            # If the particles are close enough, apply the repulsion force
            if r < 1.0:
                R, dR = repulsion_f(r, c_rep)  # Get repulsion force and its derivative
                add_xy(R_grad, i, rx, ry, dR)  # Apply repulsion force gradient to particle i
                add_xy(R_grad, j, rx, ry, -dR)  # Apply negative repulsion force gradient to particle j
                R_val[i] += R  # Accumulate the repulsion force value for particle i
                R_val[j] += R  # Accumulate the repulsion force value for particle j

            # Apply the potential energy gradient (Gaussian potential)
            K, dK = peak_f(r, mu_k, sigma_k, w_k)  # Get potential energy and its gradient
            add_xy(U_grad, i, rx, ry, dK)  # Apply gradient to particle i
            add_xy(U_grad, j, rx, ry, -dK)  # Apply negative gradient to particle j
            U_val[i] += K  # Accumulate potential energy value for particle i
            U_val[j] += K  # Accumulate potential energy value for particle j

# Repulsion function between particles
def repulsion_f(x, c_rep):
    t = max(1.0 - x, 0.0)  # Repulsion decreases as particles get further apart
    return [0.5 * c_rep * t * t, -c_rep * t]  # Return the repulsion force and its derivative

# Fast exponential approximation to avoid slow math.exp calls
def fast_exp(x):
    t = 1.0 + x / 32.0
    t *= t
    t *= t
    t *= t
    t *= t
    t *= t  # Apply exponentiation by squaring (this is a fast approximation of exp(x))
    return t

# Peak function representing the potential energy as a Gaussian
def peak_f(x, mu, sigma, w=1.0):
    t = (x - mu) / sigma  # Standardize the distance
    y = w / fast_exp(t * t)  # Apply the Gaussian formula (with fast exp approximation)
    return [y, -2.0 * t * y / sigma]  # Return the potential value and its derivative

# Simulation step function: Updates particle positions based on forces
def step():
    R_val, U_val, R_grad, U_grad = fields["R_val"], fields["U_val"], fields["R_grad"], fields["U_grad"]
    mu_g, sigma_g, dt = params_type1["mu_g"], params_type1["sigma_g"], params_type1["dt"]

    compute_fields()  # Update the fields (forces and potential energy)
    total_E = 0.0  # Initialize total energy

    # Update positions of particles based on the forces
    for i in range(point_n):
        G, dG = peak_f(U_val[i], mu_g, sigma_g)  # Get the potential and gradient for the current particle
        vx = dG * U_grad[i * 2] - R_grad[i * 2]  # Calculate x component of velocity
        vy = dG * U_grad[i * 2 + 1] - R_grad[i * 2 + 1]  # Calculate y component of velocity
        add_xy(points, i, vx, vy, dt)  # Update the position based on velocity and time step
        total_E += R_val[i] - G  # Update the total energy

    return total_E / point_n  # Return the average energy per particle

# Pygame setup for visualization
pygame.init()
width, height = 800, 800  # Set the screen size
screen = pygame.display.set_mode((width, height))  # Create the screen for rendering
pygame.display.set_caption("Multi-Particle-Type Simulation")  # Set the window title
clock = pygame.time.Clock()  # Set the clock for controlling the frame rate

# Main loop to run the simulation
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:  # If the user closes the window, stop the simulation
            running = False

    # Update the simulation for a few steps to ensure smooth movement
    for _ in range(steps_per_frame):
        step()

    # Render particles to the screen
    screen.fill((0, 0, 0))  # Clear the screen with a black background
    s = width / world_width  # Scale the particles to fit the screen
    for i in range(point_n):
        x = points[i * 2] * s + width / 2  # Convert the particle x position to screen coordinates
        y = points[i * 2 + 1] * s + height / 2  # Convert the particle y position to screen coordinates

        # Dynamic size and color for particles based on their type
        if types[i] == 0:
            r = 5 + fields["R_val"][i] * 10  # Size based on repulsion for type 0 particles
            color = (255, 0, 0)  # Red for type 0 particles
        else:
            r = 7 + fields["R_val"][i] * 8  # Size based on repulsion for type 1 particles
            color = (0, 0, 255)  # Blue for type 1 particles

        # Draw the particle as a circle with dynamic size and color
        pygame.draw.circle(screen, color, (int(x), int(y)), max(1, int(r)))

    pygame.display.flip()  # Update the screen with the new frame
    clock.tick(60)  # Control the frame rate (60 FPS)

pygame.quit()  # Quit the Pygame library when the simulation ends
