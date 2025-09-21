import pygame
import random
import math
import numpy as np

# ----------- ADJUSTABLE PARAMETERS -----------
# The following parameters can be changed to modify the simulation.
# This allows easy experimentation with different settings.

# Parameters for two particle types (group 1 and group 2)
params_type1 = {
    "mu_k": 4.0,  # Mean for the potential energy function (group 1)
    "sigma_k": 1.0,  # Standard deviation for the potential energy (group 1)
    "w_k": 0.022,  # Width of the potential energy peak (group 1)
    "mu_g": 0.6,  # Mean for the gradient function (group 1)
    "sigma_g": 0.15,  # Standard deviation for the gradient function (group 1)
    "c_rep": 1.0,  # Repulsion strength within the same group (group 1)
    "c_rep_inter": 2.0,  # Repulsion strength between groups (group 1)
    "dt": 0.1  # Time step for simulation
}

params_type2 = {
    "mu_k": 2.0,  # Mean for the potential energy function (group 2)
    "sigma_k": 0.5,  # Standard deviation for the potential energy (group 2)
    "w_k": 0.1,  # Width of the potential energy peak (group 2)
    "mu_g": 0.8,  # Mean for the gradient function (group 2)
    "sigma_g": 0.2,  # Standard deviation for the gradient function (group 2)
    "c_rep": 1.2,  # Repulsion strength within the same group (group 2)
    "c_rep_inter": 2.0,  # Repulsion strength between groups (group 2)
    "dt": 0.1  # Time step for simulation
}

point_n = 200  # Total number of particles (adjust this to simulate more/fewer particles)
steps_per_frame = 10  # Number of simulation steps per frame (can adjust for performance or detail)
world_width = 25.0  # Width of the world for scaling particles (larger value = more spread out particles)

# -----------------------------------------------
# INITIALIZATION AND SETUP
# -----------------------------------------------

# The init function initializes the positions of the particles,
# assigns types (group 1 and group 2), and assigns them to groups.
def init(points, types, groups):
    for i in range(point_n):
        # Initialize random positions for each particle in the 2D space
        points[i * 2] = (random.random() - 0.5) * 12  # Random x position
        points[i * 2 + 1] = (random.random() - 0.5) * 12  # Random y position
        # Assign particle types based on index (first half type 0, second half type 1)
        types[i] = 0 if i < point_n // 2 else 1
        # Assign particles to groups similarly (group 0 for first half, group 1 for second half)
        groups[i] = 0 if i < point_n // 2 else 1
    return points, types, groups

# Arrays to store particle data (positions, types, and groups)
points = np.zeros(point_n * 2, dtype=np.float32)  # Store 2D position (x, y) for each particle
types = np.zeros(point_n, dtype=int)  # Store particle types (0 or 1)
groups = np.zeros(point_n, dtype=int)  # Store particle groups (0 or 1)

# Initialize particles' data using the init function
points, types, groups = init(points, types, groups)

# -----------------------------------------------
# FIELD CALCULATION AND PARTICLE INTERACTIONS
# -----------------------------------------------

# The following dictionary stores the fields and their gradients for each particle.
fields = {
    "R_val": np.zeros(point_n, dtype=np.float32),  # Repulsion values for each particle
    "U_val": np.zeros(point_n, dtype=np.float32),  # Potential energy values for each particle
    "R_grad": np.zeros(point_n * 2, dtype=np.float32),  # Gradients of the repulsion field (2D)
    "U_grad": np.zeros(point_n * 2, dtype=np.float32)  # Gradients of the potential energy field (2D)
}

# Helper function to add x and y forces to the gradients
def add_xy(a, i, x, y, c):
    a[i * 2] += x * c  # Update x gradient for particle i
    a[i * 2 + 1] += y * c  # Update y gradient for particle i

# Function to compute repulsion and potential energy fields between particles
def compute_fields():
    # Extract the field values and gradients for ease of access
    R_val, U_val, R_grad, U_grad = fields["R_val"], fields["U_val"], fields["R_grad"], fields["U_grad"]
    c_rep, c_rep_inter, mu_k, sigma_k, w_k = params_type1["c_rep"], params_type1["c_rep_inter"], params_type1["mu_k"], params_type1["sigma_k"], params_type1["w_k"]

    # Initialize field values for the particles (at distance 0)
    R_val.fill(repulsion_f(0.0, c_rep)[0])  # Repulsion potential at distance 0
    U_val.fill(peak_f(0.0, mu_k, sigma_k, w_k)[0])  # Potential energy at distance 0
    R_grad.fill(0)  # Initialize repulsion gradients to zero
    U_grad.fill(0)  # Initialize potential energy gradients to zero

    # Loop through each pair of particles to compute interactions
    for i in range(point_n - 1):
        for j in range(i + 1, point_n):
            # Calculate the distance between particles i and j in 2D space
            rx = points[i * 2] - points[j * 2]
            ry = points[i * 2 + 1] - points[j * 2 + 1]
            r = math.sqrt(rx * rx + ry * ry) + 1e-20  # Calculate distance between particles, avoid division by 0
            rx /= r  # Normalize the x component of the gradient
            ry /= r  # Normalize the y component of the gradient

            # Intra-group interactions (same group)
            if groups[i] == groups[j]:
                if r < 1.0:  # Strong repulsion within the same group if too close
                    R, dR = repulsion_f(r, c_rep)  # Calculate repulsion and its gradient
                    add_xy(R_grad, i, rx, ry, dR)  # Add force to particle i
                    add_xy(R_grad, j, rx, ry, -dR)  # Add force to particle j
                    R_val[i] += R  # Add repulsion energy to particle i
                    R_val[j] += R  # Add repulsion energy to particle j

                # Calculate the potential energy and its gradient between particles i and j
                K, dK = peak_f(r, mu_k, sigma_k, w_k)
                add_xy(U_grad, i, rx, ry, dK)  # Update potential gradient for particle i
                add_xy(U_grad, j, rx, ry, -dK)  # Update potential gradient for particle j
                U_val[i] += K  # Add potential energy to particle i
                U_val[j] += K  # Add potential energy to particle j

            # Inter-group interactions (between different groups)
            else:
                if r < 1.5:  # Stronger repulsion between particles from different groups
                    R, dR = repulsion_f(r, c_rep_inter)  # Calculate repulsion and its gradient
                    add_xy(R_grad, i, rx, ry, dR)  # Add force to particle i
                    add_xy(R_grad, j, rx, ry, -dR)  # Add force to particle j
                    R_val[i] += R  # Add repulsion energy to particle i
                    R_val[j] += R  # Add repulsion energy to particle j

# -----------------------------------------------
# FORCE AND MOTION CALCULATION
# -----------------------------------------------

# Repulsion function: returns repulsion value and its gradient
def repulsion_f(x, c_rep):
    t = max(1.0 - x, 0.0)  # Repulsion decreases as particles move apart
    return [0.5 * c_rep * t * t, -c_rep * t]  # Repulsion value and gradient

# Fast exponential approximation for calculating potential energy
def fast_exp(x):
    t = 1.0 + x / 32.0  # Approximation of exponential function
    t *= t
    t *= t
    t *= t
    t *= t
    t *= t  # t **= 32
    return t

# Potential energy function based on a Gaussian peak
def peak_f(x, mu, sigma, w=1.0):
    t = (x - mu) / sigma  # Normalize distance
    y = w / fast_exp(t * t)  # Calculate the potential energy value
    return [y, -2.0 * t * y / sigma]  # Return potential energy and gradient

# Step function that updates particle positions based on forces
def step():
    # Extract field values and gradients for easier access
    R_val, U_val, R_grad, U_grad = fields["R_val"], fields["U_val"], fields["R_grad"], fields["U_grad"]
    mu_g, sigma_g, dt = params_type1["mu_g"], params_type1["sigma_g"], params_type1["dt"]

    # Compute fields and interactions between particles
    compute_fields()
    total_E = 0.0  # Initialize total energy

    # Update the position of each particle based on computed forces
    for i in range(point_n):
        # Calculate gradient force due to potential energy
        G, dG = peak_f(U_val[i], mu_g, sigma_g)
        # Calculate velocity (force) from gradients and apply repulsion force
        vx = dG * U_grad[i * 2] - R_grad[i * 2]
        vy = dG * U_grad[i * 2 + 1] - R_grad[i * 2 + 1]

        # Apply directional movement based on group
        if groups[i] == 0:  # Group 1 moves right
            vx += 0.1
            vy += 0.0  # No vertical movement for group 1
        else:  # Group 2 moves left
            vx += -0.1
            vy += 0.0  # No vertical movement for group 2

        # Update particle position based on velocity and time step
        add_xy(points, i, vx, vy, dt)
        total_E += R_val[i] - G  # Calculate total energy change

    # Return average energy per particle
    return total_E / point_n

# -----------------------------------------------
# PYGAME SETUP AND RENDERING
# -----------------------------------------------

# Initialize Pygame and set up the display window
pygame.init()
width, height = 800, 800  # Window size for visualization
screen = pygame.display.set_mode((width, height))  # Set display mode
pygame.display.set_caption("Two Interacting Cells Simulation")  # Window title
clock = pygame.time.Clock()  # Clock for controlling the frame rate

# Main simulation loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:  # Check for quit event
            running = False  # Stop the simulation

    # Run simulation for each frame
    for _ in range(steps_per_frame):
        step()  # Perform one simulation step

    # Render particles on the screen
    screen.fill((0, 0, 0))  # Fill the screen with black background
    s = width / world_width  # Scale factor for particle size and position
    for i in range(point_n):
        # Convert particle position to screen coordinates
        x = points[i * 2] * s + width / 2
        y = points[i * 2 + 1] * s + height / 2

        # Dynamic size based on repulsion field interaction
        if groups[i] == 0:  # Group 1 (Red particles)
            r = 5 + fields["R_val"][i] * 10  # Size influenced by repulsion
            color = (255, 0, 0)  # Red color for group 1
        else:  # Group 2 (Blue particles)
            r = 7 + fields["R_val"][i] * 8  # Size influenced by repulsion
            color = (0, 0, 255)  # Blue color for group 2

        # Draw the particle as a circle on the screen
        pygame.draw.circle(screen, color, (int(x), int(y)), max(1, int(r)))

    # Update the display
    pygame.display.flip()
    clock.tick(60)  # Limit the frame rate to 60 FPS

# Quit Pygame when done
pygame.quit()
