import pygame
import numpy as np
import math
import random

# Parameters for particles
params = {
    "mu_k": 4.0, "sigma_k": 1.0, "w_k": 0.022,
    "mu_g": 0.6, "sigma_g": 0.15, "c_rep": 1.0,
    "dt": 0.1  # Time step: how much time is passed each step in the simulation
}

# Food Parameters (easily adjustable)
food_params = {
    "food_attraction_strength": 0.1,  # Controls how fast particles are attracted to the food
    "food_radius": 2.0,  # Radius around the food where repulsion will occur
    "food_spawn_min_dist": 5.0  # Minimum distance between food and particles when spawning food
}

point_n = 100  # Total number of particles
steps_per_frame = 10  # Number of simulation steps per frame (higher = smoother)
world_width = 25.0  # The width of the simulation space (the world boundaries)


# Function to initialize the particles at random positions
def init(points):
    for i in range(point_n):
        points[i * 2] = (random.random() - 0.5) * 12  # Random x positions within a certain range
        points[i * 2 + 1] = (random.random() - 0.5) * 12  # Random y positions within a certain range
    return points


points = np.zeros(point_n * 2, dtype=np.float32)  # Initialize all particle positions as (0, 0)
points = init(points)  # Call the init function to assign random positions to particles

# Fields that store information for repulsion, attraction, and gradients for each particle
fields = {
    "R_val": np.zeros(point_n, dtype=np.float32),  # Repulsion values for each particle
    "U_val": np.zeros(point_n, dtype=np.float32),  # Attraction values for each particle
    "R_grad": np.zeros(point_n * 2, dtype=np.float32),  # Gradients of the repulsion force
    "U_grad": np.zeros(point_n * 2, dtype=np.float32)  # Gradients of the attraction force
}


# Helper function to add values to the x and y components of a particle's position
def add_xy(a, i, x, y, c):
    a[i * 2] += x * c
    a[i * 2 + 1] += y * c


# Function to compute the fields for repulsion and attraction forces between particles
def compute_fields(points):
    # Extract the field arrays
    R_val, U_val, R_grad, U_grad = fields["R_val"], fields["U_val"], fields["R_grad"], fields["U_grad"]
    c_rep, mu_k, sigma_k, w_k = params["c_rep"], params["mu_k"], params["sigma_k"], params["w_k"]

    # Initialize the fields for each particle
    R_val.fill(repulsion_f(0.0, c_rep)[0])  # Initialize repulsion values
    U_val.fill(peak_f(0.0, mu_k, sigma_k, w_k)[0])  # Initialize attraction values
    R_grad.fill(0)  # Clear the repulsion gradients
    U_grad.fill(0)  # Clear the attraction gradients

    # Loop over each pair of particles and compute interactions (repulsion and attraction)
    for i in range(point_n - 1):
        for j in range(i + 1, point_n):
            # Compute the distance vector between two particles
            rx = points[i * 2] - points[j * 2]
            ry = points[i * 2 + 1] - points[j * 2 + 1]
            r = math.sqrt(rx * rx + ry * ry) + 1e-20  # Distance between particles (with small offset)
            rx /= r  # Normalize the direction
            ry /= r  # Normalize the direction

            if r < 1.0:  # Repulsion force only applies if the particles are close
                # Repulsion force function and its gradient
                R, dR = repulsion_f(r, c_rep)
                add_xy(R_grad, i, rx, ry, dR)  # Add gradient for particle i
                add_xy(R_grad, j, rx, ry, -dR)  # Add gradient for particle j
                R_val[i] += R  # Update repulsion value for particle i
                R_val[j] += R  # Update repulsion value for particle j

            # Attraction force function and its gradient
            K, dK = peak_f(r, mu_k, sigma_k, w_k)
            add_xy(U_grad, i, rx, ry, dK)  # Add gradient for particle i
            add_xy(U_grad, j, rx, ry, -dK)  # Add gradient for particle j
            U_val[i] += K  # Update attraction value for particle i
            U_val[j] += K  # Update attraction value for particle j


# Repulsion function: Defines how the particles repel each other when too close
def repulsion_f(x, c_rep):
    t = max(1.0 - x, 0.0)  # Apply a threshold to the distance to prevent negative values
    return [0.5 * c_rep * t * t, -c_rep * t]  # Return the repulsion force and its gradient


# Fast exponential approximation function (used to calculate attraction function)
def fast_exp(x):
    t = 1.0 + x / 32.0
    t *= t
    t *= t
    t *= t
    t *= t
    t *= t  # t **= 32 for fast exponential approximation
    return t


# Peak function: Defines how the attraction force behaves with distance
def peak_f(x, mu, sigma, w=1.0):
    t = (x - mu) / sigma  # Standardize the distance
    y = w / fast_exp(t * t)  # Apply a fast exponential approximation to calculate the peak value
    return [y, -2.0 * t * y / sigma]  # Return the attraction value and its gradient


# Step function: Updates the positions of particles based on repulsion and attraction forces
def step(points, food_pos):
    R_val, U_val, R_grad, U_grad = fields["R_val"], fields["U_val"], fields["R_grad"], fields["U_grad"]
    mu_g, sigma_g, dt = params["mu_g"], params["sigma_g"], params["dt"]
    food_attraction_strength = food_params["food_attraction_strength"]  # Attraction strength towards food

    compute_fields(points)  # Update fields (repulsion and attraction) for all particles
    total_E = 0.0  # Initialize total energy of the system

    # Calculate center of the cell (the average position of all particles)
    center_x = np.mean(points[::2])
    center_y = np.mean(points[1::2])

    # Add strong attraction force toward the food
    dx = food_pos[0] - center_x
    dy = food_pos[1] - center_y
    dist = math.sqrt(dx * dx + dy * dy)
    if dist > 1.0:  # Only apply attraction if not too close to food
        dx /= dist
        dy /= dist
        for i in range(point_n):
            points[i * 2] += dx * food_attraction_strength * dt  # Move particles towards food
            points[i * 2 + 1] += dy * food_attraction_strength * dt  # Move particles towards food

    # Repulsive force between particles and food (if too close)
    food_radius = food_params["food_radius"]  # Repulsion radius
    for i in range(point_n):
        # Calculate distance to the food
        px, py = points[i * 2], points[i * 2 + 1]
        dist_to_food = math.sqrt((px - food_pos[0]) ** 2 + (py - food_pos[1]) ** 2)

        # If the particle is within the repulsion radius, apply a repulsive force
        if dist_to_food < food_radius:
            repulsion_strength = food_params["food_attraction_strength"] * (food_radius - dist_to_food)
            dx = px - food_pos[0]
            dy = py - food_pos[1]
            dist_to_food = max(dist_to_food, 1e-10)  # Prevent division by zero
            dx /= dist_to_food  # Normalize direction
            dy /= dist_to_food  # Normalize direction
            points[i * 2] += dx * repulsion_strength  # Apply repulsion force in x direction
            points[i * 2 + 1] += dy * repulsion_strength  # Apply repulsion force in y direction

    # Update particle positions based on the fields
    for i in range(point_n):
        G, dG = peak_f(U_val[i], mu_g, sigma_g)  # Get attraction value and its gradient
        # Calculate velocity (change in position)
        vx = dG * U_grad[i * 2] - R_grad[i * 2]
        vy = dG * U_grad[i * 2 + 1] - R_grad[i * 2 + 1]
        add_xy(points, i, vx, vy, dt)  # Update the particle's position
        total_E += R_val[i] - G  # Update the total energy of the system

    return total_E / point_n, center_x, center_y  # Return average energy and the center of the cell


# Function to spawn food at a random position, ensuring it doesn't overlap with any particles
def spawn_food(points, min_dist=food_params["food_spawn_min_dist"]):
    while True:
        # Randomly generate food position
        food_x = random.uniform(-world_width / 2, world_width / 2)
        food_y = random.uniform(-world_width / 2, world_width / 2)

        # Check if food is too close to any particle
        overlap = False
        for i in range(point_n):
            px = points[i * 2]
            py = points[i * 2 + 1]
            dist = math.sqrt((food_x - px) ** 2 + (food_y - py) ** 2)
            if dist < min_dist:  # If food is too close to any particle, regenerate position
                overlap = True
                break

        if not overlap:
            return food_x, food_y  # Return food position if it's far enough from particles


# Pygame setup for visualization
pygame.init()
width, height = 800, 800  # Screen size
screen = pygame.display.set_mode((width, height))  # Create the screen for rendering
pygame.display.set_caption("Cell and Food Simulation")  # Set the window title
clock = pygame.time.Clock()  # Create a clock object to control frame rate

food_pos = spawn_food(points)  # Initial food position

# Main loop
running = True
while running:
    for event in pygame.event.get():  # Handle user input (e.g., closing the window)
        if event.type == pygame.QUIT:
            running = False

    # Update simulation for a set number of steps per frame
    for _ in range(steps_per_frame):
        total_E, center_x, center_y = step(points, food_pos)  # Update particle positions

        # Check if food has been eaten (if particles have reached the food)
        dx = food_pos[0] - center_x
        dy = food_pos[1] - center_y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 1.0:  # If food is eaten, spawn new food
            food_pos = spawn_food(points)

    # Render particles (draw them as red circles)
    screen.fill((0, 0, 0))  # Clear the screen (background color black)
    s = width / world_width  # Scaling factor for particle positions to fit screen
    for i in range(point_n):
        x = points[i * 2] * s + width / 2  # Scale x position to fit screen
        y = points[i * 2 + 1] * s + height / 2  # Scale y position to fit screen
        pygame.draw.circle(screen, (255, 0, 0), (int(x), int(y)), 5)  # Draw the particle as a red circle

    # Render food (draw it as a green circle)
    food_x = food_pos[0] * s + width / 2  # Scale food's x position to fit screen
    food_y = food_pos[1] * s + height / 2  # Scale food's y position to fit screen
    pygame.draw.circle(screen, (0, 255, 0), (int(food_x), int(food_y)), 10)  # Draw the food as a green circle

    pygame.display.flip()  # Update the display
    clock.tick(60)  # Control the frame rate (60 frames per second)

pygame.quit()  # Quit pygame when the loop ends
