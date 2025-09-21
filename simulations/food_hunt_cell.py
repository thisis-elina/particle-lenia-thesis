import pygame  # Library for rendering and visualization of the simulation
import numpy as np  # Used for efficient numerical computations (arrays, matrices)
import math  # Contains mathematical functions (like square root, trigonometry)
import random  # Used for generating random numbers (initializing positions, spawning food)

# Parameters for particles: Defines how particles interact and move
params = {
    "mu_k": 4.0,  # Mean value for the attraction force (control parameter)
    "sigma_k": 1.0,  # Standard deviation for attraction (affects spread)
    "w_k": 0.022,  # Weight for attraction strength (controls magnitude)
    "mu_g": 0.6,  # Mean value for the particle’s velocity behavior
    "sigma_g": 0.15,  # Standard deviation for the particle's velocity
    "c_rep": 1.0,  # Repulsion coefficient (how strongly particles repel each other)
    "dt": 0.1  # Time step (how much time passes per update in the simulation)
}

# Parameters related to the food item (how food behaves in the simulation)
food_params = {
    "food_attraction_strength": 0.1,  # How strongly particles are attracted to the food
    "food_radius": 2.0,  # Radius around the food where particles will feel repulsion
    "food_spawn_min_dist": 5.0  # Minimum distance between particles and food when spawning new food
}

# Number of particles to simulate
point_n = 100  # Total number of particles in the simulation
steps_per_frame = 10  # Number of simulation steps per frame (increases smoothness)
world_width = 25.0  # Width of the simulation world

# Function to initialize the particle positions randomly
def init(points):
    for i in range(point_n):  # Loop through each particle
        points[i * 2] = (random.random() - 0.5) * 12  # Random x position for the particle
        points[i * 2 + 1] = (random.random() - 0.5) * 12  # Random y position for the particle
    return points  # Return the array of particle positions

points = np.zeros(point_n * 2, dtype=np.float32)  # Initialize an array to hold particle positions (x, y)
points = init(points)  # Call the function to assign random positions to each particle

# Create fields that will store information related to repulsion and attraction forces
fields = {
    "R_val": np.zeros(point_n, dtype=np.float32),  # Repulsion values for each particle
    "U_val": np.zeros(point_n, dtype=np.float32),  # Attraction values for each particle
    "R_grad": np.zeros(point_n * 2, dtype=np.float32),  # Gradients of repulsion force
    "U_grad": np.zeros(point_n * 2, dtype=np.float32)  # Gradients of attraction force
}

# Helper function to update the x and y positions of particles based on a force
def add_xy(a, i, x, y, c):
    a[i * 2] += x * c  # Update x position of the particle
    a[i * 2 + 1] += y * c  # Update y position of the particle

# Function that computes the repulsion and attraction forces for each particle
def compute_fields(points):
    # Extract the arrays storing repulsion and attraction data
    R_val, U_val, R_grad, U_grad = fields["R_val"], fields["U_val"], fields["R_grad"], fields["U_grad"]
    c_rep, mu_k, sigma_k, w_k = params["c_rep"], params["mu_k"], params["sigma_k"], params["w_k"]

    # Initialize all values to 0 or default values
    R_val.fill(repulsion_f(0.0, c_rep)[0])  # Repulsion at distance 0
    U_val.fill(peak_f(0.0, mu_k, sigma_k, w_k)[0])  # Attraction at distance 0
    R_grad.fill(0)  # Initialize repulsion gradients to 0
    U_grad.fill(0)  # Initialize attraction gradients to 0

    # Loop over all pairs of particles to compute the forces (repulsion & attraction)
    for i in range(point_n - 1):
        for j in range(i + 1, point_n):  # Compare each particle with every other particle
            # Calculate the distance vector between two particles
            rx = points[i * 2] - points[j * 2]
            ry = points[i * 2 + 1] - points[j * 2 + 1]
            r = math.sqrt(rx * rx + ry * ry) + 1e-20  # Distance between particles (adding small value to avoid divide-by-zero)
            rx /= r  # Normalize the direction
            ry /= r  # Normalize the direction

            # If particles are too close, apply repulsion
            if r < 1.0:
                # Calculate repulsion force and its gradient
                R, dR = repulsion_f(r, c_rep)
                add_xy(R_grad, i, rx, ry, dR)  # Update repulsion gradient for particle i
                add_xy(R_grad, j, rx, ry, -dR)  # Update repulsion gradient for particle j
                R_val[i] += R  # Add the repulsion force to the value for particle i
                R_val[j] += R  # Add the repulsion force to the value for particle j

            # Calculate attraction force and its gradient
            K, dK = peak_f(r, mu_k, sigma_k, w_k)
            add_xy(U_grad, i, rx, ry, dK)  # Update attraction gradient for particle i
            add_xy(U_grad, j, rx, ry, -dK)  # Update attraction gradient for particle j
            U_val[i] += K  # Add attraction force to the value for particle i
            U_val[j] += K  # Add attraction force to the value for particle j

# Repulsion function: Defines how particles repel each other when too close
def repulsion_f(x, c_rep):
    t = max(1.0 - x, 0.0)  # Apply a threshold to avoid negative repulsion
    return [0.5 * c_rep * t * t, -c_rep * t]  # Return the repulsion force and its gradient

# Fast exponential function for approximating attraction (more efficient)
def fast_exp(x):
    t = 1.0 + x / 32.0
    t *= t  # Efficiently compute t ** 32
    t *= t
    t *= t
    t *= t
    t *= t
    return t

# Attraction function: Defines how the attraction force behaves with distance
def peak_f(x, mu, sigma, w=1.0):
    t = (x - mu) / sigma  # Standardize the distance
    y = w / fast_exp(t * t)  # Apply an approximation for attraction strength
    return [y, -2.0 * t * y / sigma]  # Return the attraction value and its gradient

# Step function: Updates the positions of particles based on forces
def step(points, food_pos):
    R_val, U_val, R_grad, U_grad = fields["R_val"], fields["U_val"], fields["R_grad"], fields["U_grad"]
    mu_g, sigma_g, dt = params["mu_g"], params["sigma_g"], params["dt"]
    food_attraction_strength = food_params["food_attraction_strength"]

    compute_fields(points)  # Compute all force fields (repulsion & attraction)
    total_E = 0.0  # Total energy of the system, initialized to 0

    # Calculate the center of mass of the particles (average position)
    center_x = np.mean(points[::2])  # Average x position
    center_y = np.mean(points[1::2])  # Average y position

    # Move particles towards the food (food attraction force)
    dx = food_pos[0] - center_x
    dy = food_pos[1] - center_y
    dist = math.sqrt(dx * dx + dy * dy)
    if dist > 1.0:  # Only apply attraction if particles are not too close
        dx /= dist  # Normalize direction
        dy /= dist
        for i in range(point_n):  # Move all particles towards the food
            points[i * 2] += dx * food_attraction_strength * dt
            points[i * 2 + 1] += dy * food_attraction_strength * dt

    # Apply repulsion if particles are too close to the food
    food_radius = food_params["food_radius"]
    for i in range(point_n):
        px, py = points[i * 2], points[i * 2 + 1]
        dist_to_food = math.sqrt((px - food_pos[0]) ** 2 + (py - food_pos[1]) ** 2)

        if dist_to_food < food_radius:  # Apply repulsion if inside the radius
            repulsion_strength = food_params["food_attraction_strength"] * (food_radius - dist_to_food)
            dx = px - food_pos[0]
            dy = py - food_pos[1]
            dist_to_food = max(dist_to_food, 1e-10)  # Prevent division by zero
            dx /= dist_to_food
            dy /= dist_to_food
            points[i * 2] += dx * repulsion_strength
            points[i * 2 + 1] += dy * repulsion_strength

    # Update particle positions based on forces
    for i in range(point_n):
        G, dG = peak_f(U_val[i], mu_g, sigma_g)  # Get attraction value and gradient for each particle
        vx = dG * U_grad[i * 2] - R_grad[i * 2]  # Compute velocity in x direction
        vy = dG * U_grad[i * 2 + 1] - R_grad[i * 2 + 1]  # Compute velocity in y direction
        add_xy(points, i, vx, vy, dt)  # Update particle positions using velocity and time step
        total_E += R_val[i] - G  # Update total energy (repulsion - attraction)

    return total_E / point_n, center_x, center_y  # Return average energy and the center of particles

# Function to spawn food at a random position, ensuring it doesn't overlap with particles
def spawn_food(points, min_dist=food_params["food_spawn_min_dist"]):
    while True:
        food_x = random.uniform(-world_width / 2, world_width / 2)  # Random x position for food
        food_y = random.uniform(-world_width / 2, world_width / 2)  # Random y position for food

        overlap = False  # Flag to check if food overlaps with particles
        for i in range(point_n):  # Check if food is too close to any particle
            px = points[i * 2]
            py = points[i * 2 + 1]
            dist = math.sqrt((food_x - px) ** 2 + (food_y - py) ** 2)
            if dist < min_dist:  # If overlap, regenerate position
                overlap = True
                break

        if not overlap:
            return food_x, food_y  # Return valid food position if no overlap

# Initialize pygame for visualization
pygame.init()
width, height = 800, 800  # Size of the window
screen = pygame.display.set_mode((width, height))  # Create the window for simulation
pygame.display.set_caption("Cell and Food Simulation")  # Title of the window
clock = pygame.time.Clock()  # Control the frame rate of the simulation

food_pos = spawn_food(points)  # Spawn initial food

# Main simulation loop
running = True
while running:
    for event in pygame.event.get():  # Check for events like closing the window
        if event.type == pygame.QUIT:
            running = False  # End the loop if window is closed

    # Update simulation for each frame (run the simulation steps multiple times per frame)
    for _ in range(steps_per_frame):
        total_E, center_x, center_y = step(points, food_pos)  # Update positions of particles

        # Check if the food has been eaten (i.e., particles are too close to food)
        dx = food_pos[0] - center_x
        dy = food_pos[1] - center_y
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 1.0:  # If food is eaten, spawn new food
            food_pos = spawn_food(points)

    # Rendering: Draw particles and food on the screen
    screen.fill((0, 0, 0))  # Clear screen with black background
    s = width / world_width  # Scaling factor to fit the world to the window
    for i in range(point_n):  # Loop through all particles
        x = points[i * 2] * s + width / 2  # Scale x position of particle to fit screen
        y = points[i * 2 + 1] * s + height / 2  # Scale y position of particle to fit screen
        pygame.draw.circle(screen, (255, 0, 0), (int(x), int(y)), 5)  # Draw particle as red circle

    # Render food as a green circle
    food_x = food_pos[0] * s + width / 2  # Scale food's x position
    food_y = food_pos[1] * s + height / 2  # Scale food's y position
    pygame.draw.circle(screen, (0, 255, 0), (int(food_x), int(food_y)), 10)  # Draw food

    pygame.display.flip()  # Update the screen
    clock.tick(60)  # Control the frame rate (60 FPS)

pygame.quit()  # Close pygame when simulation ends
