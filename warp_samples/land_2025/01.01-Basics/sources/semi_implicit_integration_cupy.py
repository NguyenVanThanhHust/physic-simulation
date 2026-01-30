
import cupy as cp

import nvtx

num_particles = 10_000_000  # Number of particles
num_steps = 100

mass = 0.1  # Mass per particle [kg]
g = 9.81  # Gravitational acceleration [m/s^2]
b = 0.05  # Drag Coefficient [kg/s]

dt = 0.01 * (2 * mass / b)

gravity = cp.array([0.0, 0.0, -g], dtype=cp.float32)

# Initial positions: random values between -1.0 and 1.0 for x, y, and z
positions = cp.random.uniform(low=-1.0, high=1.0, size=(num_particles, 3), dtype=cp.float32)

# Initial velocities: random values between -0.5 and 0.5 m/s for vx, vy, and vz
velocities = cp.random.uniform(low=-0.5, high=0.5, size=(num_particles, 3), dtype=cp.float32)

for step in range(num_steps):
    with nvtx.annotate("Step", color="rapids"):
        acceleration = (-b * velocities) / mass + gravity
        velocities += acceleration * dt
        positions += velocities * dt

# Make sure work is complete before terminating
cp.cuda.Stream.null.synchronize()
