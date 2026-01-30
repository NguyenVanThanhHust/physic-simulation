
import numpy as np

import warp as wp

import nvtx

num_particles = 10_000_000  # Number of particles
num_steps = 100

mass = 0.1  # Mass per particle [kg]
g = 9.81  # Gravitational acceleration [m/s^2]
b = 0.05  # Drag Coefficient [kg/s]

dt = 0.01 * (2 * mass / b)

gravity = wp.vec3([0.0, 0.0, -g])

# Initial positions: random values between -1.0 and 1.0 for x, y, and z
rng = np.random.default_rng(12345)
positions_np = rng.uniform(low=-1.0, high=1.0, size=(num_particles, 3))
positions = wp.array(positions_np, dtype=wp.vec3)

# Initial velocities: random values between -0.5 and 0.5 m/s for vx, vy, and vz
velocities_np = rng.uniform(low=-0.5, high=0.5, size=(num_particles, 3))
velocities = wp.array(velocities_np, dtype=wp.vec3)


@wp.kernel
def integrate(positions: wp.array(dtype=wp.vec3), velocities: wp.array(dtype=wp.vec3)):
    i = wp.tid()

    acceleration = (-b * velocities[i]) / mass + gravity
    velocities[i] += acceleration * dt
    positions[i] += velocities[i] * dt


for step in range(num_steps):
    with nvtx.annotate("Step", color="rapids"):
        wp.launch(integrate, dim=(num_particles,), inputs=[positions, velocities])

wp.synchronize()
