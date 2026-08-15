import numpy as np
import os

G     = 6.67430e-11
M_sun = 1.989e30
AU    = 1.496e11

year = 365.25 * 24 * 3600

M_star = M_sun

eta = 0.002

def Omega_K(r):

    return np.sqrt(G * M_star / r**3)


def v_K(r):

    return r * Omega_K(r)


def v_radial(r, St):

    vk = v_K(r)

    return -2.0 * eta * vk * St / (1.0 + St**2)


def v_azimuthal(r, St):

    vk = v_K(r)

    return vk * (1.0 - eta/(1.0 + St**2))


def simulate_particle(
        St=0.1,
        r0_AU=30,
        total_years=2e6,
        dt_years=0.005
    ):

    r   = r0_AU * AU
    phi = 0.0

    dt = dt_years * year

    N = int(total_years / dt_years)

    r_vals   = []
    phi_vals = []
    t_vals   = []

    for i in range(N):

        vr   = v_radial(r, St)
        vphi = v_azimuthal(r, St)

        # evolve system
        r   += vr * dt
        phi += (vphi / r) * dt

        # store
        r_vals.append(r / AU)
        phi_vals.append(phi)
        t_vals.append(i * dt_years / 1e6)

        # stop near star
        if r < 0.1:
            break

    return (
        np.array(t_vals),
        np.array(r_vals),
        np.array(phi_vals)
    )


def perfect_orbit(
        r0_AU=30,
        total_years=1000,
        dt_years=0.005
    ):

    r   = r0_AU * AU
    phi = 0.0

    dt = dt_years * year

    N = int(total_years / dt_years)

    r_vals   = []
    phi_vals = []

    vk = v_K(r)

    for i in range(N):

        phi += (vk / r) * dt

        r_vals.append(r / AU)
        phi_vals.append(phi)

    return (
        np.array(r_vals),
        np.array(phi_vals)
    )


SAVE_DIR = "saved_data"

os.makedirs(SAVE_DIR, exist_ok=True)

Stokes_numbers = [0.01, 0.1, 1]


for St in Stokes_numbers:

    filename = f"{SAVE_DIR}/St_{St}.npz"

    if not os.path.exists(filename):

        print(f"Generating simulation for St={St}")

        t, r, phi = simulate_particle(
            St=St,
            r0_AU=30,
            total_years=2e6,
            dt_years=0.005
        )

        np.savez(
            filename,
            t=t,
            r=r,
            phi=phi
        )

        print(f"Saved -> {filename}")

    else:

        print(f"File already exists -> {filename}")



perfect_file = f"{SAVE_DIR}/perfect_orbit.npz"

if not os.path.exists(perfect_file):

    print("Generating perfect orbit")

    r, phi = perfect_orbit()

    np.savez(
        perfect_file,
        r=r,
        phi=phi
    )

    print(f"Saved -> {perfect_file}")

else:

    print(f"File already exists -> {perfect_file}")


print("\nSaved files:")

for file in os.listdir(SAVE_DIR):

    print(file)