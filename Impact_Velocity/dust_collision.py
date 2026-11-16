
import numpy as np
import matplotlib.pyplot as plt
import os

## SIMULATION PARAMETERS

alpha = 1e-4
Re = 1e8

a_min = 1e-5
a_max = 1e3
N = 250

delta_vert = alpha
delta_turb = alpha

run_name = f"alpha_{alpha:.0e}"

## DISC PARAMETERS (CGS)

kB = 1.380649e-16
mH = 1.6735575e-24
G = 6.67430e-8
AU = 1.495978707e13
Msun = 1.98847e33

r = 1.0 * AU
Mstar = 1.0 * Msun

T = 280.0
rho_g = 1.4e-9
cs = 1.0e5

rho_s = 1.5

Omega = np.sqrt(G * Mstar / r**3)
vK = Omega * r
H = cs / Omega
eta = 1.8e-3

## STOKES NUMBER (EPSTEIN + STOKES DRAG)

sigma_mol = 2e-15
molecular_mass = 2.34 * mH

mean_free_path = molecular_mass / (rho_g * sigma_mol)
v_th = np.sqrt(8 / np.pi) * cs

def stokes_number(a):

    a = np.asarray(a)

    t_stop_epstein = rho_s * a / (rho_g * v_th)

    t_stop_stokes = (
        4 * rho_s * a**2
        / (9 * rho_g * v_th * mean_free_path)
    )

    t_stop = np.where(
        a < 2.25 * mean_free_path,
        t_stop_epstein,
        t_stop_stokes
    )

    return Omega * t_stop


## PARTICLE MASS

def particle_mass(a):

    return (4/3) * np.pi * rho_s * a**3


## BROWNIAN VELOCITY

def brownian_velocity(a1, a2):

    m1 = particle_mass(a1)
    m2 = particle_mass(a2)

    dv = np.sqrt(
        8 * kB * T * (m1 + m2)
        / (np.pi * m1 * m2)
    )

    return np.minimum(dv, cs)


## TURBULENT VELOCITY

def turbulent_velocity(a1, a2):

    St1 = stokes_number(a1)
    St2 = stokes_number(a2)

    St1, St2 = np.maximum(St1, St2), np.minimum(St1, St2)

    St_min = 1.0 / np.sqrt(Re)

    St_star = np.maximum(
        St_min,
        np.minimum(1.6 * St1, 1.0)
    )

    v_turb = np.sqrt(delta_turb) * cs

    ratio = np.divide(
        St1 - St2,
        St1 + St2,
        out=np.zeros_like(np.asarray(St1 + St2, dtype=float)),
        where=(St1 + St2) > 0
    )

    bracket = (
        St1**2 / (St_star + St1)
        - St1**2 / (1 + St1)
        - St2**2 / (St_star + St2)
        + St2**2 / (1 + St2)
    )

    dv_I_sq = 1.5 * v_turb**2 * ratio * bracket

    term1 = (
        (St_star + St_min) * St1
        + St_star * St_min
    ) / (
        (St_star + St1) * (St1 + St_min)
    )

    term2 = (
        (St_star + St_min) * St2
        + St_star * St_min
    ) / (
        (St_star + St2) * (St2 + St_min)
    )

    dv_II_sq = (
        1.5 * v_turb**2
        * (St_star - St_min)
        * (term1 + term2)
    )

    dv_I = np.sqrt(np.maximum(dv_I_sq, 0))
    dv_II = np.sqrt(np.maximum(dv_II_sq, 0))

    return np.sqrt(dv_I**2 + dv_II**2)


## RADIAL DRIFT VELOCITY

def radial_velocity(a):

    St = stokes_number(a)

    return -2 * eta * vK * St / (1 + St**2)


def radial_relative_velocity(a1, a2):

    return np.abs(
        radial_velocity(a1) - radial_velocity(a2)
    )


## AZIMUTHAL DRIFT VELOCITY

def azimuthal_velocity(a):

    St = stokes_number(a)

    return vK * (1 - eta / (1 + St**2))


def azimuthal_relative_velocity(a1, a2):

    return np.abs(
        azimuthal_velocity(a1) - azimuthal_velocity(a2)
    )


## VERTICAL SETTLING VELOCITY

def dust_scale_height(a):

    St = stokes_number(a)

    factor = np.sqrt(
        delta_vert /
        (np.minimum(St, 0.5) * (1 + St**2))
    )

    return H * np.minimum(1.0, factor)


def vertical_relative_velocity(a1, a2):

    St1 = stokes_number(a1)
    St2 = stokes_number(a2)

    h1 = dust_scale_height(a1)
    h2 = dust_scale_height(a2)

    return Omega * np.abs(
        h1 * np.minimum(St1, 0.5)
        - h2 * np.minimum(St2, 0.5)
    )


## INDIVIDUAL COLLISION CALCULATOR

def calculate_velocities(a1, a2):

    components = {}

    components["Brownian"] = brownian_velocity(a1, a2)
    components["Turbulent"] = turbulent_velocity(a1, a2)
    components["Radial"] = radial_relative_velocity(a1, a2)
    components["Azimuthal"] = azimuthal_relative_velocity(a1, a2)
    components["Vertical"] = vertical_relative_velocity(a1, a2)

    total = np.sqrt(sum(v**2 for v in components.values()))

    return components, total


## USER INPUT

print("\nDUST COLLISION VELOCITY CALCULATOR")

a1 = float(input("Enter particle 1 radius [cm]: "))
a2 = float(input("Enter particle 2 radius [cm]: "))

if a1 <= 0 or a2 <= 0:
    raise ValueError("Particle radii must be positive.")

components, total = calculate_velocities(a1, a2)

print("\nPARTICLE PARAMETERS")
print(f"Particle 1 radius = {a1:.4e} cm")
print(f"Particle 2 radius = {a2:.4e} cm")
print(f"Stokes number 1 = {stokes_number(a1):.4e}")
print(f"Stokes number 2 = {stokes_number(a2):.4e}")

print("\nRELATIVE VELOCITY CONTRIBUTIONS")

for name, velocity in components.items():

    percentage = 100 * velocity**2 / total**2

    print(f"{name:<15} {velocity:>14.6e} cm/s   {percentage:>8.3f}%")

print(f"\nTOTAL IMPACT VELOCITY = {total:.6e} cm/s")
print(f"                         {total/100:.6e} m/s")


## PARTICLE SIZE GRID

a_grid = np.logspace(
    np.log10(a_min),
    np.log10(a_max),
    N
)

A1, A2 = np.meshgrid(a_grid, a_grid)


## CALCULATE VELOCITY COMPONENTS

V_brown = brownian_velocity(A1, A2)
V_turb = turbulent_velocity(A1, A2)
V_rad = radial_relative_velocity(A1, A2)
V_azi = azimuthal_relative_velocity(A1, A2)
V_vert = vertical_relative_velocity(A1, A2)

V_total = np.sqrt(
    V_brown**2
    + V_turb**2
    + V_rad**2
    + V_azi**2
    + V_vert**2
)


## DOMINANT MECHANISM STATISTICS

V_stack = np.stack([
    V_brown**2,
    V_turb**2,
    V_rad**2,
    V_azi**2,
    V_vert**2
])

dominant = np.argmax(V_stack, axis=0)

mechanisms = [
    "Brownian",
    "Turbulent",
    "Radial",
    "Azimuthal",
    "Vertical"
]

print("\nDOMINANT MECHANISM STATISTICS")

for i, name in enumerate(mechanisms):

    percentage = 100 * np.mean(dominant == i)

    print(f"{name:<15}: {percentage:.2f}%")


## SAVE SIMULATION DATA

output_dir = os.path.join("Impact_Velocity", "Results", run_name)

os.makedirs(output_dir, exist_ok=True)

np.savez(
    os.path.join(output_dir, "velocity_maps.npz"),
    particle_sizes=a_grid,
    brownian=V_brown,
    turbulent=V_turb,
    radial=V_rad,
    azimuthal=V_azi,
    vertical=V_vert,
    total=V_total,
    alpha=alpha,
    Re=Re
)

print(f"\nResults saved in: {output_dir}")
print(f"Stokes number range: {stokes_number(a_grid).min():.4e} - {stokes_number(a_grid).max():.4e}")
print("Simulation completed.")
