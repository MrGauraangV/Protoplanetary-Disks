import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf

# Constants

G = 6.67430e-11
k_B = 1.380649e-23
m_H = 1.6735575e-27
AU = 1.495978707e11
M_sun = 1.98847e30

# Disk parameters

M_star = 1.0 * M_sun
T0 = 100.0
mu = 2.34

Sigma0 = 1700.0
r0 = 1.0 * AU
p = 1.5

f = 0.01
rho_m = 1000.0

# Vertical positions

N_H = 1.0
zf = 0.0

# Initial particle sizes

initial_sizes = [0.01, 0.1, 1.0, 10, 100.0]  # mm

# Sound speed

def sound_speed(T):
    return np.sqrt(k_B * T / (mu * m_H))

# Keplerian angular frequency

def omega_K(r):
    return np.sqrt(G * M_star / r**3)

# Gas scale height

def scale_height(r):
    cs = sound_speed(T0)
    return cs / omega_K(r)

# Gas surface density

def Sigma_g(r):
    return Sigma0 * (r / r0)**(-p)

# Particle size

def particle_size(r, s0_mm):

    s0 = s0_mm / 1000.0  # mm -> m

    h = scale_height(r)
    zi = N_H * h

    growth = (
        f * Sigma_g(r)
        / (8.0 * rho_m)
        * (
            erf(zi / (np.sqrt(2.0) * h))
            - erf(zf / (np.sqrt(2.0) * h))
        )
    )

    return (s0 + growth) * 1000.0  # m -> mm

# Simplified particle size

def particle_size_simplified(r, s0_mm):

    s0 = s0_mm / 1000.0  # mm -> m

    growth = (
        f * Sigma_g(r)
        / (8.0 * rho_m)
    )

    return (s0 + growth) * 1000.0  # m -> mm

# Radial grid

r = np.logspace(
    np.log10(0.1),
    np.log10(100.0),
    500
) * AU

# Plot particle size vs radius

fig, ax = plt.subplots(figsize=(8, 5))

for s0_mm in initial_sizes:

    s = particle_size(r, s0_mm)
    s_simplified = particle_size_simplified(r, s0_mm)

    ax.loglog(
        r / AU,
        s,
        label=f"$s_0$ = {s0_mm:g} mm"
    )
    """
    ax.loglog(
         r / AU,
         s_simplified,
         "--"
     )
     """

ax.set_xlabel("Radius [AU]")
ax.set_ylabel("Particle size $s$ [mm]")
ax.set_title("Particle Size vs Radius")

ax.grid(True, which="both", alpha=0.3)
ax.legend()

# Parameter box

params = (
    f"$z_i$ = {N_H} H\n"
    f"$z_f$ = {zf / AU:.2f} AU\n"
    f"$f$ = {f:.3f}\n"
    f"$T_0$ = {T0:.0f} K\n"
    f"$p$ = {p:.1f}"
)

ax.text(
    0.97,
    0.97,
    params,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="top",
    horizontalalignment="right",
    bbox=dict(
        boxstyle="round,pad=0.4",
        facecolor="white",
        edgecolor="black",
        alpha=0.8
    )
)

plt.tight_layout()
plt.show()