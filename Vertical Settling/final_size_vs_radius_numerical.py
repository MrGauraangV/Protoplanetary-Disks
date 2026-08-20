import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp


# Constants

G = 6.67430e-11
k_B = 1.380649e-23
m_H = 1.6735575e-27
AU = 1.495978707e11
M_sun = 1.98847e30


# DISK PARAMETERS

M_star = 1.0 * M_sun

T0 = 150.0
mu = 2.34

Sigma0 = 1700.0
r0 = 1.0 * AU

p_law = -3/2
t_law = -3/7

rho_m = 1000.0
f = 1e-2

N_H = 1.0


# DISK FUNCTIONS

def temperature(r):
    return T0 * (r / r0)**t_law


def sound_speed(T):
    return np.sqrt(k_B * T / (mu * m_H))


def omega_K(r):
    return np.sqrt(G * M_star / r**3)


def scale_height(r):
    T = temperature(r)
    cs = sound_speed(T)
    return cs / omega_K(r)


def Sigma_g(r):
    return Sigma0 * (r / r0)**p_law


# GAS DENSITY

def rho_g(z):

    H = scale_height(r)

    rho0 = Sigma_g(r) / (np.sqrt(2 * np.pi) * H)

    return rho0 * np.exp(-(z / H)**2 / 2)


# DIFFERENTIAL EQUATION

def rhs(z, s):

    rho = rho_g(z)

    ds_dz = -rho * f / (4 * rho_m)

    return [ds_dz]


# INITIAL PARTICLE SIZES

initial_sizes = [0.01, 0.1, 1.0, 10.0, 100.0]   # mm

# RADIAL GRID

r_values = np.logspace(
    np.log10(0.1),
    np.log10(100.0),
    500
) * AU


# FIGURE

fig, ax = plt.subplots(figsize=(13, 8))

ax.set_title(
    "Particle Growth During Vertical Settling",
    fontsize=13,
    fontweight="bold"
)


# SOLVE AND PLOT

for s0_mm in initial_sizes:

    final_sizes = []

    for r in r_values:

        H = scale_height(r)

        zi = N_H * H
        zf = 0.0

        s0 = s0_mm / 1000.0

        z_eval = np.linspace(zi, zf, 500)

        sol = solve_ivp(
            rhs,
            (zi,zf),
            [s0],
            t_eval=z_eval,
            method="RK45",
            rtol=1e-8,
            atol=1e-12
        )

        s = sol.y[0]

        s_final = s[-1]

        final_sizes.append(s_final * 1000.0)


    ax.loglog(
        r_values / AU,
        final_sizes,
        label=f"$s_0$ = {s0_mm:g} mm"
    )


# FORMAT

ax.set_xlabel("Radius [AU]")
ax.set_ylabel("Final particle size $s$ [mm]")

ax.grid(True, which="both", alpha=0.3)

ax.legend(title="Initial size")

# PARAMETER BOX

params = (
    f"$z_i$ = {N_H} H\n"
    f"$z_f$ = 0\n"
    f"$f$ = {f:.3f}\n"
    f"$T_0$ = {T0:.0f} K\n"
    f"$\\Sigma \\propto r^{{{p_law:.1f}}}$\n"
    f"$T \\propto r^{{{t_law:.1f}}}$"
)

fig.text(
    0.82,
    0.75,
    params,
    fontsize=10,
    verticalalignment="top",
    horizontalalignment="left",
    bbox=dict(
        boxstyle="round,pad=0.4",
        facecolor="white",
        edgecolor="black",
        alpha=0.8
    )
)

plt.tight_layout(rect=[0, 0, 0.78, 1])
plt.show()