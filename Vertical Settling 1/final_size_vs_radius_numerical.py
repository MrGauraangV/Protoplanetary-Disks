import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp



# PLOT STYLE


plt.rcParams.update({
    "axes.linewidth": 1.2,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "xtick.major.pad": 10,
    "ytick.major.pad": 10,
})



# CONSTANTS


G = 6.67430e-11
k_B = 1.380649e-23
m_H = 1.6735575e-27

AU = 1.495978707e11
M_sun = 1.98847e30



# DISK PARAMETERS


M_star = 1.0 * M_sun

T0 = 150.0
mu = 2.34

# 1000 g/cm² converted to kg/m²
Sigma0 = 1000.0 * 10.0

r0 = 1.0 * AU

p_law = -3 / 2
t_law = -3 / 7



# DUST PARAMETERS


rho_m = 1000.0       # particle material density [kg/m³]

f = 1e-2             # dust-to-gas mass ratio

N_H = 1.0            # starting height in units of H



# DISK FUNCTIONS


def temperature(r):
    """
    Midplane temperature.
    """
    return T0 * (r / r0)**t_law


def sound_speed(T):
    """
    Isothermal sound speed.
    """
    return np.sqrt(
        k_B * T / (mu * m_H)
    )


def omega_K(r):
    """
    Keplerian angular frequency.
    """
    return np.sqrt(
        G * M_star / r**3
    )


def scale_height(r):
    """
    Gas scale height:

        H = c_s / Ω_K
    """
    T = temperature(r)

    cs = sound_speed(T)

    return cs / omega_K(r)


def Sigma_g(r):
    """
    Gas surface density.
    """
    return Sigma0 * (
        r / r0
    )**p_law



# GAS DENSITY


def rho_g(z, r):
    """
    Vertically isothermal hydrostatic gas density:

        ρ_g(z) = ρ_0 exp[-z²/(2H²)]
    """

    H = scale_height(r)

    rho0 = (
        Sigma_g(r)
        / (np.sqrt(2 * np.pi) * H)
    )

    return (
        rho0
        * np.exp(
            -(z / H)**2 / 2.0
        )
    )



# DIFFERENTIAL EQUATION


def rhs(z, s, r):
    """
    Particle growth equation:

        ds/dz = -ρ_g f / (4ρ_m)

    Integration proceeds downward from z_i to z = 0.
    """

    rho = rho_g(z, r)

    ds_dz = (
        -rho * f
        / (4.0 * rho_m)
    )

    return [ds_dz]



# INITIAL PARTICLE SIZES


initial_sizes = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0
]   # mm



# RADIAL GRID


r_values = np.logspace(
    np.log10(0.1),
    np.log10(1000.0),
    1000
) * AU



# FIGURE


fig, ax = plt.subplots(
    figsize=(10, 7.5)
)



# SOLVE AND PLOT


for s0_mm in initial_sizes:

    final_sizes = []

    for r in r_values:

        
        # Local gas scale height
        

        H = scale_height(r)

        
        # Initial and final heights
        

        zi = N_H * H
        zf = 0.0

        
        # Convert initial size from mm to m
        

        s0 = s0_mm / 1000.0

        
        # Vertical integration grid
        

        z_eval = np.linspace(
            zi,
            zf,
            500
        )

        
        # Numerical solution
        

        sol = solve_ivp(
            lambda z, s: rhs(z, s, r),
            (zi, zf),
            [s0],
            t_eval=z_eval,
            method="RK45",
            rtol=1e-8,
            atol=1e-12
        )

        
        # Final particle size
        

        s_final = sol.y[0, -1]

        # Convert m back to mm

        final_sizes.append(
            s_final * 1000.0
        )

    # Plot radial dependence

    ax.loglog(
        r_values / AU,
        final_sizes,
        linewidth=2.6,
        label=f"s0 = {s0_mm:g} mm"
    )



# AXES


ax.set_xlabel(
    "Orbital radius r [AU]",
    fontsize=17,
    labelpad=8
)

ax.set_ylabel(
    "Final particle size s [mm]",
    fontsize=17,
    labelpad=8
)

ax.set_title(
    "Particle Growth Finale (Laminar Well-Mixed Disk)",
    fontsize=18,
    pad=14
)



# TICKS


ax.tick_params(
    axis="both",
    which="major",
    direction="in",
    length=6,
    width=1.1
)

ax.tick_params(
    axis="both",
    which="minor",
    direction="in",
    length=3,
    width=0.8
)



# GRID


ax.grid(
    True,
    which="major",
    linestyle="-",
    alpha=0.20
)

ax.grid(
    True,
    which="minor",
    linestyle=":",
    alpha=0.12
)



# LEGEND

ax.legend(
    title="Initial particle size",
    fontsize=12.5,
    title_fontsize=13,
    loc="lower left",
    frameon=True,
    framealpha=0.95,
    borderpad=0.7,
    labelspacing=0.5
)

# AXIS LIMITS


ax.set_xlim(
    0.1,
    1000.0
)



# LAYOUT


plt.subplots_adjust(
    left=0.13,
    right=0.97,
    bottom=0.13,
    top=0.88
)



# SAVE AS PDF


plt.savefig(
    "Vertical Settling 1/particle_growth_vertical_settling.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05
)

plt.savefig(
    "plots/settling_1.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.01
)


# DISPLAY


plt.show()