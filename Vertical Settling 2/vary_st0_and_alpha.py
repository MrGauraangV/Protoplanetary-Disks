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


# PARAMETER

# S = St0 / alpha
#
# Larger S means stronger settling relative to turbulent diffusion.
# S -> infinity corresponds to the no-turbulence limit.

S_values = [
    1e-4,
    1e-2,
    1e-1,
    1e0,
    1e1,
    1e2,
    1e4
]


# VERTICAL GRID

x_span = (
    0.0,
    5.0
)

x = np.linspace(
    0.0,
    5.0,
    2000
)


# GAS DENSITY

def rho_gas(x):
    """
    Vertically isothermal hydrostatic gas density:

        rho_g(x) / rho_g(0) = exp(-x^2/2)

    where x = z/H.
    """

    return np.exp(
        -x**2 / 2.0
    )


# HEIGHT-DEPENDENT EPSTEIN STOPPING TIME

def t_fric_ratio(x):
    """
    Height-dependent Epstein stopping time:

        t_fric(z) / t_fric,0
        =
        rho_g(0) / rho_g(z)

        =
        exp(x^2/2).
    """

    return 1.0 / rho_gas(x)


# STEADY-STATE DUST-TO-GAS RATIO EQUATION

def rhs(x, f, S):
    """
    Steady-state equation:

        df/dx
        =
        -S x
        [t_fric(z) / t_fric,0]
        f

    where

        S = St0 / alpha.
    """

    return [
        -S
        * x
        * t_fric_ratio(x)
        * f[0]
    ]


# INITIAL CONDITION

f0 = [1.0]


# GAS DENSITY ON THE GRID

rho_g = rho_gas(x)


# FIGURE

fig, ax = plt.subplots(
    figsize=(10, 7.5)
)


# DUST DENSITY PROFILES

for S in S_values:

    # Solve the steady-state equation

    sol = solve_ivp(
        rhs,
        x_span,
        f0,
        args=(S,),
        t_eval=x,
        method="Radau",
        rtol=1e-9,
        atol=1e-14
    )

    # Dust-to-gas ratio

    f = sol.y[0]

    # Normalized dust density

    rho_d = f * rho_g

    # Label for the largest S value

    label = f"St0 / α = {S:.0e}"

    ax.plot(
        x,
        rho_d,
        linewidth=2.7,
        label=label
    )


# AXES

ax.set_xlabel(
    "z/H",
    fontsize=17,
    labelpad=8
)

ax.set_ylabel(
    "ρ_d(z) / ρ_d(0)",
    fontsize=17,
    labelpad=8
)

ax.set_title(
    "Effect of Settling-to-Diffusion Ratio on Dust Density",
    fontsize=18,
    pad=14
)


# LIMITS

ax.set_xlim(
    0.0,
    5.0
)

ax.set_ylim(
    0.0,
    1.05
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
    fontsize=12.5,
    loc="upper right",
    frameon=True,
    framealpha=0.95,
    borderpad=0.7,
    labelspacing=0.5
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
    "Vertical Settling 2/dust_density_St0_over_alpha.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05
)

plt.savefig(
    "plots/settling_2_2.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.01
)


# DISPLAY

plt.show()