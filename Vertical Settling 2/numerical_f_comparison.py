import numpy as np
import matplotlib.pyplot as plt



# PLOT STYLE


plt.rcParams.update({
    "axes.linewidth": 1.2,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "xtick.major.pad": 10,
    "ytick.major.pad": 10,
})



# PARAMETERS


alpha = 1e-2
St0 = 1e-2



# VERTICAL GRID


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



# WELL-MIXED DUST


def rho_d_well_mixed(x):
    """
    Well-mixed dust distribution:

        f(z) = f0

    Therefore:

        rho_d(z) = f0 rho_g(z)

    and hence

        rho_d(z) / rho_d(0)
        =
        rho_g(z) / rho_g(0)
        =
        exp(-x^2/2).
    """

    return np.exp(
        -x**2 / 2.0
    )



# CONSTANT STOPPING-TIME APPROXIMATION


def rho_d_constant_tfric(x):
    """
    Constant stopping time approximation.

    The steady-state dust-to-gas ratio is

        f(z) / f0
        =
        exp[-(St0/alpha) x^2/2].

    Therefore,

        rho_d(z) / rho_d(0)
        =
        exp[-x^2/2]
        exp[-(St0/alpha) x^2/2].
    """

    return np.exp(
        -0.5 * x**2
        * (
            1.0
            + St0 / alpha
        )
    )



# HEIGHT-DEPENDENT EPSTEIN STOPPING TIME


def rho_d_changing_tfric(x):
    """
    Full height-dependent Epstein stopping-time solution.

    For a vertically isothermal disk,

        t_fric(z) / t_fric,0
        =
        exp(x^2/2).

    The steady-state dust density is

        rho_d(z)
        =
        rho_g,0 f0
        exp[
            -x^2/2
            -(St0/alpha)
            (exp(x^2/2)-1)
        ].

    Therefore,

        rho_d(z) / rho_d(0)
        =
        exp[
            -x^2/2
            -(St0/alpha)
            (exp(x^2/2)-1)
        ].
    """

    return np.exp(
        -x**2 / 2.0
        - (St0 / alpha)
        * (
            np.exp(x**2 / 2.0)
            - 1.0
        )
    )



# CALCULATE PROFILES


rho_well_mixed = rho_d_well_mixed(x)

rho_constant = rho_d_constant_tfric(x)

rho_changing = rho_d_changing_tfric(x)



# FIGURE


fig, ax = plt.subplots(
    figsize=(10, 7.5)
)



# WELL-MIXED DUST


ax.plot(
    x,
    rho_well_mixed,
    linewidth=2.8,
    linestyle="--",
    label=r"Well-mixed dust: f(z) = f0"
)



# CONSTANT STOPPING TIME


ax.plot(
    x,
    rho_constant,
    linewidth=2.7,
    linestyle=":",
    label=r"Turbulence, constant stopping time"
)



# HEIGHT-DEPENDENT STOPPING TIME


ax.plot(
    x,
    rho_changing,
    linewidth=2.8,
    linestyle="-",
    label=r"Turbulence, height-dependent stopping time"
)


# AXES


ax.set_xlabel(
    r"z/H",
    fontsize=17,
    labelpad=8
)

ax.set_ylabel(
    r"ρ_d(z) / ρ_d(0)",
    fontsize=17,
    labelpad=8
)

ax.set_title(
    "Comparison of Vertical Dust Density Profiles",
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
    "Vertical Settling 2/dust_density_comparison.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05
)
plt.savefig(
    "plots/settling_2_1.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.01
)


# DISPLAY


plt.show()