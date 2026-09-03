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

# 1700 g/cm² converted to kg/m²
Sigma0 = 1000.0 * 10.0

r0 = 1.0 * AU

p_law = -3.0 / 2.0
t_law = -3.0 / 7.0



# DUST / TURBULENCE PARAMETERS


rho_m = 1000.0

f0 = 1e-2

alpha = 1e-6

v_th = 1000.0

N_H = 5.0



# INITIAL PARTICLE SIZES


initial_sizes = [
    0.01,
    1.0,
    100.0
]   # mm



# DISK FUNCTIONS


def temperature(r):

    return (
        T0
        * (r / r0)**t_law
    )


def sound_speed(T):

    return np.sqrt(
        k_B * T / (mu * m_H)
    )


def omega_K(r):

    return np.sqrt(
        G * M_star / r**3
    )


def scale_height(r):

    T = temperature(r)

    cs = sound_speed(T)

    return cs / omega_K(r)


def Sigma_g(r):

    return (
        Sigma0
        * (r / r0)**p_law
    )



# GAS DENSITY


def rho_g(z, r):

    H = scale_height(r)

    rho0 = (
        Sigma_g(r)
        / (np.sqrt(2.0 * np.pi) * H)
    )

    return (
        rho0
        * np.exp(
            -0.5 * (z / H)**2
        )
    )



# MIDPLANE GAS DENSITY


def rho_g0(r):

    H = scale_height(r)

    return (
        Sigma_g(r)
        / (np.sqrt(2.0 * np.pi) * H)
    )



# MIDPLANE STOKES NUMBER


def St0(s, r):

    return (
        omega_K(r)
        * rho_m
        * s
        / (
            rho_g0(r)
            * v_th
        )
    )



# TURBULENT DUST-TO-GAS RATIO


def dust_to_gas_ratio(z, s, r):

    H = scale_height(r)

    x = z / H

    St = St0(s, r)

    exponent = (
        -(St / alpha)
        * (
            np.exp(x**2 / 2.0)
            - 1.0
        )
    )

    return (
        f0
        * np.exp(exponent)
    )



# NON-TURBULENT GROWTH EQUATION


def rhs_no_turbulence(z, s, r):

    # No turbulence:
    #
    # f = f0
    #
    # ds/dz = -f0 rho_g / (4 rho_m)

    rho = rho_g(
        z,
        r
    )

    ds_dz = (
        -f0
        * rho
        / (4.0 * rho_m)
    )

    return [
        ds_dz
    ]



# TURBULENT GROWTH EQUATION


def rhs_turbulent(z, s, r):

    rho = rho_g(
        z,
        r
    )

    f = dust_to_gas_ratio(
        z,
        s[0],
        r
    )

    rho_d = (
        f
        * rho
    )

    ds_dz = (
        -rho_d
        / (4.0 * rho_m)
    )

    return [
        ds_dz
    ]



# RADIAL GRID


r_values = np.logspace(
    np.log10(0.1),
    np.log10(500.0),
    1000
) * AU



# FIGURE


fig, ax = plt.subplots(
    figsize=(10, 7.5)
)



# COLORS


# One colour for each initial particle size.
# The same colour is used for the turbulent
# and non-turbulent curves.

colors = plt.rcParams[
    "axes.prop_cycle"
].by_key()["color"]



# SOLVE FOR EACH INITIAL SIZE


for i, s0_mm in enumerate(initial_sizes):

    turbulent_final_sizes = []

    non_turbulent_final_sizes = []

    # Same colour for both cases

    color = colors[
        i % len(colors)
    ]


    
    # RADIAL LOOP
    

    for r in r_values:

        
        # Local scale height
        

        H = scale_height(r)

        
        # Initial and final heights
        

        zi = N_H * H
        zf = 0.0

        
        # Initial particle size
        

        s0 = s0_mm / 1000.0

        
        # Vertical integration grid
        

        z_eval = np.linspace(
            zi,
            zf,
            500
        )


        
        # NON-TURBULENT SOLUTION
        

        sol_no_turbulence = solve_ivp(
            lambda z, s: rhs_no_turbulence(
                z,
                s,
                r
            ),
            (zi, zf),
            [s0],
            t_eval=z_eval,
            method="RK45",
            rtol=1e-8,
            atol=1e-12
        )


        s_final_no_turbulence = (
            sol_no_turbulence.y[0, -1]
        )


        
        # TURBULENT SOLUTION
        

        sol_turbulent = solve_ivp(
            lambda z, s: rhs_turbulent(
                z,
                s,
                r
            ),
            (zi, zf),
            [s0],
            t_eval=z_eval,
            method="Radau",
            rtol=1e-8,
            atol=1e-12
        )


        s_final_turbulent = (
            sol_turbulent.y[0, -1]
        )


        
        # PREVENT NEGATIVE NUMERICAL VALUES
        

        s_final_no_turbulence = max(
            s_final_no_turbulence,
            0.0
        )

        s_final_turbulent = max(
            s_final_turbulent,
            0.0
        )


        
        # CONVERT m -> mm
        

        non_turbulent_final_sizes.append(
            s_final_no_turbulence
            * 1000.0
        )

        turbulent_final_sizes.append(
            s_final_turbulent
            * 1000.0
        )


    
    # PLOT NON-TURBULENT
    

    ax.loglog(
        r_values / AU,
        non_turbulent_final_sizes,
        linewidth=2.3,
        linestyle="--",
        color=color,
        label="_nolegend_"
    )


    
    # PLOT TURBULENT
    

    ax.loglog(
        r_values / AU,
        turbulent_final_sizes,
        linewidth=2.6,
        linestyle="-",
        color=color,
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
    "Particle Growth Turbulent vs. Laminar",
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
    fontsize=11.5,
    loc="lower left",
    frameon=True,
    framealpha=0.95,
    borderpad=0.7,
    labelspacing=0.5
)


# AXIS LIMITS


ax.set_xlim(
    0.1,
    500.0
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
    "Vertical Settling 3/particle_growth_turbulent_vs_no_turbulence.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05
)
plt.savefig(
    "plots/settling_3_3.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.01
)


# DISPLAY


plt.show()