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

p_law = -3.0 / 2.0
t_law = -3.0 / 7.0



# DUST / TURBULENCE PARAMETERS


alpha = 1e-3

rho_m = 1000.0

f0 = 1e-2

v_th = 1000.0

N_H = 5.0



# INITIAL PARTICLE SIZES


initial_sizes_um = [
    1.0,
    0.1,
    0.01
]



# RADIAL LOCATION


r = 1.0 * AU



# DISK FUNCTIONS


def temperature(r):

    return T0 * (r / r0)**t_law


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

    return Sigma0 * (
        r / r0
    )**p_law



# LOCAL DISK QUANTITIES


H = scale_height(r)

Omega = omega_K(r)

Sigma = Sigma_g(r)

rho_g0 = (
    Sigma
    / (np.sqrt(2.0 * np.pi) * H)
)



# GAS DENSITY


def rho_gas(z):

    return (
        rho_g0
        * np.exp(
            -0.5 * (z / H)**2
        )
    )



# MIDPLANE STOKES NUMBER


def St0_of_s(s):

    return (
        Omega
        * rho_m
        * s
        / (rho_g0 * v_th)
    )



# STOPPING TIME


def t_fric(z, s):

    rho = rho_gas(z)

    return (
        rho_m
        * s
        / (rho * v_th)
    )



# TURBULENT DUST-TO-GAS RATIO


def dust_to_gas_ratio(z, s):

    x = z / H

    St0 = St0_of_s(s)

    return (
        f0
        * np.exp(
            -(St0 / alpha)
            * (
                np.exp(x**2 / 2.0) - 1.0
            )
        )
    )



# TURBULENT EQUATIONS


def rhs_turbulent(t, y):

    s = y[0]
    z = y[1]

    if z < 0.0:
        z = 0.0

    rho = rho_gas(z)

    f = dust_to_gas_ratio(z, s)

    rho_d = f * rho

    t_stop = t_fric(z, s)

    v_settle = (
        t_stop
        * Omega**2
        * z
    )

    # Particle growth

    dsdt = (
        rho_d
        / (4.0 * rho_m)
        * v_settle
    )

    # Vertical settling

    dzdt = -v_settle

    return [
        dsdt,
        dzdt
    ]



# NON-TURBULENT / LAMINAR EQUATIONS


def rhs_laminar(t, y):

    s = y[0]
    z = y[1]

    if z < 0.0:
        z = 0.0

    rho = rho_gas(z)

    # In the laminar case the dust-to-gas ratio
    # remains equal to its midplane value.

    rho_d = f0 * rho

    t_stop = t_fric(z, s)

    v_settle = (
        t_stop
        * Omega**2
        * z
    )

    # Particle growth

    dsdt = (
        rho_d
        / (4.0 * rho_m)
        * v_settle
    )

    # Vertical settling

    dzdt = -v_settle

    return [
        dsdt,
        dzdt
    ]



# INITIAL HEIGHT


zi = N_H * H



# TIME RANGE


t_max = (
    1200.0
    * 365.25
    * 24.0
    * 3600.0
)

t_eval = np.linspace(
    0.0,
    t_max,
    1500
)



# MIDPLANE EVENT


def reach_midplane(t, y):

    return y[1]


reach_midplane.terminal = True
reach_midplane.direction = -1



# LINE STYLES


# Color = initial particle size
# Solid = turbulent
# Dashed = laminar

linestyles = [
    "-",
    "-",
    "-"
]



# FIGURE 1:
# PARTICLE SIZE VS TIME


fig1, ax1 = plt.subplots(
    figsize=(10, 7.5)
)


for s0_um, ls in zip(
    initial_sizes_um,
    linestyles
):

    
    # Initial conditions
    

    s0 = s0_um * 1e-6

    y0 = [
        s0,
        zi
    ]


    
    # TURBULENT SOLUTION
    

    sol_turb = solve_ivp(
        rhs_turbulent,
        (0.0, t_max),
        y0,
        method="Radau",
        rtol=1e-9,
        atol=[
            1e-18,
            1e-3
        ],
        t_eval=t_eval,
        events=reach_midplane
    )


    
    # LAMINAR SOLUTION
    

    sol_lam = solve_ivp(
        rhs_laminar,
        (0.0, t_max),
        y0,
        method="RK45",
        rtol=1e-9,
        atol=[
            1e-18,
            1e-3
        ],
        t_eval=t_eval,
        events=reach_midplane
    )


    
    # TURBULENT CURVE
    

    turbulent_line = ax1.plot(
        sol_turb.t / (
            365.25 * 24.0 * 3600.0
        ),
        sol_turb.y[0] / 1e-6,
        linewidth=2.6,
        linestyle=ls,
        label=f"{s0_um:g} μm, turbulent"
    )[0]


    # Get the color assigned to the turbulent curve

    curve_color = turbulent_line.get_color()


    
    # LAMINAR CURVE
    

    ax1.plot(
        sol_lam.t / (
            365.25 * 24.0 * 3600.0
        ),
        sol_lam.y[0] / 1e-6,
        linewidth=2.2,
        linestyle=":",
        color=curve_color,
        alpha=0.8,
        label=f"{s0_um:g} μm, laminar"
    )



# AXES


ax1.set_xlabel(
    "Time [yr]",
    fontsize=17,
    labelpad=8
)

ax1.set_ylabel(
    "Particle size s [μm]",
    fontsize=17,
    labelpad=8
)

ax1.set_title(
    "Particle Growth During Vertical Settling",
    fontsize=18,
    pad=14
)

ax1.set_yscale("log")



# TICKS


ax1.tick_params(
    axis="both",
    which="major",
    direction="in",
    length=6,
    width=1.1
)

ax1.tick_params(
    axis="both",
    which="minor",
    direction="in",
    length=3,
    width=0.8
)



# GRID


ax1.grid(
    True,
    which="major",
    linestyle="-",
    alpha=0.20
)

ax1.grid(
    True,
    which="minor",
    linestyle=":",
    alpha=0.12
)



# LEGEND


ax1.legend(
    fontsize=11.5,
    loc="lower right",
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



# SAVE FIGURE 1


fig1.savefig(
    "Vertical Settling 3/particle_size_vs_time_turbulent_vs_laminar.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05
)
fig1.savefig(
    "plots/settling_3_1.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.01
)


# FIGURE 2:
# z/H VS TIME


fig2, ax2 = plt.subplots(
    figsize=(10, 7.5)
)


for s0_um, ls in zip(
    initial_sizes_um,
    linestyles
):

    
    # Initial conditions
    

    s0 = s0_um * 1e-6

    y0 = [
        s0,
        zi
    ]


    
    # TURBULENT SOLUTION
    

    sol_turb = solve_ivp(
        rhs_turbulent,
        (0.0, t_max),
        y0,
        method="Radau",
        rtol=1e-9,
        atol=[
            1e-18,
            1e-3
        ],
        t_eval=t_eval,
        events=reach_midplane
    )


    
    # LAMINAR SOLUTION
    

    sol_lam = solve_ivp(
        rhs_laminar,
        (0.0, t_max),
        y0,
        method="RK45",
        rtol=1e-9,
        atol=[
            1e-18,
            1e-3
        ],
        t_eval=t_eval,
        events=reach_midplane
    )


    
    # TURBULENT z/H
    

    turbulent_line = ax2.plot(
        sol_turb.t / (
            365.25 * 24.0 * 3600.0
        ),
        sol_turb.y[1] / H,
        linewidth=2.6,
        linestyle=ls,
        label=f"{s0_um:g} μm, turbulent"
    )[0]


    # Get the turbulent curve color

    curve_color = turbulent_line.get_color()


    
    # LAMINAR z/H
    

    ax2.plot(
        sol_lam.t / (
            365.25 * 24.0 * 3600.0
        ),
        sol_lam.y[1] / H,
        linewidth=2.2,
        linestyle=":",
        color=curve_color,
        alpha=0.8,
        label=f"{s0_um:g} μm, laminar"
    )



# AXES


ax2.set_xlabel(
    "Time [yr]",
    fontsize=17,
    labelpad=8
)

ax2.set_ylabel(
    "z/H",
    fontsize=17,
    labelpad=8
)

ax2.set_title(
    "Vertical Dust Settling",
    fontsize=18,
    pad=14
)



# LIMITS


ax2.set_xlim(
    0.0,
    1200.0
)

ax2.set_ylim(
    0.0,
    5.2
)



# TICKS


ax2.tick_params(
    axis="both",
    which="major",
    direction="in",
    length=6,
    width=1.1
)

ax2.tick_params(
    axis="both",
    which="minor",
    direction="in",
    length=3,
    width=0.8
)



# GRID


ax2.grid(
    True,
    which="major",
    linestyle="-",
    alpha=0.20
)

ax2.grid(
    True,
    which="minor",
    linestyle=":",
    alpha=0.12
)



# LEGEND


ax2.legend(
    fontsize=11.5,
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



# SAVE FIGURE 2


fig2.savefig(
    "Vertical Settling 3/z_over_H_vs_time_turbulent_vs_laminar.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.05
)
fig2.savefig(
    "plots/settling_3_2.pdf",
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.01
)


# SHOW


plt.show()