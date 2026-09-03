import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp



# PLOT STYLE


plt.rcParams.update({
    "axes.linewidth": 1.2,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "xtick.major.pad": 8,
    "ytick.major.pad": 8,
})



# CONSTANTS


G = 6.67430e-11
k_B = 1.380649e-23
m_H = 1.6735575e-27

AU = 1.495978707e11
M_sun = 1.98847e30

year = 365.25 * 24.0 * 3600.0



# DISK PARAMETERS


M_star = 1.0 * M_sun

T0 = 150.0
mu = 2.34

# 1700 g/cm^2 -> kg/m^2
Sigma0 = 1000.0 * 10.0

r0 = 1.0 * AU

p_law = -3.0 / 2.0
t_law = -3.0 / 7.0



# DUST / TURBULENCE PARAMETERS


rho_m = 1000.0

f0 = 1e-2

v_th = 1000.0

N_H = 5.0



# INITIAL PARTICLE SIZE


# Same particle at every radius
s0_mm = 0.1

s0 = s0_mm / 1000.0       # mm -> m



# TURBULENCE VALUES


alphas = [
    0.0,
    1e-4,
    1e-3,
    1e-2,
    1e-1
]



# TIMES TO SHOW


times_yr = [
    0,
    200,
    400,
    600,
    800,
    1000
]

times = np.array(times_yr) * year



# RADIAL GRID


r_values = np.logspace(
    np.log10(0.1),
    np.log10(100.0),
    250
) * AU



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



# DUST-TO-GAS RATIO


def dust_to_gas_ratio(z, s, r, alpha):

    # No turbulence
    if alpha == 0.0:
        return f0

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



# STOPPING TIME


def stopping_time(z, s, r):

    rho = rho_g(z, r)

    return (
        rho_m
        * s
        / (rho * v_th)
    )



# COUPLED SETTLING + GROWTH EQUATIONS


def rhs(t, y, r, alpha):

    z = y[0]
    s = y[1]

    # Prevent numerical problems once particle reaches
    # the midplane
    if z < 0.0:
        z = 0.0

    # Prevent negative particle size
    s = max(s, 1e-30)

    # Local gas density
    rho = rho_g(z, r)

    # Dust-to-gas ratio
    f = dust_to_gas_ratio(
        z,
        s,
        r,
        alpha
    )

    # Dust density
    rho_d = f * rho

    # Local stopping time
    t_fric = stopping_time(
        z,
        s,
        r
    )

    # Keplerian frequency
    Omega = omega_K(r)

    # Settling velocity
    v_settle = (
        t_fric
        * Omega**2
        * z
    )

    
    # Vertical settling
    

    dz_dt = -v_settle

    
    # Particle growth
    #
    # This is the time-dependent form of the same
    # ds/dz = -rho_d/(4 rho_m)
    # used in your existing code.
    

    ds_dt = (
        rho_d
        * v_settle
        / (4.0 * rho_m)
    )

    return [
        dz_dt,
        ds_dt
    ]



# SOLVE ONE RADIAL LOCATION


def solve_at_radius(r, alpha):

    H = scale_height(r)

    # Start at 5H, same vertical setup used
    # in the settling calculation
    z_initial = N_H * H

    y0 = [
        z_initial,
        s0
    ]

    sol = solve_ivp(
        lambda t, y: rhs(
            t,
            y,
            r,
            alpha
        ),
        (0.0, times[-1]),
        y0,
        t_eval=times,
        method="Radau",
        rtol=1e-8,
        atol=1e-12
    )

    return sol.y[0], sol.y[1]



# CALCULATE s(r,t)


results = {}

for alpha in alphas:

    print(f"alpha = {alpha:g}")

    results[alpha] = np.zeros(
        (len(times), len(r_values))
    )

    for j, r in enumerate(r_values):

        z_solution, s_solution = solve_at_radius(
            r,
            alpha
        )

        results[alpha][:, j] = (
            s_solution * 1000.0
        )



# SIX-PANEL FIGURE


fig, axes = plt.subplots(
    2,
    3,
    figsize=(13, 7),
    sharex=True,
    sharey=True
)

axes = axes.flatten()



# PLOT


for i, t_yr in enumerate(times_yr):

    ax = axes[i]

    for j, alpha in enumerate(alphas):

        if alpha == 0.0:
            label = r"$\alpha=0$"
        else:
            label = rf"$\alpha={alpha:g}$"

        ax.loglog(
            r_values / AU,
            results[alpha][i],
            linewidth=2.0,
            label=label
        )

    ax.set_title(
        rf"$t={t_yr}\,\mathrm{{yr}}$",
        fontsize=14
    )

    ax.grid(
        True,
        which="both",
        alpha=0.2
    )



# AXIS LABELS


fig.supxlabel(
    r"Orbital radius $r$ [AU]",
    fontsize=16
)

fig.supylabel(
    r"Particle size $s$ [mm]",
    fontsize=16
)



# SIMPLE LEGEND


axes[0].legend(
    fontsize=11,
    frameon=False
)



# LAYOUT


plt.tight_layout()

plt.show()