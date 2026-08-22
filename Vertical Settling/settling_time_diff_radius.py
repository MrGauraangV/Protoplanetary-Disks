import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp


G = 6.67430e-11
k_B = 1.380649e-23
m_H = 1.6735575e-27

AU = 1.495978707e11
M_sun = 1.98847e30


M_star = 1.0 * M_sun

T0 = 150.0
mu = 2.34

Sigma0 = 1700.0
r0 = 1.0 * AU

p_law = -3/2
t_law = -3/7

rho_m = 1000.0
f = 1e-2

v_th = 1000.0
N_H = 5.0


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


def rho_g(z, r):
    H = scale_height(r)
    rho0 = Sigma_g(r) / (np.sqrt(2 * np.pi) * H)
    return rho0 * np.exp(-0.5 * (z / H)**2)


def rhs(t, y, r):

    m = y[0]
    z = y[1]

    rho = rho_g(z, r)
    Omega = omega_K(r)

    s = (3 * m / (4 * np.pi * rho_m))**(1/3)

    dmdt = (
        3/4
        * Omega**2
        * f
        / v_th
        * z
        * m
    )

    dzdt = (
        -(rho_m / rho)
        * (s / v_th)
        * Omega**2
        * z
    )

    return [dmdt, dzdt]


s0_um = 1.0
s0 = s0_um * 1e-6

m0 = (
    4/3
    * np.pi
    * rho_m
    * s0**3
)


r_values = np.array([
    0.1,
    0.3,
    1.0,
    3.0,
    10.0,
    30.0,
    100.0
]) * AU


t0 = 0.0
tf = 1000.0 * 365.25 * 24 * 3600

t_eval = np.linspace(
    t0,
    tf,
    1200
)


fig, ax = plt.subplots(figsize=(9, 6))

ax.set_title(
    f"Particle Growth During Vertical Settling "
    f"($s_0={s0_um:g}\\,\\mu m$)",
    fontsize=13,
    fontweight="bold"
)


for r in r_values:

    H = scale_height(r)
    z0 = N_H * H

    y0 = [
        m0,
        z0
    ]

    sol = solve_ivp(
        lambda t, y: rhs(t, y, r),
        (t0, tf),
        y0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-12
    )

    t = sol.t
    m = sol.y[0]

    s = (
        3 * m /
        (4 * np.pi * rho_m)
    )**(1/3)

    ax.plot(
        t / (365.25 * 24 * 3600),
        s / 1e-6,
        label=f"{r/AU:g} AU"
    )

    print(
        f"r = {r/AU:g} AU, "
        f"final size = {s[-1]/1e-6:.5f} μm"
    )


ax.set_xlabel("Time [yr]")
ax.set_ylabel(r"Particle size $s$ [$\mu$m]")

ax.set_yscale("log")

ax.grid(
    True,
    which="both",
    alpha=0.3
)

ax.legend(title="Radius")

plt.tight_layout()
plt.show()