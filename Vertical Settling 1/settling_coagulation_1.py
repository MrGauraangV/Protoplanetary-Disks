import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

from astropy.constants import G, M_sun
from astropy import units as u

# DISK PARAMETERS

r = 1 * u.au
Sigma = 1e3 * u.g / u.cm**2
H = 3e11 * u.cm
rho_m = 3 * u.g / u.cm**3
f = 1e-2
v_th = 1e5 * u.cm / u.s
z0 = 5 * H

Omega = np.sqrt(G * M_sun / r**3).to(1/u.s)

rho0 = Sigma / (np.sqrt(2*np.pi) * H)


def rho_g(z):
    return rho0 * np.exp(-(z/H)**2 / 2)

# DIFFERENTIAL EQUATIONS

def rhs(t, y):

    m = y[0] * u.g
    z = y[1] * u.cm

    rho = rho_g(z)

    s = (3*m/(4*np.pi*rho_m))**(1/3)

    # Mass growth
    dmdt = (
        3/4 * Omega**2 * f / v_th * z * m
    )

    # Vertical settling
    dzdt = (
        -(rho_m/rho)
        * (s/v_th)
        * Omega**2
        * z
    )

    return [
        dmdt.to(u.g/u.s).value,
        dzdt.to(u.cm/u.s).value
    ]

# INITIAL CONDITIONS

initial_sizes = [0.01, 0.1, 1]   # micrometers

linestyles = [
    "-",
    "--",
    (0, (8, 4))
]

# TIME

t0 = 0 * u.yr
tf = 1000 * u.yr

t_eval = np.linspace(
    t0.to(u.s).value,
    tf.to(u.s).value,
    1200
)

# FIGURE

fig, ax = plt.subplots(figsize=(7, 5))

ax.set_title(
    "Dust Settling in a Laminar Protoplanetary Disk",
    fontsize=13,
    fontweight="bold"
)

# SOLVE AND PLOT

for s0_um, ls in zip(initial_sizes, linestyles):

    s0 = s0_um * u.micrometer

    m0 = (
        4/3 * np.pi * rho_m * s0**3
    )

    sol = solve_ivp(
        rhs,
        (
            t0.to(u.s).value,
            tf.to(u.s).value
        ),
        [
            m0.to(u.g).value,
            z0.to(u.cm).value
        ],
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-12
    )

    t = sol.t * u.s
    m = sol.y[0] * u.g
    z = sol.y[1] * u.cm

    ax.plot(
        t.to(u.yr).value,
        (z/H).value,
        ls=ls,
        color="black",
        label=f"{s0_um} μm"
    )


# FORMAT

ax.set_ylabel(r"$z/H$")
ax.set_xlabel("Time (yr)")

ax.set_ylim(0, 5.2)

ax.grid(alpha=0.3)

ax.legend(title="Initial size")

plt.tight_layout()
plt.show()