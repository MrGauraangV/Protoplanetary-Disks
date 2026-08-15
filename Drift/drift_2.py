import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u
from astropy.constants import G, M_sun


fig, (ax1, ax2) = plt.subplots(
    2, 1,
    figsize=(7, 8),
    sharex=False
)

# Figure 4.2 : Radial drift velocity

eta = 7.5e-3
v_r_gas = -3.75e-5          # in units of v_K
St = np.logspace(-5, 4, 1000)

v_r = (v_r_gas - eta * St) / (1 + St**2)

ax1.loglog(
    St,
    np.abs(v_r),
    color="black"
)

ax1.set_xlabel(r"$\tau_{\rm fric}=t_{\rm fric}\Omega_K$")
ax1.set_ylabel(r"$|v_r|/v_K$")

ax1.grid(alpha=0.3)

# Figure 4.3 : Minimum drift time

r = np.logspace(-1, 2, 500) * u.au

for h in [0.025, 0.05, 0.075]:

    eta = 1.5 * h**2
    St = 1
    vK = np.sqrt(G * M_sun / r)

    v_r = (
        -eta * St / (1 + St**2)
    ) * vK

    t_drift = (r / np.abs(v_r)).to(u.yr)

    ax2.loglog(
        r.to(u.au).value,
        t_drift.value,
        label=f"$h/r={h}$"
    )

ax2.set_xlabel("r (AU)")
ax2.set_ylabel(r"$t_{\rm drift}$ (yr)")

ax2.grid(alpha=0.3)

ax2.legend()
ax1.set_title(
    "The Radial Drift Velocity of Particles at the Mid-plane",
    fontsize=13,
    pad=10
)

ax2.set_title(
    "Minimum Time Scale for the Radial Drift of Solid Particles\nas a Function of Radius",
    fontsize=13,
    pad=10
)
plt.tight_layout()
plt.show()