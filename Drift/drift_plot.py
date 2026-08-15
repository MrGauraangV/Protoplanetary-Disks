import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


SAVE_DIR = "saved_data"

data_001 = np.load(f"{SAVE_DIR}/St_0.01.npz")
t_001   = data_001["t"]
r_001   = data_001["r"]
phi_001 = data_001["phi"]

data_01 = np.load(f"{SAVE_DIR}/St_0.1.npz")
t_01   = data_01["t"]
r_01   = data_01["r"]
phi_01 = data_01["phi"]

data_1 = np.load(f"{SAVE_DIR}/St_1.npz")
t_1   = data_1["t"]
r_1   = data_1["r"]
phi_1 = data_1["phi"]

perfect = np.load(f"{SAVE_DIR}/perfect_orbit.npz")
r_perfect   = perfect["r"]
phi_perfect = perfect["phi"]


G     = 6.67430e-11
M_sun = 1.989e30
AU    = 1.496e11
year  = 365.25 * 24 * 3600

M_star = M_sun
eta = 0.002

def Omega_K(r):

    return np.sqrt(G * M_star / r**3)


def v_K(r):

    return r * Omega_K(r)


def v_radial(r, St):

    vk = v_K(r)

    return -2.0 * eta * vk * St / (1.0 + St**2)


def sparse_orbit_plot(
        r,
        phi,
        plot_orbits=1,
        skip_orbits=5
    ):

    """
    Plot a few full consecutive orbits,
    then skip several orbits,
    then repeat.

    Example:
        plot 1 orbit
        skip 5 orbits
        plot 1 orbit
        skip 5 orbits
    """

    orbit_num = phi / (2*np.pi)

    orbit_int = orbit_num.astype(int)

    cycle = plot_orbits + skip_orbits

    mask = (orbit_int % cycle) < plot_orbits

    return r[mask], phi[mask]


def plot_spiral(
        ax,
        r,
        phi,
        St,
        plot_orbits=1,
        skip_orbits=5
    ):

    r_plot, phi_plot = sparse_orbit_plot(
        r,
        phi,
        plot_orbits=plot_orbits,
        skip_orbits=skip_orbits
    )

    x = r_plot * np.cos(phi_plot)
    y = r_plot * np.sin(phi_plot)

    ax.plot(
        x,
        y,
        lw=1.2
    )

    # star
    ax.scatter(
        0,
        0,
        marker='*',
        s=180,
        zorder=10
    )

    # drift time estimate
    r0 = 30 * AU

    td = (
        r0 /
        np.abs(v_radial(r0, St))
    ) / year / 1e6

    ax.set_title(
        f"St = {St}\n"
        f"Drift Time ≈ {td:.2f} Myr",
        fontsize=13
    )

    ax.set_xlabel("x [AU]")
    ax.set_ylabel("y [AU]")

    ax.set_aspect('equal')

    ax.grid(alpha=0.3)


fig = plt.figure(figsize=(18, 8))

# Right graph shorter vertically
gs = GridSpec(
    2,
    3,
    width_ratios=[1, 1, 1.8],
    height_ratios=[1, 1],
    wspace=0.30,
    hspace=0.35
)

# Left panels
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[1,0])
ax4 = fig.add_subplot(gs[1,1])

# Right panel ONLY upper part
axR = fig.add_subplot(gs[:,2])

# manually shrink vertical height
pos = axR.get_position()

axR.set_position([
    pos.x0,
    pos.y0 + 0.10,
    pos.width,
    pos.height * 0.65
])


plot_spiral(
    ax1,
    r_001,
    phi_001,
    0.01,
    plot_orbits=1,
    skip_orbits=50
)

plot_spiral(
    ax2,
    r_01,
    phi_01,
    0.1,
    plot_orbits=1,
    skip_orbits=50
)

plot_spiral(
    ax3,
    r_1,
    phi_1,
    1,
    plot_orbits=1,
    skip_orbits=50
)


r_plot, phi_plot = sparse_orbit_plot(
    r_perfect,
    phi_perfect,
    plot_orbits=1,
    skip_orbits=15
)

x = r_plot * np.cos(phi_plot)
y = r_plot * np.sin(phi_plot)

ax4.plot(
    x,
    y,
    lw=1.2
)

ax4.scatter(
    0,
    0,
    marker='*',
    s=180
)

ax4.set_title(
    "Perfect Keplerian Orbit\n(No Gas Drag)",
    fontsize=13
)

ax4.set_xlabel("x [AU]")
ax4.set_ylabel("y [AU]")

ax4.set_aspect('equal')

ax4.grid(alpha=0.3)


axR.plot(
    t_001,
    r_001,
    lw=2,
    label='St = 0.01'
)

axR.plot(
    t_01,
    r_01,
    lw=2,
    label='St = 0.1'
)

axR.plot(
    t_1,
    r_1,
    lw=2,
    label='St = 1'
)

axR.axhline(
    30,
    linestyle='--',
    lw=2,
    label='Keplerian (No Drag)'
)

axR.set_xscale('log')
axR.set_yscale('log')

axR.set_xlabel(
    "Time [Myr]",
    fontsize=14
)

axR.set_ylabel(
    "Radius [AU]",
    fontsize=14
)

axR.set_title(
    "Radius vs Time",
    fontsize=18
)

axR.grid(alpha=0.3)

axR.legend(fontsize=12)


fig.suptitle(
    "Dust Spiral Decay vs Pure Keplerian Orbit",
    fontsize=22,
    y=0.98
)

plt.savefig(
    "dust_spiral_decay.png",
    dpi=600,                 # very high resolution
    bbox_inches='tight',    # removes extra whitespace
    facecolor='white'
)

plt.savefig(
    "dust_spiral_decay.pdf",   # vector format (BEST for PPT)
    bbox_inches='tight',
    facecolor='white'
)