
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import os

## PLOT FORMATTING

plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 21,
    "axes.labelsize": 19,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "figure.titlesize": 23,
    "figure.dpi": 150,
    "savefig.dpi": 300
})

## SELECT SIMULATION

alpha = 1e-2
run_name = f"alpha_{alpha:.0e}"

data_path = f"Impact_Velocity/Results/{run_name}/velocity_maps.npz"

## LOAD DATA

data = np.load(data_path)

a = data["particle_sizes"]
A1, A2 = np.meshgrid(a, a)

V_stack = np.stack([
    data["brownian"]**2,
    data["turbulent"]**2,
    data["radial"]**2,
    data["azimuthal"]**2,
    data["vertical"]**2
])

## DOMINANT COLLISION MECHANISM

dominant = np.argmax(V_stack, axis=0)

mechanisms = [
    "Brownian",
    "Turbulent",
    "Radial drift",
    "Azimuthal drift",
    "Vertical settling"
]

## DOMINANT MECHANISM STATISTICS

print("\nDOMINANT MECHANISM STATISTICS")

for i, name in enumerate(mechanisms):
    count = np.sum(dominant == i)
    percentage = 100 * count / dominant.size
    print(f"{name:<20}: {percentage:.2f}%")

## PLOT DOMINANT MECHANISM

cmap = ListedColormap([
    "royalblue",
    "darkorange",
    "forestgreen",
    "crimson",
    "mediumpurple"
])

norm = BoundaryNorm(np.arange(-0.5, 5.5, 1), cmap.N)

fig, ax = plt.subplots(figsize=(11, 9))

mesh = ax.pcolormesh(
    A1, A2,
    dominant,
    shading="auto",
    cmap=cmap,
    norm=norm
)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("Particle 1 radius [cm]", fontsize=19)
ax.set_ylabel("Particle 2 radius [cm]", fontsize=19)
ax.set_title(
    f"Dominant Collision Mechanism\nα = {alpha:.0e}",
    fontsize=21
)

ax.tick_params(axis="both", which="major", labelsize=16)
ax.tick_params(axis="both", which="minor", labelsize=12)

cbar = fig.colorbar(mesh, ax=ax, ticks=np.arange(5))
cbar.ax.set_yticklabels(mechanisms, fontsize=15)
cbar.set_label("Dominant mechanism", fontsize=18)
cbar.ax.tick_params(labelsize=15)

plt.tight_layout()

## SAVE FIGURE

output_dir = f"Impact_Velocity/Results/{run_name}"
os.makedirs(output_dir, exist_ok=True)

plt.savefig(
    f"{output_dir}/dominant_collision_mechanism.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()
