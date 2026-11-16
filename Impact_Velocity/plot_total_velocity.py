
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
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
V_total = data["total"]
A1, A2 = np.meshgrid(a, a)

## TOTAL RELATIVE COLLISION VELOCITY

fig, ax = plt.subplots(figsize=(11, 9))

positive = V_total[V_total > 0]

mesh = ax.pcolormesh(
    A1, A2,
    np.maximum(V_total, 1e-15),
    shading="auto",
    cmap="inferno",
    norm=LogNorm(vmin=positive.min(), vmax=positive.max())
)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("Particle 1 radius [cm]", fontsize=19)
ax.set_ylabel("Particle 2 radius [cm]", fontsize=19)
ax.set_title(
    f"Total Relative Collision Velocity\nα = {alpha:.0e}",
    fontsize=21
)

ax.tick_params(axis="both", which="major", labelsize=16)
ax.tick_params(axis="both", which="minor", labelsize=12)

cbar = fig.colorbar(mesh, ax=ax)
cbar.set_label("Relative velocity [cm/s]", fontsize=18)
cbar.ax.tick_params(labelsize=16)

plt.tight_layout()

## SAVE FIGURE

output_dir = f"Impact_Velocity/Results/{run_name}"
os.makedirs(output_dir, exist_ok=True)

plt.savefig(
    f"{output_dir}/total_collision_velocity.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()
