
import matplotlib.pyplot as plt
import numpy as np
from astropy.constants import G, M_sun
from astropy import units as u

r = 1 * u.au
surface_density = 1e3 * u.g / u.cm**2   #at r = 1AU
rho_m = 3 * u.g / u.cm**3
rho = 6e-10 * u.g / u.cm**3

v_th = 1e5 * u.cm / u.s
c_s = v_th * np.sqrt(np.pi/8)
particle_sizes = [1 * u.micrometer, 10 * u.micrometer, 1000 * u.micrometer] 

Omega_k = np.sqrt(G*M_sun/pow(r,3))
h = c_s / Omega_k

z = np.linspace(0, 5, 1000) * h

# for fixed z:
# print(f"v= {v_settle.to(u.cm / u.s)}, \n t= {t_settle.to(u.Myr)}")

def settling_time(s, z):
    t_0 = 2/np.pi * surface_density / rho_m / s / Omega_k
    return t_0 * np.exp(-pow(z,2) / (2*h**2))

for s in particle_sizes:
    plt.plot(
        z.to(u.au).value,
        settling_time(s, z).to(u.Myr).value,
        label=f"{s.to(u.micrometer).value:.0f} μm"
    )

plt.legend(title="Particle size")

plt.xlabel("z (AU)")
plt.ylabel("Settling time (Myr)")
plt.grid(True)
plt.show()