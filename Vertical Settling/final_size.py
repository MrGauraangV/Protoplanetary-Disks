import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from astropy.constants import G, M_sun, k_B, m_p
from astropy import units as u

# PARAMETERS

M_star = 1.0 * M_sun

# Radius -- CHANGE THIS to compare different disk locations
r = 1.0 * u.au

# IMPORTANT:
# T and Sigma are CONSTANT during the vertical evolution
# at a given r.
T = 280.0 * u.K
Sigma_g = 1.0e3 * u.g / u.cm**2

# Dust-to-gas mass ratio
f_d = 1.0e-2

# Material density of dust
rho_s = 3.0 * u.g / u.cm**3

# Mean molecular weight
mu = 2.34

# Initial particle size
a0 = 1.0e-4 * u.cm

# Initial height
z0_fraction_H = 3.0


# BASIC DISK QUANTITIES

# Angular frequency
Omega = np.sqrt(G * M_star / r**3)

# Sound speed
c_s = np.sqrt(k_B * T / (mu * m_p))

# Scale height
H = c_s / Omega

# Midplane gas density
rho_g0 = Sigma_g / (np.sqrt(2.0 * np.pi) * H)

# Thermal velocity
v_th = np.sqrt(8.0 / np.pi) * c_s

# Initial height
z0 = z0_fraction_H * H


# GAS DENSITY

def gas_density(z):
    """
    Vertically isothermal Gaussian gas density.
    Sigma_g and T remain fixed at this radius.
    """
    return rho_g0 * np.exp(-z**2 / (2.0 * H**2))


# STOPPING TIME

def stopping_time(a, z):
    """
    Epstein stopping time.
    """
    rho_g = gas_density(z)

    return rho_s * a / (rho_g * v_th)


# RELATIVE VELOCITY

def relative_velocity(a, z):
    """
    Relative velocity used for coagulation.

    Here we use the differential vertical-settling
    velocity scale.

    For equal-sized particles, this is an approximate
    coagulation velocity. If you have a specific
    v_rel formula from our previous derivation, replace
    this function only.
    """

    t_stop = stopping_time(a, z)

    v_set = Omega**2 * t_stop * z

    return np.abs(v_set)


# COAGULATION + SETTLING EQUATIONS

def equations(t, y):

    # y[0] = z
    # y[1] = a

    z = y[0] * u.cm
    a = y[1] * u.cm

    rho_g = gas_density(z)

    # Dust density
    rho_d = f_d * rho_g

    # Stopping time
    t_stop = stopping_time(a, z)

    # Vertical settling

    dz_dt = -Omega**2 * t_stop * z

    # Relative velocity for coagulation

    v_rel = relative_velocity(a, z)

    # Particle growth
    #
    # da/dt = rho_d * v_rel / (4 rho_s)

    da_dt = rho_d * v_rel / (4.0 * rho_s)

    return [
        dz_dt.to(u.cm / u.s).value,
        da_dt.to(u.cm / u.s).value
    ]


# SETTLING EVENT

def settled(t, y):

    z = y[0]

    # Stop when |z| reaches 10^-4 H
    return z - (1e-4 * H).to(u.cm).value


settled.terminal = True
settled.direction = -1


# TIME SPAN

# Maximum integration time
t_max = 1e7 * u.yr

t_span = (
    0.0,
    t_max.to(u.s).value
)


# INITIAL CONDITIONS

y0 = [
    z0.to(u.cm).value,
    a0.to(u.cm).value
]


# SOLVE

solution = solve_ivp(
    equations,
    t_span,
    y0,
    events=settled,
    rtol=1e-8,
    atol=1e-12,
    max_step=(1e3 * u.yr).to(u.s).value
)


# RESULTS

time = solution.t * u.s
z = solution.y[0] * u.cm
a = solution.y[1] * u.cm

time_yr = time.to(u.yr).value
z_over_H = (z / H).value
a_um = a.to(u.um).value


print("\n========================================")
print("VERTICAL COAGULATION + SETTLING")
print("========================================")

print(f"Radius              = {r.to(u.au):.3g}")
print(f"Temperature         = {T:.3g}")
print(f"Surface density     = {Sigma_g:.3g}")
print(f"Omega               = {Omega.to(1/u.s):.3e}")
print(f"Scale height H      = {H.to(u.au):.3e}")
print(f"Midplane density    = {rho_g0.to(u.g/u.cm**3):.3e}")

print("\nInitial conditions:")
print(f"Initial size        = {a0.to(u.um):.3e}")
print(f"Initial z/H         = {z0_fraction_H}")

print("\nFinal result:")
print(f"Final size          = {a[-1].to(u.cm):.6e}")
print(f"Final size          = {a[-1].to(u.um):.6e}")
print(f"Growth factor       = {(a[-1]/a0).value:.3e}")
print(f"Settling time       = {time[-1].to(u.yr):.6e}")


# PLOTS

fig, ax = plt.subplots(1, 2, figsize=(12, 5))


# z/H vs time

ax[0].plot(
    time_yr,
    z_over_H
)

ax[0].set_xlabel("Time [yr]")
ax[0].set_ylabel(r"$z/H$")
ax[0].set_yscale("log")
ax[0].set_title("Vertical settling")



ax[1].plot(
    time_yr,
    a_um
)

ax[1].set_xlabel("Time [yr]")
ax[1].set_ylabel(r"Particle radius [$\mu$m]")
ax[1].set_yscale("log")
ax[1].set_title("Vertical coagulation")

plt.tight_layout()
plt.show()