import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import erf



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

# 1700 g/cm^2 = 17000 kg/m^2
Sigma0 = 1700.0 * 10.0

r0 = 1.0 * AU

p_law = -3 / 2
t_law = -3 / 7



# DUST PARAMETERS


rho_m = 1000.0       # particle material density [kg/m^3]

f = 1e-2             # dust-to-gas mass ratio

N_H = 1.0            # starting height in units of H



# DISK FUNCTIONS


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



# GAS DENSITY


def rho_g(z, r):

    H = scale_height(r)

    rho0 = Sigma_g(r) / (np.sqrt(2.0 * np.pi) * H)

    return rho0 * np.exp(-(z / H)**2 / 2.0)



# DIFFERENTIAL EQUATION


def rhs(z, s, r):

    rho = rho_g(z, r)

    ds_dz = -rho * f / (4.0 * rho_m)

    return [ds_dz]



# INITIAL PARTICLE SIZES


initial_sizes = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0
]   # mm



# RADIAL GRID


r_values = np.logspace(
    np.log10(0.1),
    np.log10(100.0),
    500
) * AU



# CALCULATE ERRORS


print("Maximum relative error")
print("----------------------")


for s0_mm in initial_sizes:

    final_sizes_numeric = []
    final_sizes_analytic = []


    ----
    # Loop over radius
    ----

    for r in r_values:

        # Local scale height
        H = scale_height(r)

        # Initial and final heights
        zi = N_H * H
        zf = 0.0

        # Initial particle size: mm -> m
        s0 = s0_mm / 1000.0


        # ====================================================
        # NUMERICAL SOLUTION
        # ====================================================

        z_eval = np.linspace(
            zi,
            zf,
            500
        )

        sol = solve_ivp(
            lambda z, s: rhs(z, s, r),
            (zi, zf),
            [s0],
            t_eval=z_eval,
            method="RK45",
            rtol=1e-8,
            atol=1e-12
        )

        s_final_numeric = sol.y[0, -1]


        # ====================================================
        # ANALYTICAL SOLUTION
        # ====================================================

        growth = (
            f * Sigma_g(r)
            / (8.0 * rho_m)
            * (
                erf(
                    zi / (np.sqrt(2.0) * H)
                )
                -
                erf(
                    zf / (np.sqrt(2.0) * H)
                )
            )
        )

        s_final_analytic = s0 + growth


        # ====================================================
        # STORE
        # ====================================================

        final_sizes_numeric.append(
            s_final_numeric
        )

        final_sizes_analytic.append(
            s_final_analytic
        )


    # Convert to arrays
    final_sizes_numeric = np.array(final_sizes_numeric)
    final_sizes_analytic = np.array(final_sizes_analytic)


    # ========================================================
    # RELATIVE ERROR
    # ========================================================

    relative_error = (
        np.abs(
            final_sizes_numeric -
            final_sizes_analytic
        )
        / np.abs(final_sizes_analytic)
    )


    # Maximum relative error over all radii
    max_error = np.max(relative_error)


    # ========================================================
    # OUTPUT
    # ========================================================

    print(
        f"s0 = {s0_mm:g} mm : "
        f"{max_error:.6e}"
    )