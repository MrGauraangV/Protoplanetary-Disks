import numpy as np
import matplotlib.pyplot as plt

# =========================
# LORENZ SYSTEM
# =========================
dt = 0.01
tmax = 40
time = np.arange(0, tmax+dt, dt)

sigma, rho, beta = 10, 28, 8/3
x = y = z = 1

xvals, yvals, zvals = [], [], []

for t in time:
    xvals.append(x)
    yvals.append(y)
    zvals.append(z)

    x_new = x + sigma*(y-x)*dt
    y_new = y + (x*(rho - z) - y)*dt
    z_new = z + (x*y - beta*z)*dt
    
    x = x_new
    y = y_new
    z = z_new

# =========================
# DOUBLE PENDULUM
# =========================
g, L = 9.8, 1
theta1 = theta2 = np.pi/2
w1 = w2 = 0

theta1_vals, theta2_vals = [], []

dt2 = 0.001
time2 = np.arange(0, tmax+dt2, dt2)

for t in time2:
    theta1_vals.append(theta1)
    theta2_vals.append(theta2)

    dw1 = (-g*(2*np.sin(theta1) - np.sin(theta2)*np.cos(theta1-theta2))
         - np.sin(theta1-theta2)*(w2**2 + w1**2*np.cos(theta1-theta2))) \
        / (2 - np.cos(theta1-theta2)**2)

    dw2 = (2*np.sin(theta1-theta2)*(2*w1**2 + g*np.cos(theta1) + w2**2*np.cos(theta1-theta2))) \
        / (2 - np.cos(theta1-theta2)**2)

    theta1 += w1 * dt2
    theta2 += w2 * dt2
    w1 += dw1 * dt2
    w2 += dw2 * dt2

x1 = L*np.sin(theta1_vals)
y1 = -L*np.cos(theta1_vals)

x2 = x1 + L*np.sin(theta2_vals)
y2 = y1 - L*np.cos(theta2_vals)

# Keep only last portion → cleaner chaos
cut = int(0.6 * len(x2))
x1_plot, y1_plot = x1[cut:], y1[cut:]
x2_plot, y2_plot = x2[cut:], y2[cut:]

# =========================
# DAMPED OSCILLATOR
# =========================
w, gamma = 1, 0.2
x, v = 1, 0

time3 = np.arange(0, 50+dt, dt)
disp, vel = [], []

for t in time3:
    disp.append(x)
    vel.append(v)

    x += v*dt
    v += -(gamma*v + w*w*x)*dt

# =========================
# COMBINED PLOT
# =========================
fig, axes = plt.subplots(2, 3, figsize=(12, 6))

# --- TOP ROW ---
axes[0,0].plot(xvals, yvals, linewidth=1)
axes[0,0].set_title("Lorenz Attractor (x vs y)")

axes[0,1].plot(time2, theta1_vals, linewidth=1)
axes[0,1].set_title("Double Pendulum: θ₁(t)")

axes[0,2].plot(time3, disp, linewidth=1)
axes[0,2].set_title("Damped Oscillator: x(t)")

# --- BOTTOM ROW (IMPROVED VISUALS) ---

# Lorenz (x vs y)
axes[1,0].plot(xvals, zvals, linewidth=0.7)
axes[1,0].set_title("Lorenz Attractor (x vs z)")


# Double Pendulum (cleaned)
axes[1,1].plot(x2_plot, y2_plot, linewidth=0.7, alpha=0.6)
axes[1,1].plot(x1_plot, y1_plot, linewidth=1.2)  # first pendulum emphasized
axes[1,1].set_title("Double Pendulum Trajectory")

# Damped Oscillator
axes[1,2].plot(disp, vel, linewidth=1)
axes[1,2].set_title("Phase Space (x vs v)")

# Clean formatting
for ax in axes.flat:
    ax.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig("dynamical_systems_portfolio.pdf", dpi=300, bbox_inches='tight')
plt.show()