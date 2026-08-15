import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Parameters
# ----------------------------
m = 1.0
k = 1.0
alpha = 0.5      # Cubic potential strength

# ----------------------------
# Potential
# V(x) = 1/2 k x^2 + 1/3 alpha x^3
# ----------------------------
def potential(x):
    return 0.5*k*x**2 + (alpha/3)*x**3

# ----------------------------
# Equations of Motion
# dx/dt = v
# dv/dt = -(dV/dx)/m
# ----------------------------
def system(t, state):
    x, v = state

    dx = v
    dv = -(k*x + alpha*x**2)/m

    return np.array([dx, dv])

# ----------------------------
# RK4 Step
# ----------------------------
def rk4_step(f, t, y, dt):

    k1 = f(t, y)
    k2 = f(t + dt/2, y + dt*k1/2)
    k3 = f(t + dt/2, y + dt*k2/2)
    k4 = f(t + dt, y + dt*k3)

    return y + dt*(k1 + 2*k2 + 2*k3 + k4)/6

# ----------------------------
# Initial Conditions
# ----------------------------
state = np.array([1.0, 0.0])

dt = 0.01
t = 0

trajectory = []

for _ in range(5000):
    trajectory.append(state.copy())
    state = rk4_step(system, t, state, dt)
    t += dt

trajectory = np.array(trajectory)

x_traj = trajectory[:,0]
v_traj = trajectory[:,1]

# ----------------------------
# Phase Space Grid
# ----------------------------
x = np.linspace(-3,3,30)
v = np.linspace(-3,3,30)

X,V = np.meshgrid(x,v)

# Vector field
U = V
W = -(k*X + alpha*X**2)/m

# Normalize arrows
mag = np.sqrt(U**2 + W**2)
mag[mag==0] = 1

U = U/mag
W = W/mag

# ----------------------------
# Energy
# ----------------------------
Energy = 0.5*m*V**2 + potential(X)

# ----------------------------
# Plot
# ----------------------------
plt.figure(figsize=(8,7))

# Filled energy contours
plt.contourf(X, V, Energy,
             levels=30,
             cmap="viridis",
             alpha=0.7)

# Energy contour lines
plt.contour(X, V, Energy,
            levels=20,
            colors="black",
            linewidths=0.5)

# Vector field
plt.quiver(X, V, U, W,
           color="white",
           pivot="mid",
           scale=35)

# RK4 trajectory
plt.plot(x_traj,
         v_traj,
         color="red",
         linewidth=2,
         label="Trajectory")

# Starting point
plt.scatter(x_traj[0],
            v_traj[0],
            color="yellow",
            s=80,
            edgecolor="black",
            label="Start")

plt.xlabel("Position x")
plt.ylabel("Velocity v")
plt.title("Phase Space with Energy Contours and Vector Field")
plt.colorbar(label="Energy")
plt.legend()
plt.grid(True)

plt.show()