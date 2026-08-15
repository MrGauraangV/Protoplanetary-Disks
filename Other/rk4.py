import numpy as np
import matplotlib.pyplot as plt

def lorenz(t, state, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])

def rk4_step(f, t, y, dt):
    k1 = f(t, y)
    k2 = f(t + dt/2, y + dt*k1/2)
    k3 = f(t + dt/2, y + dt*k2/2)
    k4 = f(t + dt, y + dt*k3)
    
    return y + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

dt = 0.01
t = 0
state = np.array([1.0, 1.0, 1.0])

trajectory = []

for _ in range(10000):
    state = rk4_step(lorenz, t, state, dt)
    t += dt
    trajectory.append(state)

trajectory = np.array(trajectory)

x = trajectory[:,0]
y = trajectory[:,1]
z = trajectory[:,2]

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(x, y, z)
plt.show()