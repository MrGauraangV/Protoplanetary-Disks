import numpy as np
import matplotlib.pyplot as plt

dt = 0.01
tmax = 50
omega = 1

time = np.arange(0, tmax, dt)

x = 1        # initial position
v = 0        # initial velocity

xvals = []
vvals = []

for t in time:

    xvals.append(x)
    vvals.append(v)

    x_new = x + v*dt
    v_new = v - omega**2 * x * dt

    x = x_new
    v = v_new

plt.plot(xvals, vvals)
plt.xlabel("x")
plt.ylabel("v")
plt.show()