
import numpy as np
import matplotlib.pyplot as plt

w = 1
gamma = 0.2

dt = 0.01
tmax = 50
time = np.arange(0, tmax+dt, dt)

x = 1
v = 0

displacement = []
velocity = []

for t in time:
    displacement.append(x)
    velocity.append(v)

    x_new = x + v*dt
    v_new = v - (gamma * v + w*w*x)*dt

    x = x_new
    v = v_new

plt.subplot(1,2,1)
plt.plot(time, displacement)

plt.subplot(1,2,2)
plt.plot(displacement, velocity)

plt.show()
    