
import numpy as np
import matplotlib.pyplot as plt

dt = 0.01
tmax = 40
time = np.arange(0, tmax+dt, dt)

xvals = []
yvals = []
zvals = []

sigma = 10
rho = 28
beta = 8/3

x = y = z = 1

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

plt.subplot(2,3,1)
plt.plot(time, xvals)

plt.subplot(2,3,2)
plt.plot(time, yvals)

plt.subplot(2,3,3)
plt.plot(time, zvals)

plt.subplot(2,3,4)
plt.plot(xvals, yvals)

plt.subplot(2,3,5)
plt.plot(xvals, zvals)

plt.subplot(2,3,6)
plt.plot(yvals, zvals)

plt.show()
    