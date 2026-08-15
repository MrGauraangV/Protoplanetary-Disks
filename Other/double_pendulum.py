from matplotlib.pylab import sin, cos
import numpy as np
import matplotlib.pyplot as plt

#array initialization
theta1_vals = []
theta2_vals = []
w1_vals = []
w2_vals = []

#constants
g = 9.8
L = 1
m = 1

#initial conditions
theta1 = np.pi/2
theta2 = np.pi/2
w1 = 0
w2 = 0

#algorithm setup
dt = 0.001
tmax = 40
time = np.arange(0, tmax+dt, dt)

for t in time:
    theta1_vals.append(theta1)
    theta2_vals.append(theta2)
    w1_vals.append(w1)
    w2_vals.append(w2)

    dw1dt = (-g*(2*sin(theta1) - sin(theta2)*cos(theta1-theta2))
         - sin(theta1-theta2)*(w2*w2 + w1*w1*cos(theta1-theta2))) \
        / (2 - cos(theta1-theta2)**2)
    
    dw2dt = (2*sin(theta1-theta2)*(2*w1*w1 + g*cos(theta1) + w2*w2*cos(theta1-theta2))) \
        / (2 - cos(theta1-theta2)**2)
    
    theta1_new = theta1 + w1 * dt
    theta2_new = theta2 + w2 * dt
    w1_new = w1 + dw1dt * dt
    w2_new = w2 + dw2dt * dt

    theta1 = theta1_new
    theta2 = theta2_new
    w1 = w1_new
    w2 = w2_new

x1 = L*sin(theta1_vals)
x2 = L*sin(theta1_vals) + L*sin(theta2_vals)

y1 = -L*cos(theta1_vals)
y2 = -L*cos(theta1_vals) - L*cos(theta2_vals)

#plotting 

fig, axes = plt.subplots(2, 3, figsize=(10, 4))

axes[0,0].plot(time, theta1_vals)

axes[0,1].plot(time, theta2_vals)

axes[1,0].plot(x1, y1)

axes[1,1].plot(x2, y2)

axes[0,2].plot(x1,y1, 'r')
axes[0,2].plot(x2,y2, 'b')

plt.show()