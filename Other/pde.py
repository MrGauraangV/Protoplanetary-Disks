import numpy as np
import matplotlib.pyplot as plt

nx = 10
dx = 2
dt = 0.1
k = 0.835
nt_list = [30,60,90,120,150,180,210]  # number of time steps

alpha = k
lambda_ = alpha * dt / dx**2

# initial condition
T = np.zeros(nx)
T[0] = 100
T[-1] = 50

for nt in nt_list:
    for _ in range(nt):
        T_new = T.copy()
        
        for i in range(1, nx-1):
            T_new[i] = T[i] + lambda_*(T[i+1] - 2*T[i] + T[i-1])
        
        # boundary conditions (fixed)
        T_new[0] = 100
        T_new[-1] = 50
        
        T = T_new

    plt.plot(T, marker='o',label=f'nt = {nt}')

plt.title("Temperature distribution")
plt.legend()
plt.show()