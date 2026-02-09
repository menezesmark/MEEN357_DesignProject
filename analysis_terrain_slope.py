import numpy as np
import matplotlib.pyplot as plt
from subfunctions import *

Crr = 0.15

slope_array_deg = np.linspace(-15, 35, 25)
v_max = np.zeros(len(slope_array_deg))

ng = get_gear_ratio(rover['wheel_assembly']['speed_reducer'])
radius = rover['wheel_assembly']['wheel']['radius']


for i in range(len(slope_array_deg)):
    angle = slope_array_deg[i]
    
    # bicstion rang
    a = 0.0
    b = motor['speed_noload']
    
    tol = 1e-6
    max_iter = 1000
    
    f_start = F_net(a, angle, rover, planet, Crr)
    
    for q in range(max_iter):
        c = (a + b) / 2
        f_c = F_net(c, angle, rover, planet, Crr)
        
        if f_c == 0 or (b - a) / 2 < tol:
            omega_eq = c
            break
        
        if np.sign(f_c) == np.sign(f_start):
            a = c
        else:
            b = c
    

    v_max[i] = (omega_eq / ng) * radius

plt.plot(slope_array_deg, v_max)
plt.title("Max Velocity vs. Terrain Slope")
plt.xlabel("Terrain Angle [deg]")
plt.ylabel("Max Rover Velocity [m/s]")

plt.show()
