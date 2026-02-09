import numpy as np
import matplotlib.pyplot as plt
from subfunctions import *

terrain_slope = 0
Crr_array = np.linspace(0.01, 0.5, 25)
v_max = np.zeros(len(Crr_array))

ng = get_gear_ratio(rover['wheel_assembly']['speed_reducer'])
radius = rover['wheel_assembly']['wheel']['radius']
w_noload = motor['speed_noload']

# --- Main Loop ---
for i in range(len(Crr_array)):
    current_Crr = Crr_array[i]
    
    # bis range
    a = 0.0
    b = w_noload
    
    tol = 1e-6
    max_iter = 100
    root_omega = 0
    
    #same as prev
    f_a = F_net(a, terrain_slope, rover, planet, current_Crr)
    f_b = F_net(b, terrain_slope, rover, planet, current_Crr)
    
    for q in range(max_iter):
        c = (a + b) / 2
        f_c = F_net(c, terrain_slope, rover, planet, current_Crr)
        
        if abs(f_c) < tol or (b - a) / 2 < tol:
            root_omega = c
            break
        
        if np.sign(f_c) == np.sign(f_a):
            a = c
        else:
            b = c

    v_max[i] = (root_omega / ng) * radius

plt.plot(Crr_array, v_max)
plt.title("Max Velocity vs. Rolling Resistance")
plt.xlabel("Coefficient of Rolling Resistance")
plt.ylabel("Max Rover Velocity [m/s]")

plt.show()
