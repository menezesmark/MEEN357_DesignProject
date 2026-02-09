import numpy as np
import matplotlib.pyplot as plt
from subfunctions import *

Crr = 0.15

slope_array_deg = np.linspace(-75, 75, 1000)
v_max = np.zeros(len(slope_array_deg))

# conv omega to rover velo
def omega_to_velocity(omega, rover):
    radius = rover['wheel_assembly']['wheel']['radius']
    ng = get_gear_ratio(rover['wheel_assembly']['speed_reducer'])
    return radius * omega / ng

# max velo for each slope
for i in range(len(slope_array_deg)):
    angle = slope_array_deg[i]
    
    omega_low = 0
    omega_high = motor['speed_noload']
    
    tol = 1e-5 
    max_iter = 100
    
    for q in range(max_iter):
        omega_mid = (omega_low + omega_high) / 2
        
        # net f @ midpoint
        f_mid = F_net(omega_mid, angle, rover, planet, Crr)
        
        if f_mid == 0 or (omega_high - omega_low) / 2 < tol: #check if close
            break
        
        f_low = F_net(omega_low, angle, rover, planet, Crr)
        
        if np.sign(f_mid) == np.sign(f_low):
            omega_low = omega_mid
        else:
            omega_high = omega_mid
            
    v_max[i] = omega_to_velocity(omega_mid, rover)

plt.plot(slope_array_deg, v_max)
plt.title("Max Velocity vs. Terrain Slope")
plt.xlabel("Terrain Angle [deg]")
plt.ylabel("Max Rover Velocity [m/s]")

plt.show()

