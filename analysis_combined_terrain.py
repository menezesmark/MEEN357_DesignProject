import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from subfunctions import *

Crr_array = np.linspace(0.01, 0.5, 25)
slope_array_deg = np.linspace(-15, 35, 25)
CRR, SLOPE = np.meshgrid(Crr_array, slope_array_deg)
VMAX = np.zeros(np.shape(CRR), dtype=float)

ng = get_gear_ratio(rover['wheel_assembly']['speed_reducer'])
radius = rover['wheel_assembly']['wheel']['radius']
w_noload = motor['speed_noload']

rows, cols = np.shape(CRR)

for i in range(rows):
    for j in range(cols):
        # cur pos
        c_current = CRR[i, j]
        slope_current = SLOPE[i, j]
        
        # brackets for bis
        a = 0.0
        b = w_noload
        
        fa = F_net(a, slope_current, rover, planet, c_current)
        fb = F_net(b, slope_current, rover, planet, c_current)
        
        # check case
        if np.sign(fa) == np.sign(fb):
            if fa > 0:
                # runaway case so nan
                VMAX[i, j] = np.nan
            else:
                # stall case so zero ig?
                VMAX[i, j] = 0.0
        else:
            # else will be normal so we chill and same math as before
            tol = 1e-5
            root_omega = 0.0
            
            for _ in range(100): # Max 100 iterations
                c = (a + b) / 2
                fc = F_net(c, slope_current, rover, planet, c_current)
                
                if abs(fc) < tol or (b - a) / 2 < tol:
                    root_omega = c
                    break
                
                if np.sign(fc) == np.sign(fa):
                    a = c
                else:
                    b = c
            
            VMAX[i, j] = (root_omega / ng) * radius


fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(CRR, SLOPE, VMAX, cmap='viridis', edgecolor='none')

ax.set_xlabel('Rolling Resistance (Crr)')
ax.set_ylabel('Terrain Slope (deg)')
ax.set_zlabel('Max Velocity (m/s)')
plt.title('Rover Maximum Velocity Surface Map')

fig.colorbar(surf, shrink=0.5, aspect=5, label='Velocity [m/s]')

plt.show()
