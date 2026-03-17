import numpy as np
import matplotlib.pyplot as plt
from subfunctions import *
import scipy.interpolate as inp

alpha_dist = experiment['alpha_dist']
alpha_deg = experiment['alpha_deg']

alpha_fun = inp.interp1d(alpha_dist, alpha_deg, kind = 'cubic', fill_value= 'extrapolate') #fit the cubic spline

x_eval = np.linspace(np.min(alpha_dist), np.max(alpha_dist), 100)
alpha_eval = alpha_fun(x_eval)

plt.figure()
plt.plot(alpha_dist, alpha_deg, '*')
plt.plot(x_eval, alpha_eval, '-', label='Interpolation')


plt.xlabel('Distance [m]')
plt.ylabel('Angle [degree]')
plt.title('Terrain Angle vs. Distance')
plt.legend()
plt.show()
