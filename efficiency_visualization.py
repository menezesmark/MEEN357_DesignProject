# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 12:09:44 2026

@author: Mark
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as inp
from subfunctions import *


motor = rover['wheel_assembly']['motor']

effcy_tau = np.array(motor['effcy_tau'])
effcy = np.array(motor['effcy'])


effcy_fun = inp.interp1d(effcy_tau,effcy,kind='cubic',fill_value='extrapolate')

x_eval = np.linspace(np.min(effcy_tau), np.max(effcy_tau), 100)
effcy_eval = effcy_fun(x_eval)

plt.figure()
plt.plot(effcy_tau, effcy, '*')
plt.plot(x_eval, effcy_eval, '-', label='Interpolation')
plt.xlabel('Torque [Nm]')
plt.ylabel('Efficiency')
plt.title('Motor Efficiency vs. Torque')

plt.legend()
plt.grid(True)
plt.show()
