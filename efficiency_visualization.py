# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 12:09:44 2026

@author: Mark
"""

import numpy as np
import matplotlib.pyplot as plt
from subfunctions import *
import scipy.interpolate as inp

effcy_tau = motor['effcy_tau']
effcy = motor['effcy']



effcy_fun = inp.interp1d(effcy_tau, effcy, kind = 'cubic') # fit the cubic spline

x_eval = np.linspace(motor['torque_noload'], motor['torque_stall'], 100)
effcy_eval = effcy_fun(x_eval)
