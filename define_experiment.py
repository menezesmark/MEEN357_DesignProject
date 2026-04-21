"""###########################################################################
#   This file initializes the experiment and end_event structures for 
#   MEEN 357 project phase 4.
#
#   Created by: MEEN 357 Simulation Team
###########################################################################"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def experiment1():
    
    experiment = {'time_range' : np.array([0,20000]),
                  'initial_conditions' : np.array([0.3125,0]),
                  'alpha_dist' : np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]),
                  'alpha_deg' : np.array([11.509, 2.032, 7.182, 0, \
                                        -30, 10.981, 35, -0.184, \
                                        0.714, 4.151, 4.042]),
                  'Crr' : 0.1}
    
    
    # Below are default values for example only:
    end_event = {'max_distance' : 1000,
                 'max_time' : 5000,
                 'min_velocity' : 0.01}
    
    '''plot the terrain shape to visualize it for testing'''
    
    
    return experiment, end_event

def terrain_stats_plots():
    experiment, end_event = experiment1()
    
    alpha_dist = experiment['alpha_dist']
    alpha_deg = experiment['alpha_deg']
    
    # Same interpolation used in rover_dynamics
    alpha_fun = interp1d(alpha_dist, alpha_deg, kind='cubic', fill_value='extrapolate')
    
    # Fine grid for smooth plots
    x_plot = np.linspace(alpha_dist[0], alpha_dist[-1], 1000)
    alpha_plot_deg = alpha_fun(x_plot)
    alpha_plot_rad = np.deg2rad(alpha_plot_deg)
    
    # Terrain slope and integrated elevation on fine grid
    slope_plot = np.tan(alpha_plot_rad)
    
    y_plot = np.zeros_like(x_plot)
    for i in range(1, len(x_plot)):
        dx = x_plot[i] - x_plot[i - 1]
        y_plot[i] = y_plot[i - 1] + 0.5 * (slope_plot[i] + slope_plot[i - 1]) * dx
    
    # Compute terrain elevation at the original source points too
    alpha_source_rad = np.deg2rad(alpha_deg)
    slope_source = np.tan(alpha_source_rad)
    
    y_source = np.zeros_like(alpha_dist, dtype=float)
    for i in range(1, len(alpha_dist)):
        dx = alpha_dist[i] - alpha_dist[i - 1]
        y_source[i] = y_source[i - 1] + 0.5 * (slope_source[i] + slope_source[i - 1]) * dx
    
    fig, axs = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    
    # Angle profile
    axs[0].plot(alpha_dist, alpha_deg, 'o', label='Given angle points')
    axs[0].plot(x_plot, alpha_plot_deg, '-', label='Cubic interpolation')
    axs[0].set_ylabel('Terrain angle [deg]')
    axs[0].set_title('Terrain Angle Profile')
    axs[0].grid(True)
    axs[0].legend()
    
    # Terrain shape profile
    axs[1].plot(x_plot, y_plot, '-', label='Integrated terrain profile')
    axs[1].plot(alpha_dist, y_source, 'o', label='Source-point elevations')
    axs[1].set_xlabel('Distance along path [m]')
    axs[1].set_ylabel('Relative elevation [m]')
    axs[1].set_title('Terrain Elevation Profile')
    axs[1].grid(True)
    axs[1].legend()
    
    plt.tight_layout()
    plt.show()
    return

terrain_stats_plots()
