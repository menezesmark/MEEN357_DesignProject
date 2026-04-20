#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 16:01:18 2026

@author: wyattmoore
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from define_experiment import experiment1

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

# terrain_stats_plots()