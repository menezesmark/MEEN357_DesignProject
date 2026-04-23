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
                  'alpha_deg' : np.array([35, 35, 10, 0, \
                                        -10, 0, 5, 0, \
                                        5, -4.151, 4.042]),
                  'Crr' : 0.1}
    
    
    # Below are default values for example only:
    end_event = {'max_distance' : 1000,
                 'max_time' : 5000,
                 'min_velocity' : 0.01}
    
    '''plot the terrain shape to visualize it for testing'''
    
    
    return experiment, end_event

def experiment_configurable(
    n_points=11,
    total_distance=1000.0,
    crr=0.1,
    max_time=5000,
    min_velocity=0.01,
    initial_conditions=np.array([0.3125, 0.0]),
    time_range=np.array([0, 20000]),

    # master switches
    use_random_angles=True,
    use_smoothing=True,
    use_target_avg_angle=False,
    use_target_height_gain=False,
    randomize_height_gain=False,
    use_angle_clipping=True,

    # angle generation settings
    base_angle_deg=2.0,
    noise_std_deg=3.0,
    angle_min_deg=-8.0,
    angle_max_deg=12.0,
    smooth_passes=2,

    # optional average-angle target
    target_avg_angle_deg=2.0,

    # optional net height gain target
    target_height_gain=20.0,
    gain_min=-20.0,
    gain_max=60.0,

    # random seed
    seed=None
):
    """
    Configurable terrain generator for Phase 4 rover testing.

    Returns experiment, end_event in the same format as experiment1().

    Main switches:
    ----------------
    use_random_angles      : if False, all points start from base_angle_deg
    use_smoothing          : if True, smooth the terrain profile
    use_target_avg_angle   : if True, shift terrain to match target_avg_angle_deg
    use_target_height_gain : if True, shift terrain to approximately match target_height_gain
    randomize_height_gain  : if True, ignore target_height_gain and draw one uniformly
                             from [gain_min, gain_max]
    use_angle_clipping     : if True, clip final angles to [angle_min_deg, angle_max_deg]
    """

    if n_points < 2:
        raise Exception('n_points must be at least 2')

    rng = np.random.default_rng(seed)

    alpha_dist = np.linspace(0.0, total_distance, n_points)

    # --------------------------------------------------------
    # 1. Build initial angle profile
    # --------------------------------------------------------
    if use_random_angles:
        alpha_deg = base_angle_deg + rng.normal(0.0, noise_std_deg, n_points)
    else:
        alpha_deg = np.full(n_points, base_angle_deg, dtype=float)

    # --------------------------------------------------------
    # 2. Smooth if requested
    # --------------------------------------------------------
    if use_smoothing:
        for _ in range(smooth_passes):
            alpha_new = alpha_deg.copy()
            for i in range(1, n_points - 1):
                alpha_new[i] = (
                    0.25 * alpha_deg[i - 1]
                    + 0.50 * alpha_deg[i]
                    + 0.25 * alpha_deg[i + 1]
                )
            alpha_deg = alpha_new

    # --------------------------------------------------------
    # 3. Apply average-angle target if requested
    # --------------------------------------------------------
    if use_target_avg_angle:
        alpha_deg = alpha_deg - np.mean(alpha_deg) + target_avg_angle_deg

    # --------------------------------------------------------
    # 4. Apply net height gain target if requested
    # --------------------------------------------------------
    if use_target_height_gain:
        if randomize_height_gain:
            target_height_gain_used = rng.uniform(gain_min, gain_max)
        else:
            target_height_gain_used = target_height_gain

        # Approximate gain relation:
        # gain ~= total_distance * tan(mean_angle)
        mean_angle_rad = np.arctan2(target_height_gain_used, total_distance)
        mean_angle_deg = np.rad2deg(mean_angle_rad)

        alpha_deg = alpha_deg - np.mean(alpha_deg) + mean_angle_deg

    # --------------------------------------------------------
    # 5. Clip if requested
    # --------------------------------------------------------
    if use_angle_clipping:
        alpha_deg = np.clip(alpha_deg, angle_min_deg, angle_max_deg)

    # --------------------------------------------------------
    # 6. Build Phase 4 output dictionaries
    # --------------------------------------------------------
    experiment = {
        'time_range': np.array(time_range, dtype=float),
        'initial_conditions': np.array(initial_conditions, dtype=float),
        'alpha_dist': np.array(alpha_dist, dtype=float),
        'alpha_deg': np.array(alpha_deg, dtype=float),
        'Crr': float(crr)
    }

    end_event = {
        'max_distance': float(total_distance),
        'max_time': float(max_time),
        'min_velocity': float(min_velocity)
    }

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

# terrain_stats_plots()
