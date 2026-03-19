# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 09:31:10 2026

@author: menez
"""

import numpy as np
import scipy as sci
import scipy.interpolate as inp


planet = {'g': 3.72}

power_subsys = {'mass': 90}

science_payload = {'mass': 75}

chassis = {'mass': 659}

motor = {
    'torque_stall': 170, 
    'torque_noload': 0, 
    'speed_noload': 3.80, 
    'mass': 5.0,
    'effcy_tau': [0, 10, 20, 40, 70, 165],
    'effcy': [0, 0.60, 0.78, 0.73, 0.52, 0.04]
}

speed_reducer = {
    'type': "reverted", 
    'diam_pinion': 0.04, 
    'diam_gear': 0.07, 
    'mass': 1.5
}

wheel = {
    'radius': 0.30, 
    'mass': 1.0
}

wheel_assembly = {
    'wheel': wheel, 
    'speed_reducer': speed_reducer, 
    'motor': motor
}

rover = {
    'wheel_assembly': wheel_assembly, 
    'chassis': chassis, 
    'science_payload': science_payload, 
    'power_subsys': power_subsys
}

experiment = {'time_range' : np.array([0,20000]),
                  'initial_conditions' : np.array([0.325,0]),
                  'alpha_dist' : np.array([0, 100, 200, 300, 400, 500, 600, \
                                           700, 800, 900, 1000]),
                  'alpha_deg' : np.array([2.032, 11.509, 2.478, 7.182, \
                                        5.511, 10.981, 5.601, -0.184, \
                                        0.714, 4.151, 4.042]),
                  'Crr' : 0.1}

end_event = {'max_distance' : 50,
                 'max_time' : 5000,
                 'min_velocity' : 0.01}


def get_mass(rover):

    '''
    Calcs total mass of the rover from info in rover dict
    Input:  rover
    Returns:  total_mass, 
    '''

    if type(rover) is not dict:
        raise Exception('rover must be a dictionary')
        
    wheel_mass = rover['wheel_assembly']['wheel']['mass']
    speed_reducer_mass = rover['wheel_assembly']['speed_reducer']['mass']
    motor_mass = rover['wheel_assembly']['motor']['mass']
    chassis_mass = rover['chassis']['mass']
    payload_mass = rover['science_payload']['mass']
    power_mass = rover['power_subsys']['mass']

    total_mass = 6 * (wheel_mass + speed_reducer_mass + motor_mass) + chassis_mass + payload_mass + power_mass
    
    return total_mass


def get_gear_ratio(speed_reducer): #return speed ratio
    '''
    Returns reduction ratio for speed reducer from speed_reducer dict
    Input:  speed_reducer
    Returns:  ng
    '''
    if type(speed_reducer) is not dict:
        raise Exception('speed_reducer must be a dictionary')

    if not speed_reducer['type'] == 'reverted':
        raise Exception("some type of error")
    
    ng = (speed_reducer['diam_gear']/speed_reducer['diam_pinion'])**2
    
    return ng    


def tau_dcmotor(omega, motor): #return motor shaft torque in rad/s
    '''
    Returns motor shaft torque given shaft speed + dict with motor specs
    Input:  omega and motor
    Returns: tau
    '''
    if type(motor) is not dict:
        raise Exception('motor must be a dictionary')

    if not (np.isscalar(omega) or (isinstance(omega, np.ndarray) and omega.ndim == 1)):
        raise Exception('omega must be a scalar or vector')    

    tau = np.maximum(0, motor['torque_stall'] * (1 - omega / motor['speed_noload']))
    
    return tau


def F_drive(omega, rover): #return drive force Fd
    '''
    Returns force applied by the drive system given drive info system and shaft speed
    Input:  omega and rover
    Returns:  Fd
    '''
    if type(rover) is not dict:
        raise Exception('rover must be a dictionary')

    if not (np.isscalar(omega) or (isinstance(omega, np.ndarray) and omega.ndim == 1)):
        raise Exception('omega must be scalar or vector')

    motor = rover['wheel_assembly']['motor']
    speed_reducer = rover['wheel_assembly']['speed_reducer']
    wheel = rover['wheel_assembly']['wheel']

    tau = tau_dcmotor(omega, motor)
    ng = get_gear_ratio(speed_reducer)

    Fd = 6 * (tau * ng) / wheel['radius']
    
    return Fd


def F_gravity(terrain_angle, rover, planet): #still having some errors w/ validation
    
    if not (np.isscalar(terrain_angle) or (isinstance(terrain_angle, np.ndarray) and terrain_angle.ndim == 1)):
        raise Exception('terrain_angle must be a scalar or vector')
    
    if np.any(terrain_angle >  75) or np.any(terrain_angle < -75):
        raise Exception('angle is out of range, must be between -75,75')
    
    if type(rover) is not dict:
        raise Exception('rover must be a dictionary')

    if type(planet) is not dict:
        raise Exception('planet must be a dictionary')
    
    
    Fgt = - get_mass(rover) * planet['g'] * np.sin(np.deg2rad(terrain_angle))
    
    return Fgt


def F_rolling(omega, terrain_angle, rover, planet, Crr): #return rolling res

    if not (np.isscalar(omega) or (isinstance(omega, np.ndarray) and omega.ndim == 1)):
        raise Exception("omega must be a scalar or 1D numpy array")

    if not (np.isscalar(terrain_angle) or (isinstance(terrain_angle, np.ndarray) and terrain_angle.ndim == 1)):
        raise Exception("terrain_angle must be a scalar or 1D numpy array")

    if isinstance(omega, np.ndarray) and isinstance(terrain_angle, np.ndarray):
        if omega.size != terrain_angle.size:
            raise Exception("omega and terrain_angle must be the same size")

    if np.any(terrain_angle > 75) or np.any(terrain_angle < -75):
        raise Exception("terrain_angle must be between -75 and 75 degrees")

    if not isinstance(rover, dict):
        raise Exception("rover must be a dictionary")

    if not isinstance(planet, dict):
        raise Exception("planet must be a dictionary")

    if not (np.isscalar(Crr) and Crr > 0):
        raise Exception("Crr must be a positive scalar")

    
    Fn = get_mass(rover) * planet['g'] * np.cos(np.deg2rad(terrain_angle))
    Frr_simple = Crr * Fn
    Ng = get_gear_ratio(rover['wheel_assembly']['speed_reducer'])
    Vrover = rover['wheel_assembly']['wheel']['radius'] * omega / Ng

    Frr =  - sci.special.erf(40 * Vrover) * Frr_simple
    
    return Frr


def F_net(omega, terrain_angle, rover, planet, Crr): #return array of forces??
    
    Fslope = F_drive(omega, rover) + F_rolling(omega, terrain_angle, rover, planet, Crr) + F_gravity(terrain_angle, rover, planet)
    
    return Fslope


def motorW(v, rover): #calc shaft speed from rover velo and characteristics w = motorW
    '''doc_string output for help()'''
    
    if not (np.isscalar(v) or (isinstance(v, np.ndarray) and v.ndim == 1)):
        raise Exception("v must be a scalar or 1D numpy array")
        
    if not isinstance(rover, dict):
        raise Exception("rover must be a dictionary")
    
    # total rotation ratio from ground speed in x,y,(z?)
    # to tangential velocity into wheel speed
    # into the gearbox, all the way to the motor
    
    r_wheel = rover['wheel_assembly']['wheel']['radius']
    Ng = get_gear_ratio(rover['wheel_assembly']['speed_reducer'])
    
    # assuming the wheels are rolling w/o slipping
    W_motor = (v * Ng)/r_wheel
    
    return W_motor


def rover_dynamics(t, y, rover, planet, experiment): #deriv of [velo, pos] -> state vector. = dydt
    # INCOMPLETE, NEEDS WORK
    '''doc_string of rover_dynamics'''
    
    if not (np.isscalar(t)):
        raise Exception("t must be a scalar")
    
    if not (isinstance(y, np.ndarray) and y.ndim == 1 and (np.size(y) == 2)):
        raise Exception("y must be numpy array of two elements, velocity (m/s) and position (m)")
    
    if not isinstance(rover, dict):
        raise Exception("rover must be a dictionary")

    if not isinstance(planet, dict):
        raise Exception("planet must be a dictionary")
    
    if not isinstance(experiment, dict):
        raise Exception("experiment must be a dictionary")
    
    v = float(y[0])
    x = float(y[1])
    
    m_net = get_mass(rover)
    alpha_fun = inp.interp1d(experiment['alpha_dist'], 
                             experiment['alpha_deg'], 
                             kind='cubic',
                             bounds_error=False,
                             fill_value=(experiment['alpha_dist'][0], experiment['alpha_deg'][0]))
    
    terrain_angle = float(alpha_fun(x))
    
    omega = motorW(v, rover)
    Fnet = float(F_net(omega, terrain_angle, rover, planet, experiment['Crr']))
    
    accel = float(Fnet / m_net)
    vel_deriv = float(v)
    
    dydt = np.array([accel, vel_deriv])
    
    return dydt


def mechpower(v, rover): # calc instant mech pwr from single motor at given velo profile. = P
    # a
    '''documentation for mechpower'''
    
    if not (np.isscalar(v) or (isinstance(v, np.ndarray) and v.ndim == 1)):
        raise Exception("v must be a scalar or 1D numpy array")
        
    if not isinstance(rover, dict):
        raise Exception("rover must be a dictionary")
    
    w = motorW(v, rover)
    tau = tau_dcmotor(w, motor)

    P = tau * w
    
    return P


def battenergy(t, v, rover): # calc total energy used over time-velo pair/ = E
    '''documentation for battenergy'''
    
    if not (np.isscalar(t) or (isinstance(t, np.ndarray) and t.ndim == 1)):
        raise Exception("t must be a scalar or 1D numpy array")    
    
    if not (np.isscalar(v) or (isinstance(v, np.ndarray) and v.ndim == 1)):
        raise Exception("v must be a scalar or 1D numpy array")
        
    if not isinstance(rover, dict):
        raise Exception("rover must be a dictionary")
    
    if isinstance(t, np.ndarray) and isinstance(v, np.ndarray):
        if t.size != v.size:
            raise Exception("t and v must be the same size")
    
    motor = rover['wheel_assembly']['motor']
    omega = motorW(v, rover)
    tau = tau_dcmotor(omega, motor)
    P_mech = mechpower(v, rover)

    effcy_tau = np.array(motor['effcy_tau'])
    effcy = np.array(motor['effcy'])

    effcy_fun = inp.interp1d(effcy_tau,
                             effcy,
                             kind='cubic',
                             fill_value='extrapolate')

    
    eta = effcy_fun(tau)
    eta = np.clip(eff, 1e-6, 1.0)
    
    Pbatt = 6*P_mech/eta
    
    E = np.trapz(Pbatt, t)
    
    return float(E)


def simulate_rover(rover, planet, experiment, end_event): # integrates trajectory of rover. = rover
    '''documentation for simulat_rover'''
    if not isinstance(rover, dict):
        raise Exception("rover must be a dictionary")
        
    if not isinstance(planet, dict):
        raise Exception("planet must be a dictionary")
    
    if not isinstance(experiment, dict):
        raise Exception("experiment must be a dictionary")
    
    if not isinstance(end_event, dict):
        raise Exception("end_event must be a dictionary")

    # Event functions
    mission_event = end_of_mission_event(end_event)

    # Solve ODE
    sol = solve_ivp(
        lambda t, y: rover_dynamics(t, y, rover, planet, experiment),
        experiment['time_range'],
        experiment['initial_conditions'],
        events=[event_distance, event_time, event_velocity],
    )

    t = sol.t
    v = sol.y[0]
    x = sol.y[1]

    P = mechpower(v, rover)

    E = battenergy(t, v, rover)

    # Make telem
    rover['telemetry'] = {
        'Time': t,
        'completion_time': t[-1],
        'velocity': v,
        'position': x,
        'distance_traveled': x[-1],
        'max_velocity': np.max(v),
        'average_velocity': np.mean(v),
        'power': P,
        'battery_energy': E,
        'energy_per_distance': E / x[-1] if x[-1] > 0 else np.nan #bc sometimes its doesn't work
    }


    return rover










