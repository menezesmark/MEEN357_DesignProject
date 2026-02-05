# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 09:31:10 2026

@author: menez
"""

import numpy as np
from scipy import special
import matplotlib.pyplot as plt



def get_mass(rover):
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
    
    if type(speed_reducer) is not dict:
        raise Exception('speed_reducer must be a dictionary')

    if not speed_reducer['type'] == 'reverted':
        raise Exception("some type of error")
    
    ng = (speed_reducer['diam_gear']/speed_reducer['diam_pinion'])**2
    return ng    


def tau_dcmotor(omega, motor): #return motor shaft torque in rad/s
    if type(motor) is not dict:
        raise Exception('motor must be a dictionary')

    if not (np.isscalar(omega) or (isinstance(omega, np.ndarray) and omega.ndim == 1)):
        raise Exception('omega must be a scalar or vector')    

    tau = np.maximum(0, motor['torque_stall'] * (1 - omega / motor['speed_noload']))
    return tau


def F_drive(omega, rover): #return drive force Fd
    
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


# def F_rolling(omega, terrain_angle, rover, planet, Crr): #return rolling res

def F_rolling(omega, terrain_angle, rover, planet, Crr):

    # --- Validation ---
    if not (np.isscalar(omega) or (isinstance(omega, np.ndarray) and omega.ndim == 1)):
        raise Exception("omega must be a scalar or 1D numpy array")

    if not (np.isscalar(terrain_angle) or (isinstance(terrain_angle, np.ndarray) and terrain_angle.ndim == 1)):
        raise Exception("terrain_angle must be a scalar or 1D numpy array")

    if np.any(terrain_angle > 75) or np.any(terrain_angle < -75):
        raise Exception("terrain_angle must be between -75 and 75 degrees")

    if not isinstance(rover, dict):
        raise Exception("rover must be a dictionary")

    if not isinstance(planet, dict):
        raise Exception("planet must be a dictionary")

    if not (np.isscalar(Crr) and Crr > 0):
        raise Exception("Crr must be a positive scalar")

    # --- Required call (even though not used) ---
    _ = get_gear_ratio(rover['wheel_assembly']['speed_reducer'])

    # --- Physics ---
    Fn = get_mass(rover) * planet['g'] * np.cos(np.deg2rad(terrain_angle))
    Frr_simple = Crr * Fn

    Vrover = rover['wheel_assembly']['wheel']['radius'] * get_gear_ratio(rover['wheel_assembly']['speed_reducer']) * omega

    # Rolling resistance opposes motion
    Frr = - special.erf(40 * Vrover) * Frr_simple

    return Frr


    # if not (np.isscalar(omega) or (isinstance(omega, np.ndarray) and omega.ndim == 1 and omega.ndim == terrain_angle.ndim)):
    #     raise Exception('omega must be a scalar or vector')

    # if not (np.isscalar(terrain_angle) or (isinstance(terrain_angle, np.ndarray) and terrain_angle.ndim == 1)):
    #     raise Exception('terrain_angle must be a scalar or vector')
    
    # if np.any(terrain_angle >  75) or np.any(terrain_angle < -75):
    #     raise Exception('angle is out of range, must be between -75,75')
    
    # if type(rover) is not dict:
    #     raise Exception('rover must be a dictionary')

    # if type(planet) is not dict:
    #     raise Exception('planet must be a dictionary')
    
    # if not np.isscalar(Crr) and Crr > 0:
    #     raise Exception('Crr must be a positive scalar')
    
    # Fn = get_mass(rover) * planet['g'] * np.cos(np.deg2rad(terrain_angle))
    # Frr_simple = Crr * Fn
    # Vrover = rover['wheel_assembly']['wheel']['radius'] * omega
    
    # Frr =  - special.erf(40 * Vrover) * Frr_simple
    # return Frr


def F_net(omega, terrain_angle, rover, planet, Crr): #return array of forces??
    Fslope = F_drive(omega, rover) - F_rolling(omega, terrain_angle, rover, planet, Crr) - F_gravity(terrain_angle, rover, planet)
    return Fslope

