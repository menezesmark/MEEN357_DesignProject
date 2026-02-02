# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 09:31:10 2026

@author: menez
"""

import numpy as np
import math as mt
import matplotlib.pyplot as plt




def get_mass(rover):
   ans = 6(rover['wheel'['mass']] + rover['speed_reducer'['mass']] + rover['motor'['mass']]) + rover['chassis'['mass']] + rover['science_payload'['mass']] + rover['power_subsys'['mass']]
   return ans
   

def get_gear_ratio(speed_reducer): #return speed ratio
    ng = (speed_reducer['diam_gear']/speed_reducer['diam_pinion'])**2
    return ng    
    
    
def tau_dcmotor(omega, motor): #return motor shaft torque in rad/s
    tau = motor['torque_noload'] - ((motor['torque_stall'] - motor['torque_noload'])/motor['speed_noload'])*omega
    return tau
    
    

def F_drive(omega, rover): #return drive force Fd
    Fd = (tau_dcmotor() * get_gear_ratio())/rover['wheel'['radius']]#idk how to get motor into thes functions
    return Fd
    
    
def F_gravity(terrain_angle, rover, planet):
    if (terrain_angle >  75) or (terrain_angle < -75):
        raise Exception('angle is to large, must be between -75,75')
    Fgt = get_mass(rover) * planet['g'] * np.sin(np.deg2rad(terrain_angle))
    return Fgt
    
    
    
def F_rolling(omega, terrain_angle, rover, planet, Crr):
    Frr_simple = Crr * get_mass(rover) * planet['g'] * np.cos(np.deg2rad(terrain_angle))
    Vrover = rover['wheel'['radius']] * omega
    Frr = mt.erf(40*Vrover)*Frr_simple
    return Frr
    
    
def F_net(omega, terrain_angle, rover, planet, Crr): #return array of forces??
    Fslope = F_drive(omega, rover) - F_rolling(omega, terrain_angle, rover, planet, Crr) - F_gravity(terrain_angle, rover, planet)
    return Fslope
    
    
    
    
    