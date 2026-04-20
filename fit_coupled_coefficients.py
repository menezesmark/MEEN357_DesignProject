#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Note: Code was created by ChatGPT due to time constraints

'''
This file finds relationships between the parachute size and fuel mass, 
compared to the mass of the rest of the rover.
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from subfunctions_Phase4 import *
from define_experiment import *


# ============================================================
# USER SETTINGS
# ============================================================

# Baseline design vector order:
# [parachute diameter, wheel radius, chassis mass, gear diameter, fuel mass]
BASE_X = np.array([19.0, 0.7, 550.0, 0.09, 250.0], dtype=float)

# Bounds used in the optimizer
PARACHUTE_MIN_BOUND = 14.0
PARACHUTE_MAX_BOUND = 19.0
FUEL_MIN_BOUND = 100.0
FUEL_MAX_BOUND = 290.0

# Search grids
PARACHUTE_GRID = np.linspace(PARACHUTE_MIN_BOUND, PARACHUTE_MAX_BOUND, 26)
FUEL_GRID = np.linspace(FUEL_MIN_BOUND, FUEL_MAX_BOUND, 25)

# Chassis masses to sweep over for calibration
CHASSIS_SWEEP = np.linspace(250.0, 800.0, 16)

# For the fuel study, use the parachute fit plus this small margin
FUEL_STUDY_PARACHUTE_MARGIN = 0.25   # meters

# Simulation settings
TMAX = 5000
ITER_INFO = False   # False keeps output cleaner

# Plot switch
MAKE_PLOTS = True


# ============================================================
# MASS DRIVER CHOICES
# ============================================================

def parachute_mass_driver(chassis_mass, baseline_fuel_mass):
    """
    Mass driver used to fit required parachute size.
    Matches your current first-pass coupled-constraint idea:
    payload-like mass = chassis mass + fuel mass.
    """
    return chassis_mass + baseline_fuel_mass


def fuel_mass_driver(chassis_mass):
    """
    Mass driver used to fit required fuel mass.
    Using chassis mass only avoids circularity.
    """
    return chassis_mass


# ============================================================
# SYSTEM BUILD / RESET
# ============================================================

def make_base_system():
    """
    Build the same baseline Phase 4 system used by the optimizer.
    """
    planet = define_planet()
    edl_system = define_edl_system()
    mission_events = define_mission_events()

    edl_system = define_chassis(edl_system, 'carbon')
    edl_system = define_motor(edl_system, 'base')
    edl_system = define_batt_pack(edl_system, 'PbAcid-1', 10)

    # Match optimizer initial conditions
    edl_system['altitude'] = 11000
    edl_system['velocity'] = -587
    edl_system['parachute']['deployed'] = True
    edl_system['parachute']['ejected'] = False
    edl_system['rover']['on_ground'] = False

    experiment, end_event = experiment1()
    return planet, edl_system, mission_events, experiment, end_event


def apply_design(edl_system, x):
    """
    Apply a design vector to a freshly reset EDL system.
    """
    edl_system = redefine_edl_system(edl_system)

    edl_system['parachute']['diameter'] = float(x[0])
    edl_system['rover']['wheel_assembly']['wheel']['radius'] = float(x[1])
    edl_system['rover']['chassis']['mass'] = float(x[2])
    edl_system['rover']['wheel_assembly']['speed_reducer']['diam_gear'] = float(x[3])
    edl_system['rocket']['initial_fuel_mass'] = float(x[4])
    edl_system['rocket']['fuel_mass'] = float(x[4])

    return edl_system


# ============================================================
# EDL EVALUATION
# ============================================================

def eval_edl_only(x, iter_info=False):
    """
    Run EDL only and judge success using touchdown conditions.
    """
    planet, edl_system, mission_events, experiment, end_event = make_base_system()
    edl_system = apply_design(edl_system, x)

    try:
        T, Y, edl_out = simulate_edl(edl_system, planet, mission_events, TMAX, iter_info)

        touchdown_speed = edl_out.get('rover_touchdown_speed', np.nan)
        rover_on_ground = edl_out['rover'].get('on_ground', False)

        final_altitude = float(Y[1, -1])
        danger_altitude = edl_out['sky_crane']['danger_altitude']
        danger_speed = edl_out['sky_crane']['danger_speed']

        success = (
            rover_on_ground
            and np.isfinite(touchdown_speed)
            and final_altitude >= danger_altitude
            and abs(touchdown_speed) <= abs(danger_speed)
        )

        return {
            'success': success,
            'touchdown_speed': touchdown_speed,
            'final_time': float(T[-1]),
            'final_altitude': final_altitude,
            'edl_system': edl_out,
        }

    except Exception:
        return {
            'success': False,
            'touchdown_speed': np.nan,
            'final_time': np.nan,
            'final_altitude': np.nan,
            'edl_system': None,
        }


# ============================================================
# THRESHOLD SEARCHES
# ============================================================

def find_min_safe_parachute(base_x, chassis_mass, diam_grid=None):
    """
    For a given chassis mass, find the minimum parachute diameter that gives
    safe EDL touchdown.
    """
    if diam_grid is None:
        diam_grid = PARACHUTE_GRID

    feasible = []

    for diam in diam_grid:
        x = np.array(base_x, dtype=float)
        x[0] = diam
        x[2] = chassis_mass

        result = eval_edl_only(x, iter_info=ITER_INFO)
        if result['success']:
            feasible.append(diam)

    if not feasible:
        return np.nan

    return float(min(feasible))


def find_min_safe_fuel(base_x, chassis_mass, parachute_diam, fuel_grid=None):
    """
    For a given chassis mass and a chosen working parachute diameter,
    find the minimum fuel mass that gives safe EDL touchdown.
    """
    if fuel_grid is None:
        fuel_grid = FUEL_GRID

    feasible = []

    for fuel_mass in fuel_grid:
        x = np.array(base_x, dtype=float)

        # Use the parachute chosen for this fuel study case
        x[0] = parachute_diam

        # Vary chassis mass and fuel mass
        x[2] = chassis_mass
        x[4] = fuel_mass

        result = eval_edl_only(x, iter_info=ITER_INFO)
        if result['success']:
            feasible.append(fuel_mass)

    if not feasible:
        return np.nan

    return float(min(feasible))


# ============================================================
# FITTING MODELS
# ============================================================

def quad_model(m, c2, c1, c0):
    return c2 * m**2 + c1 * m + c0


def fit_quadratic(xdata, ydata):
    popt, _ = curve_fit(quad_model, xdata, ydata)
    return popt


def fuel_study_parachute_for_chassis(chassis_mass, base_x, parachute_coeff):
    """
    Use the parachute fit to choose a conservative parachute diameter for
    the fuel study at a given chassis mass.

    Returns np.nan if the required parachute exceeds the allowed range.
    """
    baseline_fuel = float(base_x[4])
    mass_driver = parachute_mass_driver(chassis_mass, baseline_fuel)

    required_parachute = quad_model(mass_driver, *parachute_coeff)
    study_parachute = required_parachute + FUEL_STUDY_PARACHUTE_MARGIN

    if study_parachute < PARACHUTE_MIN_BOUND:
        study_parachute = PARACHUTE_MIN_BOUND

    if study_parachute > PARACHUTE_MAX_BOUND:
        return np.nan

    return float(study_parachute)


# ============================================================
# MAIN ANALYSES
# ============================================================

def run_parachute_study(base_x):
    mass_points = []
    parachute_points = []

    baseline_fuel = float(base_x[4])

    print("\n=== PARACHUTE COEFFICIENT STUDY ===")
    for chassis_mass in CHASSIS_SWEEP:
        required_diam = find_min_safe_parachute(base_x, chassis_mass)

        if np.isfinite(required_diam):
            m_driver = parachute_mass_driver(chassis_mass, baseline_fuel)
            mass_points.append(m_driver)
            parachute_points.append(required_diam)
            print(
                f"chassis_mass={chassis_mass:8.3f} kg | "
                f"mass_driver={m_driver:8.3f} | "
                f"required_parachute={required_diam:8.3f} m"
            )
        else:
            print(
                f"chassis_mass={chassis_mass:8.3f} kg | "
                f"no safe parachute found in search grid"
            )

    mass_points = np.array(mass_points, dtype=float)
    parachute_points = np.array(parachute_points, dtype=float)

    if len(mass_points) < 3:
        raise RuntimeError("Not enough valid parachute data points to fit coefficients.")

    coeff = fit_quadratic(mass_points, parachute_points)
    mass_range = [float(np.min(mass_points)), float(np.max(mass_points))]

    return {
        'mass_points': mass_points,
        'response_points': parachute_points,
        'coeff': coeff,
        'mass_range': mass_range,
    }


def run_fuel_study(base_x, parachute_result):
    mass_points = []
    fuel_points = []

    print("\n=== FUEL COEFFICIENT STUDY ===")

    for chassis_mass in CHASSIS_SWEEP:
        fuel_study_parachute = fuel_study_parachute_for_chassis(
            chassis_mass,
            base_x,
            parachute_result['coeff']
        )

        if not np.isfinite(fuel_study_parachute):
            print(
                f"chassis_mass={chassis_mass:8.3f} kg | "
                f"required parachute exceeds {PARACHUTE_MAX_BOUND:.1f} m, skipping"
            )
            continue

        required_fuel = find_min_safe_fuel(
            base_x,
            chassis_mass,
            fuel_study_parachute
        )

        if np.isfinite(required_fuel):
            m_driver = fuel_mass_driver(chassis_mass)
            mass_points.append(m_driver)
            fuel_points.append(required_fuel)
            print(
                f"chassis_mass={chassis_mass:8.3f} kg | "
                f"fuel-study parachute={fuel_study_parachute:6.3f} m | "
                f"mass_driver={m_driver:8.3f} | "
                f"required_fuel={required_fuel:8.3f} kg"
            )
        else:
            print(
                f"chassis_mass={chassis_mass:8.3f} kg | "
                f"fuel-study parachute={fuel_study_parachute:6.3f} m | "
                f"no safe fuel mass found in search grid"
            )

    mass_points = np.array(mass_points, dtype=float)
    fuel_points = np.array(fuel_points, dtype=float)

    if len(mass_points) < 3:
        raise RuntimeError("Not enough valid fuel data points to fit coefficients.")

    coeff = fit_quadratic(mass_points, fuel_points)
    mass_range = [float(np.min(mass_points)), float(np.max(mass_points))]

    return {
        'mass_points': mass_points,
        'response_points': fuel_points,
        'coeff': coeff,
        'mass_range': mass_range,
    }


# ============================================================
# OUTPUT / PLOTS
# ============================================================

def print_results(parachute_result, fuel_result):
    print("\n" + "=" * 60)
    print("COPY THESE INTO opt_edl_sys.py")
    print("=" * 60)

    print(f"PARACHUTE_COEFF = {list(parachute_result['coeff'])}")
    print(f"PARACHUTE_MASS_RANGE = {parachute_result['mass_range']}")
    print()
    print(f"FUEL_COEFF = {list(fuel_result['coeff'])}")
    print(f"FUEL_MASS_RANGE = {fuel_result['mass_range']}")

    print("\nRecommended driver logic in opt_edl_sys.py:")
    print("parachute_mass_driver(x, edl_system) -> chassis_mass + fuel_mass")
    print("fuel_mass_driver(x, edl_system) -> chassis_mass")


def make_plots(parachute_result, fuel_result):
    plt.figure()
    plt.plot(
        parachute_result['mass_points'],
        parachute_result['response_points'],
        'o',
        label='Raw points'
    )
    xfit = np.linspace(
        np.min(parachute_result['mass_points']),
        np.max(parachute_result['mass_points']),
        200
    )
    yfit = quad_model(xfit, *parachute_result['coeff'])
    plt.plot(xfit, yfit, '-', label='Quadratic fit')
    plt.xlabel('Parachute mass driver')
    plt.ylabel('Required parachute diameter [m]')
    plt.title('Coupled Parachute Constraint Fit')
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(
        fuel_result['mass_points'],
        fuel_result['response_points'],
        'o',
        label='Raw points'
    )
    xfit = np.linspace(
        np.min(fuel_result['mass_points']),
        np.max(fuel_result['mass_points']),
        200
    )
    yfit = quad_model(xfit, *fuel_result['coeff'])
    plt.plot(xfit, yfit, '-', label='Quadratic fit')
    plt.xlabel('Fuel mass driver')
    plt.ylabel('Required fuel mass [kg]')
    plt.title(f'Coupled Fuel Constraint Fit (Parachute from fitted threshold + {FUEL_STUDY_PARACHUTE_MARGIN:.2f} m margin)')
    plt.grid(True)
    plt.legend()

    plt.show()


# ============================================================
# MAIN
# ============================================================

def main():
    parachute_result = run_parachute_study(BASE_X)
    fuel_result = run_fuel_study(BASE_X, parachute_result)

    print_results(parachute_result, fuel_result)

    if MAKE_PLOTS:
        make_plots(parachute_result, fuel_result)


if __name__ == "__main__":
    main()

