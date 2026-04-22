#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 11:43:10 2026

@author: wyattmoore
"""

"""
repeat_opt_edl.py

Framework for repeated optimization runs using the same conventions and style
as opt_edl_sys.py.

This script is meant to:
    - build one optimization problem with the same setup style as opt_edl
    - choose optimizer by method string
    - accept x0 and optimizer settings from a single settings block
    - run one optimization at a time
    - check feasibility after each optimization
    - evaluate and store successful designs
    - run repeated/batch comparisons over discrete rover architecture choices
    - allow terrain switching by passing different experiment functions
"""

import numpy as np
from subfunctions_Phase4 import *
from define_experiment import *
from scipy.optimize import minimize, differential_evolution
from scipy.optimize import Bounds
from scipy.optimize import NonlinearConstraint
import pickle
import sys
import csv
import os

INPUT_CSV = 'saved_rovers_WM.csv'
INPUT_CSV_INDEX = 4
OUTPUT_CSV = INPUT_CSV


# ============================================================
# SETTINGS BLOCK
# ============================================================

SETTINGS = {
    # architecture choices
    'chassis_type': 'magnesium',
    'motor_type': 'speed',
    'battery_type': 'LiFePO4',
    'battery_modules': 10,

    # terrain / experiment choice
    'experiment_function': experiment1,
    'terrain_plot': False,

    # optimization method
    # options: 'trust-constr', 'SLSQP', 'differential_evolution', 'COBYLA'
    'method': 'differential_evolution',

    # initial guess options
    'use_csv_start': False,
    'csv_start_file': 'saved_rovers_WM.csv',
    'csv_start_row': 0,     # zero-based
    'x0_default': np.array([19.0, 0.70, 550.0, 0.09, 250.0]),

    # main simulation / mission settings
    'tmax': 5000,
    'max_rover_velocity': -1.0,
    'min_strength': 40000.0,
    'max_cost': 7.2e6,

    # bounds in same order as x
    # [parachute diameter, wheel radius, chassis mass, d2, fuel mass]
    'bounds_lb': [15.0, 0.65, 250.0, 0.05, 150.0],
    'bounds_ub': [19.0, 0.70, 800.0, 0.12, 290.0],

    # trust-constr options
    'trust_maxiter': 5,
    'trust_verbose': 3,
    'trust_disp': True,

    # SLSQP options
    'slsqp_maxiter': 5,
    'slsqp_disp': True,

    # differential evolution options
    'de_popsize': 20,
    'de_maxiter': 20,
    'de_disp': True,
    'de_polish': False,
    'de_seed': None,
    'de_workers': 1,

    # COBYLA options
    'cobyla_maxiter': 50,
    'cobyla_disp': True,

    # logging / storage
    'save_results': True,
    'results_csv': 'saved_rovers_WM.csv',
    'save_pickle': False,
    'pickle_file': 'SP26_501team64_repeat.pickle',
    'team_name': 'SixtySevenMinusThree',
    'team_number': 64,

    # batch comparison storage
    'batch_csv': 'temp_batch_results.csv',
    'reset_batch_csv': False,

    # behavior
    'use_callback': False,
    'debug_on_infeasible': True
}


# ============================================================
# DISCRETE OPTION STORAGE FOR BATCH RUNS
# ============================================================

ARCH_OPTIONS = {
    'chassis_type': ['steel', 'magnesium', 'carbon'],
    'motor_type': ['base', 'torque', 'speed'],
    'battery': [
        ('PbAcid-1', 10),
        ('PbAcid-2', 10),
        ('NiMH', 10),
        ('LiFePO4', 10)
    ],
    'experiment_function': [experiment1]
}


# ============================================================
# CSV HELPERS
# ============================================================

def append_result_to_csv(csv_path, row_dict):
    fixed_fields = [
        'method', 'terrain_name', 'motor_tested',
        'chassis_type', 'battery_type', 'battery_modules',
        'status', 'feasible', 'largest_violation',
        'parachute_diameter', 'wheel_radius', 'chassis_mass',
        'gear_diameter', 'fuel_mass_per_rocket',
        'time_edl', 'time_rover', 'time_total',
        'landing_velocity', 'avg_velocity', 'distance_traveled',
        'energy_per_distance', 'total_cost', 'objective_value'
    ]

    row_out = {}
    for key in fixed_fields:
        row_out[key] = row_dict.get(key, '')

    file_exists = os.path.exists(csv_path)

    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fixed_fields)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_out)


def load_x0_from_csv(csv_path, row_index):
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if row_index < 0 or row_index >= len(rows):
        raise IndexError(f'row_index {row_index} out of range for {len(rows)} saved rows')

    row = rows[row_index]

    x0_loaded = np.array([
        float(row['parachute_diameter']),
        float(row['wheel_radius']),
        float(row['chassis_mass']),
        float(row['gear_diameter']),
        float(row['fuel_mass_per_rocket'])
    ], dtype=float)

    return x0_loaded, row


def reset_csv(csv_path):
    if os.path.exists(csv_path):
        os.remove(csv_path)


# ============================================================
# CUSTOM OPTIMIZATION HELPERS (same style as opt_edl)
# ============================================================

PARACHUTE_COEFF = [
    np.float64(1.947251937461125e-06),
    np.float64(0.002157054850622941),
    np.float64(14.84407140656271)
]

PARACHUTE_MASS_RANGE = [500.0, 1013.3333333333333]
LARGE_PENALTY = 1.0e12


def unpack_x(x):
    parachute_diam = x[0]
    wheel_radius = x[1]
    chassis_mass = x[2]
    gear_diam = x[3]
    fuel_mass = x[4]
    return parachute_diam, wheel_radius, chassis_mass, gear_diam, fuel_mass


def parachute_mass_driver(x, edl_system):
    parachute_diam, wheel_radius, chassis_mass, gear_diam, fuel_mass = unpack_x(x)
    return chassis_mass + fuel_mass


def fuel_mass_driver(x, edl_system):
    parachute_diam, wheel_radius, chassis_mass, gear_diam, fuel_mass = unpack_x(x)
    return chassis_mass


def estimated_payload_mass(x, edl_system):
    return parachute_mass_driver(x, edl_system)


def required_parachute_from_mass(payload_mass):
    a2, a1, a0 = PARACHUTE_COEFF
    return a2*payload_mass**2 + a1*payload_mass + a0


def coupled_constraints(x, edl_system):
    parachute_diam, wheel_radius, chassis_mass, gear_diam, fuel_mass = unpack_x(x)
    parachute_mass = parachute_mass_driver(x, edl_system)

    c = []

    c.append(PARACHUTE_MASS_RANGE[0] - parachute_mass)
    c.append(parachute_mass - PARACHUTE_MASS_RANGE[1])

    required_parachute = required_parachute_from_mass(parachute_mass)
    c.append(required_parachute - parachute_diam)

    return np.array(c, dtype=float)


def combined_constraints(x, edl_system, planet, mission_events, tmax,
                         experiment, end_event, min_strength,
                         max_rover_velocity, max_cost, max_batt_energy_per_meter):

    c_existing = constraints_edl_system(
        x, edl_system, planet, mission_events, tmax,
        experiment, end_event, min_strength,
        max_rover_velocity, max_cost, max_batt_energy_per_meter
    )

    c_coupled = coupled_constraints(x, edl_system)

    return np.concatenate((c_existing, c_coupled))


def mvp_check(x, edl_system, planet, mission_events, experiment, end_event):
    parachute_diam, wheel_radius, chassis_mass, gear_diam, fuel_mass = unpack_x(x)

    c_coupled = coupled_constraints(x, edl_system)
    if np.any(c_coupled > 0):
        return False

    if gear_diam >= wheel_radius:
        return False

    if wheel_radius <= 0.0 or parachute_diam <= 0.0 or fuel_mass <= 0.0:
        return False

    payload_mass = estimated_payload_mass(x, edl_system)

    rough_drive_metric = wheel_radius / gear_diam
    if rough_drive_metric < 2.0:
        return False

    if payload_mass > 1200.0:
        return False

    return True


def screened_obj_fun(x, edl_system, planet, mission_events, tmax, experiment, end_event):
    if not mvp_check(x, edl_system, planet, mission_events, experiment, end_event):
        return LARGE_PENALTY

    try:
        return obj_fun_time(x, edl_system, planet, mission_events, tmax,
                            experiment, end_event)
    except Exception:
        return LARGE_PENALTY


def debug_candidate(x, edl_system, planet, mission_events, tmax, experiment, end_event):
    edl_dbg = redefine_edl_system(edl_system)

    edl_dbg['parachute']['diameter'] = float(x[0])
    edl_dbg['rover']['wheel_assembly']['wheel']['radius'] = float(x[1])
    edl_dbg['rover']['chassis']['mass'] = float(x[2])
    edl_dbg['rover']['wheel_assembly']['speed_reducer']['diam_gear'] = float(x[3])
    edl_dbg['rocket']['initial_fuel_mass'] = float(x[4])
    edl_dbg['rocket']['fuel_mass'] = float(x[4])

    print('\n================ DEBUG: LAST CANDIDATE =================')
    print('Candidate x = {}'.format(x))
    print('Parachute diameter         = {:.6f} m'.format(x[0]))
    print('Wheel radius               = {:.6f} m'.format(x[1]))
    print('Chassis mass               = {:.6f} kg'.format(x[2]))
    print('Speed reducer gear diam    = {:.6f} m'.format(x[3]))
    print('Fuel mass per rocket       = {:.6f} kg'.format(x[4]))

    try:
        T_edl, Y_edl, edl_dbg = simulate_edl(edl_dbg, planet, mission_events, tmax, True)

        remaining_total_fuel = Y_edl[2, -1]
        remaining_fuel_per_rocket = remaining_total_fuel / edl_dbg['num_rockets']

        print('\n--- EDL diagnostics ---')
        print('EDL final time             = {:.6f} s'.format(T_edl[-1]))
        print('EDL final altitude         = {:.6f} m'.format(Y_edl[1, -1]))
        print('EDL final velocity         = {:.6f} m/s'.format(Y_edl[0, -1]))
        print('Rover touchdown speed      = {:.6f} m/s'.format(edl_dbg.get('rover_touchdown_speed', np.nan)))
        print('Rover on ground            = {}'.format(edl_dbg['rover'].get('on_ground', None)))
        print('Remaining fuel total       = {:.6f} kg'.format(remaining_total_fuel))
        print('Remaining fuel per rocket  = {:.6f} kg'.format(remaining_fuel_per_rocket))

    except Exception as e:
        print('\n--- EDL diagnostics ---')
        print('EDL rerun failed with exception: {}'.format(e))
        return

    try:
        rover_dbg = simulate_rover(edl_dbg['rover'], planet, experiment, end_event)
        telemetry = rover_dbg['telemetry']

        print('\n--- Rover diagnostics ---')
        print('Completion time            = {:.6f} s'.format(telemetry['completion_time']))
        print('Distance traveled          = {:.6f} m'.format(telemetry['distance_traveled']))
        print('Max velocity               = {:.6f} m/s'.format(telemetry['max_velocity']))
        print('Average velocity           = {:.6f} m/s'.format(telemetry['average_velocity']))
        print('Battery energy used        = {:.6f} J'.format(telemetry['battery_energy']))
        print('Energy per distance        = {:.6f} J/m'.format(telemetry['energy_per_distance']))

    except Exception as e:
        print('\n--- Rover diagnostics ---')
        print('Rover rerun failed with exception: {}'.format(e))

    print('=======================================================\n')


def make_callback(problem):
    state = {'Nfeval': 1}

    def callbackF(Xi):
        if state['Nfeval'] == 1:
            print('Iter        x0         x1        x2        x3         x4           fval        mvp')

        passed = mvp_check(
            Xi,
            problem['edl_system'],
            problem['planet'],
            problem['mission_events'],
            problem['experiment'],
            problem['end_event']
        )
        fval = problem['obj_f'](Xi)

        print('{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}  {5: 3.6f}   {6: 3.6f}   {7}'.format(
            state['Nfeval'], Xi[0], Xi[1], Xi[2], Xi[3], Xi[4], fval, passed
        ))
        state['Nfeval'] += 1

    return callbackF


# ============================================================
# PROBLEM BUILDER
# ============================================================

def problem_builder(settings):
    planet = define_planet()
    edl_system = define_edl_system()
    mission_events = define_mission_events()

    edl_system = define_chassis(edl_system, settings['chassis_type'])
    edl_system = define_motor(edl_system, settings['motor_type'])
    edl_system = define_batt_pack(edl_system, settings['battery_type'], settings['battery_modules'])

    edl_system['altitude'] = 11000
    edl_system['velocity'] = -587
    edl_system['parachute']['deployed'] = True
    edl_system['parachute']['ejected'] = False
    edl_system['rover']['on_ground'] = False

    experiment, end_event = settings['experiment_function']()

    if settings['terrain_plot'] and 'terrain_stats_plots' in globals():
        try:
            terrain_stats_plots()
        except Exception:
            pass

    tmax = settings['tmax']
    max_rover_velocity = settings['max_rover_velocity']
    min_strength = settings['min_strength']
    max_cost = settings['max_cost']
    max_batt_energy_per_meter = edl_system['rover']['power_subsys']['battery']['capacity']/1000

    bounds = Bounds(settings['bounds_lb'], settings['bounds_ub'])

    if settings['use_csv_start']:
        x0, loaded_row = load_x0_from_csv(settings['csv_start_file'], settings['csv_start_row'])
    else:
        x0 = np.array(settings['x0_default'], dtype=float)
        loaded_row = None

    obj_f = lambda x: screened_obj_fun(
        x, edl_system, planet, mission_events, tmax,
        experiment, end_event
    )

    cons_f = lambda x: combined_constraints(
        x, edl_system, planet, mission_events, tmax,
        experiment, end_event, min_strength,
        max_rover_velocity, max_cost, max_batt_energy_per_meter
    )

    nonlinear_constraint = NonlinearConstraint(cons_f, -np.inf, 0)

    ineq_cons = {
        'type': 'ineq',
        'fun': lambda x: -1.0 * cons_f(x)
    }

    problem = {
        'planet': planet,
        'edl_system': edl_system,
        'mission_events': mission_events,
        'experiment': experiment,
        'end_event': end_event,
        'tmax': tmax,
        'bounds': bounds,
        'x0': x0,
        'x0_loaded_row': loaded_row,
        'obj_f': obj_f,
        'cons_f': cons_f,
        'nonlinear_constraint': nonlinear_constraint,
        'ineq_cons': ineq_cons,
        'settings': settings
    }

    if settings['use_callback']:
        problem['callback'] = make_callback(problem)
    else:
        problem['callback'] = None

    return problem


# ============================================================
# OPTIMIZER WRAPPER
# ============================================================

def build_cobyla_constraints(bounds, ineq_cons):
    cons_cobyla = []

    for factor in range(len(bounds.lb)):
        lower = float(bounds.lb[factor])
        upper = float(bounds.ub[factor])

        l = {'type': 'ineq',
             'fun': lambda x, lb=lower, i=factor: x[i] - lb}
        u = {'type': 'ineq',
             'fun': lambda x, ub=upper, i=factor: ub - x[i]}

        cons_cobyla.append(l)
        cons_cobyla.append(u)

    cons_cobyla.append(ineq_cons)
    return cons_cobyla


def run_optimizer(problem, settings):
    method = settings['method']
    obj_f = problem['obj_f']
    bounds = problem['bounds']
    x0 = problem['x0']
    nonlinear_constraint = problem['nonlinear_constraint']
    ineq_cons = problem['ineq_cons']
    callback = problem['callback']

    if method == 'trust-constr':
        options = {
            'maxiter': settings['trust_maxiter'],
            'verbose': settings['trust_verbose'],
            'disp': settings['trust_disp']
        }

        res = minimize(
            obj_f, x0, method='trust-constr',
            constraints=nonlinear_constraint,
            options=options,
            bounds=bounds
        )

    elif method == 'SLSQP':
        options = {
            'maxiter': settings['slsqp_maxiter'],
            'disp': settings['slsqp_disp']
        }

        res = minimize(
            obj_f, x0, method='SLSQP',
            constraints=ineq_cons,
            bounds=bounds,
            options=options,
            callback=callback
        )

    elif method == 'differential_evolution':
        res = differential_evolution(
            obj_f,
            bounds=bounds,
            constraints=nonlinear_constraint,
            popsize=settings['de_popsize'],
            maxiter=settings['de_maxiter'],
            disp=settings['de_disp'],
            polish=settings['de_polish'],
            seed=settings['de_seed'],
            workers=settings['de_workers']
        )

    elif method == 'COBYLA':
        cons_cobyla = build_cobyla_constraints(bounds, ineq_cons)

        options = {
            'maxiter': settings['cobyla_maxiter'],
            'disp': settings['cobyla_disp']
        }

        res = minimize(
            obj_f, x0, method='COBYLA',
            constraints=cons_cobyla,
            options=options
        )

    else:
        raise Exception("unknown optimization method string")

    return res


# ============================================================
# FEASIBILITY CHECK
# ============================================================

def check_feasibility(problem, res):
    c = problem['cons_f'](res.x)
    feasible = np.max(c) <= 0.0

    return {
        'feasible': feasible,
        'constraint_vector': c,
        'largest_violation': float(np.max(c))
    }


# ============================================================
# FINAL EVALUATION / RESULT STORAGE
# ============================================================

def evaluate_best_design(problem, res):
    edl_system = redefine_edl_system(problem['edl_system'])
    planet = problem['planet']
    mission_events = problem['mission_events']
    experiment = problem['experiment']
    end_event = problem['end_event']
    tmax = problem['tmax']
    settings = problem['settings']
    xbest = res.x

    edl_system['parachute']['diameter'] = xbest[0]
    edl_system['rover']['wheel_assembly']['wheel']['radius'] = xbest[1]
    edl_system['rover']['chassis']['mass'] = xbest[2]
    edl_system['rover']['wheel_assembly']['speed_reducer']['diam_gear'] = xbest[3]
    edl_system['rocket']['initial_fuel_mass'] = xbest[4]
    edl_system['rocket']['fuel_mass'] = xbest[4]

    edl_system['team_name'] = settings['team_name']
    edl_system['team_number'] = settings['team_number']

    time_edl_run, _, edl_system = simulate_edl(edl_system, planet, mission_events, tmax, True)
    time_edl = time_edl_run[-1]

    edl_system['rover'] = simulate_rover(edl_system['rover'], planet, experiment, end_event)
    time_rover = edl_system['rover']['telemetry']['completion_time']

    total_time = time_edl + time_rover
    total_cost = get_cost_edl(edl_system)

    result_row = {
        'method': settings['method'],
        'terrain_name': settings['experiment_function'].__name__,
        'parachute_diameter': float(xbest[0]),
        'wheel_radius': float(xbest[1]),
        'chassis_mass': float(xbest[2]),
        'gear_diameter': float(xbest[3]),
        'fuel_mass_per_rocket': float(xbest[4]),
        'time_edl': float(time_edl),
        'time_rover': float(time_rover),
        'time_total': float(total_time),
        'landing_velocity': float(edl_system['rover_touchdown_speed']),
        'avg_velocity': float(edl_system['rover']['telemetry']['average_velocity']),
        'distance_traveled': float(edl_system['rover']['telemetry']['distance_traveled']),
        'energy_per_distance': float(edl_system['rover']['telemetry']['energy_per_distance']),
        'total_cost': float(total_cost),
        'objective_value': float(res.fun),
        'motor_type': edl_system['rover']['wheel_assembly']['motor'].get('type', ''),
        'chassis_type': edl_system['rover']['chassis'].get('type', ''),
        'battery_type': edl_system['rover']['power_subsys']['battery'].get('battery_type', ''),
        'battery_modules': int(edl_system['rover']['power_subsys']['battery'].get('num_modules', 0)),
        'team_name': edl_system['team_name'],
        'team_number': edl_system['team_number']
    }

    return {
        'xbest': np.array(xbest, dtype=float),
        'fbest': float(res.fun),
        'time_edl': float(time_edl),
        'time_rover': float(time_rover),
        'time_total': float(total_time),
        'total_cost': float(total_cost),
        'edl_system': edl_system,
        'result_row': result_row
    }


# ============================================================
# MAIN ONE-RUN WRAPPER
# ============================================================

def run_one_optimization(settings=None):
    if settings is None:
        settings = SETTINGS.copy()
    else:
        merged = SETTINGS.copy()
        merged.update(settings)
        settings = merged

    problem = problem_builder(settings)
    res = run_optimizer(problem, settings)
    feas = check_feasibility(problem, res)

    output = {
        'problem': problem,
        'res': res,
        'feasibility': feas,
        'status': 'success' if feas['feasible'] else 'infeasible'
    }

    if feas['feasible']:
        eval_data = evaluate_best_design(problem, res)
        output.update(eval_data)

        if settings['save_results']:
            append_result_to_csv(settings['results_csv'], output['result_row'])

        if settings['save_pickle']:
            with open(settings['pickle_file'], 'wb') as handle:
                pickle.dump(output['edl_system'], handle, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        if settings['debug_on_infeasible']:
            print('\nConstraint vector at res.x:')
            print(feas['constraint_vector'])
            print('Largest constraint violation = {:.6f}'.format(feas['largest_violation']))
            debug_candidate(
                res.x,
                problem['edl_system'],
                problem['planet'],
                problem['mission_events'],
                problem['tmax'],
                problem['experiment'],
                problem['end_event']
            )

    return output


# ============================================================
# BATCH / LOOP HELPERS
# ============================================================

def run_batch(settings=None, arch_options=None):
    if settings is None:
        settings = SETTINGS.copy()
    else:
        merged = SETTINGS.copy()
        merged.update(settings)
        settings = merged

    if arch_options is None:
        arch_options = ARCH_OPTIONS

    if settings['reset_batch_csv']:
        reset_csv(settings['batch_csv'])

    all_results = []

    for chassis_type in arch_options['chassis_type']:
        for motor_type in arch_options['motor_type']:
            for battery_type, battery_modules in arch_options['battery']:
                for experiment_function in arch_options['experiment_function']:

                    run_settings = settings.copy()
                    run_settings['chassis_type'] = chassis_type
                    run_settings['motor_type'] = motor_type
                    run_settings['battery_type'] = battery_type
                    run_settings['battery_modules'] = battery_modules
                    run_settings['experiment_function'] = experiment_function
                    run_settings['save_results'] = False

                    print('\n==================================================')
                    print('Running batch case:')
                    print('chassis_type       =', chassis_type)
                    print('motor_type         =', motor_type)
                    print('battery_type       =', battery_type)
                    print('battery_modules    =', battery_modules)
                    print('experiment         =', experiment_function.__name__)
                    print('method             =', run_settings['method'])
                    print('==================================================')

                    out = run_one_optimization(run_settings)

                    if out['status'] == 'success':
                        append_result_to_csv(settings['batch_csv'], out['result_row'])

                    all_results.append(out)

    return all_results


# ============================================================
# MINI DRIVER: test one battery list with differential evolution
# ============================================================

battery_list = [10, 9, 8, 7, 6, 3, 2, 1]
# ['LiFePO4', 'NiMH', 'NiCD', 'PbAcid-1', 'PbAcid-2', 'PbAcid-3']
temp_csv = 'temp_number_battery_comparison.csv'

# Start fresh each time
reset_csv(temp_csv)
print("Running battery evaluation:")
for battery in battery_list:
    run_settings = SETTINGS.copy()

    # Keep defaults, only change what we care about
    run_settings['method'] = 'differential_evolution'
    run_settings['battery_modules'] = battery
    run_settings['save_results'] = False          # do not write to the main results CSV
    run_settings['results_csv'] = temp_csv        # not used since save_results=False
    run_settings['de_popsize'] = 15
    run_settings['de_maxiter'] = 30
    run_settings['de_disp'] = True
    run_settings['de_polish'] = False

    print('\n==============================================')
    print('Running battery test for:', battery)
    print('==============================================')

    result = run_one_optimization(run_settings)

    # Build a row whether feasible or not
    if result['status'] == 'success':
        row = result['result_row'].copy()
        row['feasible'] = True
        row['status'] = result['status']
        row['battery_tested'] = battery
    else:
        row = {
            'method': run_settings['method'],
            'terrain_name': run_settings['experiment_function'].__name__,
            'battery_tested': battery,
            'chassis_type': run_settings['chassis_type'],
            'battery_type': run_settings['battery_type'],
            'battery_modules': run_settings['battery_modules'],
            'status': result['status'],
            'feasible': False,
            'largest_violation': result['feasibility']['largest_violation'],
            'parachute_diameter': float(result['res'].x[0]),
            'wheel_radius': float(result['res'].x[1]),
            'chassis_mass': float(result['res'].x[2]),
            'gear_diameter': float(result['res'].x[3]),
            'fuel_mass_per_rocket': float(result['res'].x[4]),
            'objective_value': float(result['res'].fun),
        }

    append_result_to_csv(temp_csv, row)

print('\nFinished battery comparison runs.')
print('Results saved to:', temp_csv)





