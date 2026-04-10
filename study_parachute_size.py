import numpy as np
import matplotlib.pyplot as plt
from define_edl_system import *
from subfunctions_EDL import *
from define_planet import *
from define_mission_events import *

test_diams = np.arange(14, 19.5, 0.5)

edl_system = define_edl_system_1()
mars = define_planet()
mission_events = define_mission_events()

# edl_system['altitude'] = 11000    # [m] initial altitude
# edl_system['velocity'] = -590     # [m/s] initial velocity
# edl_system['rocket']['on'] = False 
# edl_system['parachute']['deployed'] = True   # our parachute is open
# edl_system['parachute']['diameter'] = test_diams   # replace with array of diameters to test
# edl_system['parachute']['ejected'] = False   # and still attached
# edl_system['heat_shield']['ejected'] = False # heat shield is still attached
# edl_system['sky_crane']['on'] = False # sky crane is not yet active
# edl_system['speed_control']['on'] = False # speed control is not yet active
# edl_system['position_control']['on'] = False # position control is not yet active

tmax = 2000   # [s] maximum simulated time

time = []
landing_speeds = []
success = []

for diam in test_diams:
    print(f"\n--- Simulating Parachute Diameter: {diam} m ---")
    
    # re-initialize the EDL system inside the loop so fuel and hardware states reset
    edl_system = define_edl_system_1()
    
    # Apply initial conditions
    edl_system['altitude'] = 11000    # [m] initial altitude
    edl_system['velocity'] = -590     # [m/s] initial velocity
    edl_system['rocket']['on'] = False 
    edl_system['parachute']['deployed'] = True   
    edl_system['parachute']['diameter'] = diam   
    edl_system['parachute']['ejected'] = False   
    edl_system['heat_shield']['ejected'] = False 
    edl_system['sky_crane']['on'] = False 
    edl_system['speed_control']['on'] = False 
    edl_system['position_control']['on'] = False 
    
    # Run the simulation for the current parachute diameter
    T, Y, edl_system_final = simulate_edl(edl_system, mars, mission_events, tmax, True)
    
    # Extract Simulated Time
    final_time = T[-1]
    time.append(final_time)
    
    # Extract Rover Speed relative to ground
    # Final EDL absolute speed + rover relative speed
    final_speed = Y[0, -1] + Y[5, -1] 
    landing_speeds.append(final_speed)
    
    # Determine Landing Success
    # Rover on the ground, sky crane above danger altitude, and touchdown speed within threshold
    final_alt = Y[1, -1]
    danger_alt = edl_system_final['sky_crane']['danger_altitude']
    danger_speed = edl_system_final['sky_crane']['danger_speed']
    
    if (edl_system_final["rover"]["on_ground"] == True and final_alt >= danger_alt and abs(final_speed) <= abs(danger_speed)):
        success.append(1) # Success
    else:
        success.append(0) # Failure


# Visualize the simulation results
fig, axs = plt.subplots(3, 1, figsize=(8, 10))
plt.tight_layout(pad=4.0)

# Plot 1: Simulated Time vs Diameter
axs[0].plot(test_diams, time, marker='o', color='b')
axs[0].set_title('Simulated Time vs. Parachute Diameter')
axs[0].set_ylabel('Time [s]')
axs[0].grid(True)

# Plot 2: Rover Touchdown Speed vs Diameter
axs[1].plot(test_diams, landing_speeds, marker='s', color='r')
axs[1].set_title('Rover Landing Speed vs. Parachute Diameter')
axs[1].set_ylabel('Speed [m/s]')
axs[1].grid(True)

# Plot 3: Landing Success vs Diameter
axs[2].step(test_diams, success, marker='^', color='g')
axs[2].set_title('Rover Landing Success vs. Parachute Diameter')
axs[2].set_xlabel('Parachute Diameter [m]')
axs[2].set_ylabel('Success (1) / Failure (0)')
axs[2].set_yticks([0, 1])
axs[2].grid(True)

plt.show()