import numpy as np
import matplotlib.pyplot as plt

from define_experiment import *
from subfunctions import *

'''This script simulates the rover'''

experiment, end_event = experiment1()

# Update end_event
end_event['max_distance'] = 1000
end_event['max_time'] = 10000
end_event['min_velocity'] = 0.01

rover = simulate_rover(rover, planet, experiment, end_event)

telemetry = rover['telemetry']

t = telemetry['Time']
v = telemetry['velocity']
x = telemetry['position']
P = telemetry['power']


plt.figure()

# Position vs Time
plt.subplot(3,1,1)
plt.plot(t, x)
plt.ylabel('Position (m)')
plt.title('Rover Simulation Results')
plt.grid(True)

# Velocity vs Time
plt.subplot(3,1,2)
plt.plot(t, v)
plt.ylabel('Velocity (m/s)')
plt.grid(True)

# Power vs Time
plt.subplot(3,1,3)
plt.plot(t, P)
plt.xlabel('Time (s)')
plt.ylabel('Power (W)')
plt.grid(True)

plt.tight_layout()
plt.show()


print("\n--- Rover Telemetry Summary ---")
print(f"Completion Time (s): {telemetry['completion_time']:.2f}")
print(f"Distance Traveled (m): {telemetry['distance_traveled']:.2f}")
print(f"Max Velocity (m/s): {telemetry['max_velocity']:.4f}")
print(f"Average Velocity (m/s): {telemetry['average_velocity']:.4f}")
print(f"Battery Energy (J): {telemetry['battery_energy']:.2f}")
print(f"Energy per Distance (J/m): {telemetry['energy_per_distance']:.2f}")

