import numpy as np
import matplotlib.pyplot as plt

from define_experiment import *
from define_rover import *
from define_planet import *
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


# ------------------------
# Plot (3x1 required format)
# ------------------------
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
