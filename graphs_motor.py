import numpy as np
import matplotlib.pyplot as plt
from subfunctions import *

omega = np.arange(0, 5, 0.01) 

tau_motor = tau_dcmotor(omega, motor)

plt.subplot(3, 1, 1)
plt.plot(tau_motor, omega)
plt.xlabel("Motor Shaft Torque [Nm]")
plt.ylabel("Motor Shaft Speed [rad/s]")
plt.title("Motor Torque vs. Speed")

plt.subplot(3, 1, 2)
motor_power = tau_motor * omega
plt.plot(tau_motor,motor_power)
plt.xlabel("Motor Shaft Torque [Nm]")
plt.ylabel("Motor Power [W]")
plt.title("Motor Torque vs. Power")

plt.subplot(3, 1, 3)
plt.plot(omega, motor_power)
plt.xlabel("Motor Shaft Speed [rad/s]")
plt.ylabel("Motor Power [W]")
plt.title("Motor Torque vs. Power")

plt.tight_layout()

plt.show()