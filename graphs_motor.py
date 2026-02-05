import numpy as np
import matplotlib.pyplot as plt

omega = np.linspace(0, 500, 0.1)
tau_motor = tau_dcmotor(omega, motor)

plt.plot(tau_motor, omega)
plt.xlabel("Motor Shaft Torque [Nm]")
plt.ylabel("Motor Shaft Speed [rad/s]")
plt.show