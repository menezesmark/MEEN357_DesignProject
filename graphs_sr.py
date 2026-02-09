import numpy as np
import matplotlib.pyplot as plt
from subfunctions import *

omega_motor = np.arange(0, 4, 0.01)

tau_motor = tau_dcmotor(omega_motor, motor)

ng = get_gear_ratio(speed_reducer)

omega_sr = omega_motor / ng

tau_sr = tau_motor * ng

power_sr = tau_sr * omega_sr

plt.figure(figsize=(8, 10))

plt.subplot(3, 1, 1)
plt.plot(tau_sr, omega_sr)
plt.xlabel("SR Output Torque [N-m]")
plt.ylabel("SR Output Speed [rad/s]")
plt.title("Speed Reducer Output: Speed vs. Torque")

plt.subplot(3, 1, 2)
plt.plot(tau_sr, power_sr)
plt.xlabel("SR Output Torque [N-m]")
plt.ylabel("SR Output Power [W]")
plt.title("Speed Reducer Output: Power vs. Torque")

plt.subplot(3, 1, 3)
plt.plot(omega_sr, power_sr)
plt.xlabel("SR Output Speed [rad/s]")
plt.ylabel("SR Output Power [W]")
plt.title("Speed Reducer Output: Power vs. Speed")

plt.tight_layout()
plt.show()
