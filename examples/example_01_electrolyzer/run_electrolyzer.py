"""
This example runs the Electrolyzer on its own and generates polarization curves.
"""
import numpy as np
import matplotlib.pyplot as plt

from electrolyzer.electrolyzer import Electrolyzer, electrolyzer_model


n_cells = 100  # number of cells in stack
cell_area = 1000  # cell area, cm^2
temperature = 60  # temperature

elec = Electrolyzer(n_cells, cell_area, temperature, dt=1)

cur = np.linspace(0, 2500, 100)
fit_error = np.zeros_like(cur)
p_actual = np.zeros_like(cur)
p_fit = np.zeros_like(cur)
voltage = np.zeros_like(cur)

for i in range(len(cur)):
    fit_error[i] = elec.calc_stack_power(cur[i]) - elec.calc_stack_power(
        electrolyzer_model(
            (elec.calc_stack_power(cur[i]), temperature), *elec.fit_params
        )
    )
    p_actual[i] = elec.calc_stack_power(cur[i])
    p_fit[i] = elec.calc_stack_power(
        electrolyzer_model(
            (elec.calc_stack_power(cur[i]), temperature), *elec.fit_params
        )
    )
    voltage[i] = elec.calc_cell_voltage(cur[i])

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

ax1.plot(cur / (1000), fit_error / elec.stack_rating_kW * 100)
ax1.grid(True)
ax1.set_ylabel("error (% rated power)")
ax1.set_title("Power error (actual - fit)")
ax2.plot(cur / (1000), p_actual / elec.stack_rating_kW)
ax2.plot(cur / (1000), p_fit / elec.stack_rating_kW)
ax2.grid(True)
ax2.set_title("power normalized by stack rating")
ax2.set_ylabel("normalized power")
ax2.set_xlabel("Current density (A/cm^2)")
ax2.legend(["actual", "fit"])

plt.figure()
plt.plot(cur / 1000, voltage)
plt.grid(True)
plt.xlabel("Current density (A/cm^2)")
plt.ylabel("Voltage (V)")
plt.title("Polarization Curve")
plt.show()
