"""
This example performs a fully controlled electrolyzer simulation using the
`run_electrolyzer` function. See `example_run.ipynb` for an interactive option.
"""

import os

import numpy as np

# from electrolyzer import run_alkaline as run_electrolyzer
from electrolyzer import run_electrolyzer


# the alkaline system is 20 5 kW stacks for an overall system rating of 1 MW
fname_input_modeling = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "default_alkaline.yaml"
)

turbine_rating = 3.4  # MW

# Create cosine test signal
test_signal_angle = np.linspace(0, 8 * np.pi, 3600 * 8 + 10)
base_value = (turbine_rating / 2) + 0.2
variation_value = turbine_rating - base_value
power_test_signal = (base_value + variation_value * np.cos(test_signal_angle)) * 1e6

elec_sys, results_df = run_electrolyzer(fname_input_modeling, power_test_signal)
print(results_df.tail())
