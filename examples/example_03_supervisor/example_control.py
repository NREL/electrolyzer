import os

import numpy as np

from electrolyzer import run_electrolyzer
from electrolyzer.inputs.validation import load_modeling_yaml


turbine_rating = 3.4  # MW

# Create cosine test signal
test_signal_angle = np.linspace(0, 8 * np.pi, 3600 * 8 + 10)
base_value = (turbine_rating / 2) + 0.2
variation_value = turbine_rating - base_value
power_test_signal = (base_value + variation_value * np.cos(test_signal_angle)) * 1e6

fname_input_modeling = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "modeling_options.yaml"
)

# fname_input_modeling = "./modeling_options.yaml"

elec_sys, result_df = run_electrolyzer(fname_input_modeling, power_test_signal)

print(type(elec_sys))
print(result_df.tail())

modeling_input2 = load_modeling_yaml(fname_input_modeling)

# change the decision controller policy
modeling_input2["electrolyzer"]["control"]["policy"]["eager_on"] = False
modeling_input2["electrolyzer"]["control"]["policy"]["eager_off"] = False
modeling_input2["electrolyzer"]["control"]["policy"]["sequential"] = True
modeling_input2["electrolyzer"]["control"]["policy"]["even_dist"] = True
modeling_input2["electrolyzer"]["control"]["policy"]["baseline"] = False

elec_sys2, result_df2 = run_electrolyzer(modeling_input2, power_test_signal)

print(result_df2.tail())
