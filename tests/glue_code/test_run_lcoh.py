import numpy as np
import os
import pandas as pd
from pandas.testing import assert_frame_equal
from pathlib import Path

from electrolyzer import run_lcoh


lcoh_breakdown = pd.DataFrame(
    {
        'Life Totals [$]': [5.388657e+06, 1.079412e+06, 1.197895e+07, 1.283473e+06],
        'Life Totals [$/kg-H2]': [1.359484, 0.272321, 3.022124, 0.323803]
    },
    index=['CapEx','OM','Feedstock','Stack Rep']
)

RESULT = (
    lcoh_breakdown,
    4.9777312843759915
)
ROOT = Path(__file__).parent.parent.parent


def test_run_lcoh():
    fname_input_modeling = os.path.join(
        ROOT, "examples", "example_04_lcoh", "cost_modeling_options.yaml"
    )
    
    turbine_rating = 3.4  # MW

    # Create cosine test signal
    test_signal_angle = np.linspace(0, 8 * np.pi, 3600 * 8 + 10)
    base_value = (turbine_rating / 2) + 0.2
    variation_value = turbine_rating - base_value
    power_test_signal = (base_value + variation_value * np.cos(test_signal_angle)) * 1e6

    lcoe = 44.18 * (1 / 1000)

    calc_result = run_lcoh(fname_input_modeling, power_test_signal, lcoe)

    assert_frame_equal(calc_result[0]["LCOH Breakdown"], RESULT[0], check_dtype=False, check_exact=False, atol=1e-4)
    assert np.isclose(calc_result[1], RESULT[1])
