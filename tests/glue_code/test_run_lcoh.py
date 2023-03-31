import os
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from electrolyzer import run_lcoh


lcoh_breakdown = pd.DataFrame(
    {
        "Life Totals [$]": [5.388657e06, 1.079412e06, 1.197895e07, 1.283473e06],
        "Life Totals [$/kg-H2]": [
            1.3594040320184078,
            0.2723048458021954,
            3.021946178528131,
            0.32378362036676683,
        ],
    },
    index=["CapEx", "OM", "Feedstock", "Stack Rep"],
)

RESULT = (lcoh_breakdown, 4.977438676715502)
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

    assert_frame_equal(
        calc_result[0]["LCOH Breakdown"],
        RESULT[0],
        check_dtype=False,
        check_exact=False,
        atol=1e-4,
    )

    print(calc_result[1], RESULT[1])
    assert np.isclose(calc_result[1], RESULT[1])
