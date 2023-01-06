import os

import numpy as np
import pandas as pd
from pytest import fixture
from numpy.testing import (
    assert_array_equal,
    assert_almost_equal,
    assert_array_almost_equal,
)

from electrolyzer import Supervisor, run_electrolyzer


turbine_rating = 3.4  # MW

# Create cosine test signal
test_signal_angle = np.linspace(0, 8 * np.pi, 3600 * 8 + 10)
base_value = (turbine_rating / 2) + 0.2
variation_value = turbine_rating - base_value
power_test_signal = (base_value + variation_value * np.cos(test_signal_angle)) * 1e6


@fixture(scope="module")
def result():
    """Run the electrolyzer once, and use its result for subsequent tests."""
    fname_input_modeling = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "test_modeling_options.yaml"
    )

    res = run_electrolyzer(fname_input_modeling, power_test_signal)

    return res


def test_run_electrolyzer(result):
    """An electrolyzer run should return two outputs."""
    assert len(result) == 2

    assert isinstance(result[0], Supervisor)
    assert isinstance(result[1], pd.DataFrame)


def test_result_df(result):
    """An electrolyzer run should return a `DataFrame` with time series output."""
    sup, df = result

    assert len(df) == len(power_test_signal)

    kg_rates = df[[col for col in df if "_kg_rate" in col]]
    cycles = df[[col for col in df if "cycles" in col]]
    degradation = df[[col for col in df if "deg" in col]]

    # Expected columns
    assert "curtailment" in df.columns
    assert "kg_rate" in df.columns
    assert "power_signal" in df.columns
    assert len(kg_rates.columns) == sup.n_stacks
    assert len(cycles.columns) == sup.n_stacks
    assert len(degradation.columns) == sup.n_stacks

    # Expected data
    assert_array_equal(df["power_signal"], power_test_signal)

    # Individual kg production should sum to full
    assert_almost_equal(df["kg_rate"].sum(), sum(kg_rates.sum()))


def test_regression(result):
    """
    Test specific values for the result. We expect this test to fail any time we make
    model changes.
    """
    _, df = result

    # Test total kg H2 produced
    assert_almost_equal(df["kg_rate"].sum(), 222.87989808848104, decimal=5)

    # Test degradation state of stacks
    degradation = df[[col for col in df if "deg" in col]]
    assert_array_almost_equal(
        degradation.tail(1).values[0],
        [
            0.00850225,
            0.00884953,
            0.00884953,
            0.00884953,
            0.00884953,
            0.00884952,
            0.00837898,
        ],
        decimal=5,
    )
