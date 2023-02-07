"""This module provides unit tests for `Stack`."""
import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_array_almost_equal

from electrolyzer import Cell, Stack


def create_stack():
    stack_dict = {
        "n_cells": 100,
        "cell_area": 1000,
        "temperature": 60,
        "max_current": 2000,
    }
    return Stack.from_dict(stack_dict)


@pytest.fixture
def stack():
    return create_stack()


def test_init(mocker):
    """Should initialize properly."""

    # mock side effects (these will have their own unit tests)
    spy_calc_state_space = mocker.spy(Stack, "calc_state_space")
    spy_create_polarization = mocker.spy(Stack, "create_polarization")
    spy_cell = mocker.spy(Cell, "from_dict")

    # for this example, set stack rating explicitly
    stack_dict = {
        "n_cells": 100,
        "cell_area": 1000,
        "temperature": 60,
        "max_current": 2000,
        "stack_rating_kW": 750,
    }

    stack = Stack.from_dict(stack_dict)

    assert stack.n_cells == stack_dict["n_cells"]
    assert stack.cell_area == stack_dict["cell_area"]
    assert stack.temperature == stack_dict["temperature"]
    assert stack.max_current == stack_dict["max_current"]

    # case: stack rating provided by user
    assert stack.stack_rating_kW == stack_dict["stack_rating_kW"]
    assert stack.stack_rating == stack_dict["stack_rating_kW"] * 1e3
    assert stack.min_power == 0.1 * stack.stack_rating

    assert stack.include_degradation_penalty is True
    assert stack.rf_track == 0.0
    assert stack.V_degradation == 0.0
    assert stack.uptime == 0.0
    assert stack.cycle_count == 0
    assert stack.fatigue_history == 0.0
    assert stack.hourly_counter == 0.0
    assert stack.hour_change is False
    assert_array_equal(stack.voltage_signal, [])
    assert_array_equal(stack.voltage_history, [])
    assert stack.stack_on is False
    assert stack.stack_waiting is False
    assert stack.turn_on_delay == 600
    assert stack.turn_on_time == 0
    assert stack.turn_off_time == -1000
    assert stack.wait_time == stack.turn_on_time
    assert stack.dt == 1.0
    assert stack.time == 0
    assert stack.tau == 5.0
    assert stack.stack_state == 0.0

    # these side effects are tested in isolation elsewhere, here we just want to
    # make sure they get called properly
    spy_calc_state_space.assert_called_once()
    spy_create_polarization.assert_called_once()
    spy_cell.assert_called_once()


def test_run(mocker):
    stack = create_stack()

    spy_update_deg = mocker.spy(Stack, "update_degradation")
    spy_calc_p = mocker.spy(Stack, "calc_stack_power")
    spy_calc_mfr = mocker.spy(Cell, "calc_mass_flow_rate")
    spy_update_dynamics = mocker.spy(Stack, "update_dynamics")
    spy_update_status = mocker.spy(Stack, "update_status")

    # stack off
    orig_state = stack.stack_state
    stack.run(stack.stack_rating)

    spy_update_deg.assert_not_called()
    spy_calc_p.assert_not_called()
    spy_calc_mfr.assert_not_called()
    spy_update_dynamics.assert_not_called()
    spy_update_status.assert_called_once()

    assert stack.time == stack.dt
    assert stack.V_degradation == 0
    assert stack.stack_state == orig_state
    assert len(stack.voltage_history) == 1

    # stack on
    stack = create_stack()
    mocker.resetall()
    stack.stack_on = True
    stack.run(stack.stack_rating)

    spy_update_deg.assert_called_once()
    spy_calc_p.assert_called_once()
    spy_calc_mfr.assert_called_once()
    spy_update_dynamics.assert_called_once()

    assert stack.stack_state != orig_state

    # degradation applied from prev timestep
    stack.run(stack.stack_rating)
    assert stack.V_degradation != 0

    # fast forward an hour
    stack.time = 3600
    stack.run(stack.stack_rating)
    assert stack.hourly_counter == 1
    assert stack.hour_change is True
    assert len(stack.voltage_signal) == 3
    assert_array_equal(stack.voltage_history, [])


def test_create_polarization(stack: Stack):
    """
    Should create a polarization curve based on fit for the specified model over a
    range of temperatures.
    """
    fit_params = stack.create_polarization()

    # this is brittle, so for now use a lenient precision check
    expected = [
        -2.28261081e-03,
        -1.50848325e-02,
        7.89259537e-03,
        4.80671306e00,
        9.74923247e-01,
        1.36179580e01,
    ]
    assert_array_almost_equal(fit_params, expected, decimal=3)


def test_convert_power_to_current(stack: Stack):
    """Converts an input power to stack current."""
    current = stack.convert_power_to_current(stack.stack_rating_kW, stack.temperature)

    # We don't get exactly max current out, for now check within 5% tolerance
    tol = 0.05
    assert abs(current - stack.max_current) < (tol * stack.max_current)

    # TODO: users should enter a power <= rated power, but should
    # this be enforced in the code? Potentially apply curtailment rules once
    # those have been developed.


def test_curtail_power():
    """TODO: This method is still being implemented."""
    pass


def test_calc_fatigue_degradation(stack: Stack):
    """Should return a voltage penalty calculated based on a fatigue analysis."""
    temperature = 60
    v_stack = (
        stack.cell.calc_cell_voltage(stack.max_current, temperature) * stack.n_cells
    )
    signal = np.linspace(0, v_stack, 100)  # 100 sec voltage ramp
    penalty = stack.calc_fatigue_degradation(signal)

    assert penalty > 0 and penalty < v_stack
    rf_track = stack.rf_track
    assert rf_track > 0

    # Longer signal over same range should lead to higher penalty
    signal2 = np.linspace(0, signal, 500)
    stack.rf_track = 0
    penalty2 = stack.calc_fatigue_degradation(signal2)

    assert penalty2 > penalty
    assert stack.rf_track > rf_track


def test_calc_steady_degradation(stack: Stack):
    # """Should return a voltage penalty as a function of uptime."""
    # penalty = stack.calc_steady_degradation()

    # assert penalty == 0

    # # penalty should increase with uptime
    # stack.uptime = 3600
    # penalty_1hr = stack.calc_steady_degradation()

    # assert penalty_1hr > penalty

    """Should return a voltage penalty as a function of voltage history"""
    stack.stack_on = True

    penalty_0s = stack.d_s

    assert penalty_0s == 0

    # Ramp power input from 0 to rated
    t = 100
    P_in = np.linspace(0, stack.stack_rating, t)
    for p in P_in:
        stack.run(p)

    # steady degradation should be greater than 0
    penalty_100s = stack.d_s

    # steady degradation should be less than if operating at rated
    max_voltage = stack.cell.calc_cell_voltage(stack.max_current, stack.temperature)
    penalty_max = max_voltage * t * stack.rate_steady

    assert penalty_100s > penalty_0s
    assert penalty_100s < penalty_max


def test_calc_onoff_degradation(stack: Stack):
    """Should return a voltage penalty as a function of on/off cycles."""
    penalty = stack.calc_onoff_degradation()

    assert penalty == 0

    # penalty should increase with cycles
    stack.cycle_count = 10
    penalty_10c = stack.calc_onoff_degradation()

    assert penalty_10c > penalty


def test_update_degradation(mocker):
    # mock side effects, as these are unit tested elsewhere
    spy_calc_fatigue = mocker.spy(Stack, "calc_fatigue_degradation")
    spy_calc_steady = mocker.spy(Stack, "calc_steady_degradation")
    spy_calc_onoff = mocker.spy(Stack, "calc_onoff_degradation")

    temperature = 60
    stack = create_stack()
    v_stack = (
        stack.cell.calc_cell_voltage(stack.max_current, temperature) * stack.n_cells
    )
    stack.update_degradation()

    # at startup, no voltage signal, hasn't been running for an hour
    spy_calc_fatigue.assert_not_called()
    spy_calc_steady.assert_called_once()
    spy_calc_onoff.assert_called_once()
    assert stack.fatigue_history == 0
    assert stack.V_degradation == 0

    # after an hour, but voltage difference is too low for fatigue
    mocker.resetall()
    stack = create_stack()
    signal = np.repeat([v_stack], 100)  # 100 sec steady voltage
    stack.voltage_signal = signal
    stack.uptime = 3600
    stack.hour_change = True
    stack.update_degradation()

    spy_calc_fatigue.assert_not_called()
    spy_calc_steady.assert_called_once()
    spy_calc_onoff.assert_called_once()
    assert stack.fatigue_history == 0
    assert stack.V_degradation > 0

    # after an hour, with sufficient voltage diff
    mocker.resetall()
    stack = create_stack()
    stack.uptime = 3600
    signal = np.linspace(0, v_stack, 100)  # 100 sec voltage ramp
    stack.voltage_signal = signal
    stack.hour_change = True
    stack.update_degradation()

    spy_calc_fatigue.assert_called_once()
    assert_array_equal(spy_calc_fatigue.call_args.args[1], signal)
    spy_calc_steady.assert_called_once()
    spy_calc_onoff.assert_called_once()
    assert stack.fatigue_history > 0
    assert stack.V_degradation > 0


def test_update_temperature(stack: Stack):
    """TODO: This method is still being implemented."""
    assert stack.update_temperature(0, 0) == stack.temperature


def test_update_dynamics(stack: Stack):
    """
    Should update stack state and apply H2 MFR filter to simulate dynamic response.
    """
    H2_mfr = stack.cell.calc_mass_flow_rate(stack.max_current) * stack.n_cells
    new_state, filtered_H2_mfr = stack.update_dynamics(H2_mfr, stack.stack_state)

    # filtered state should be lower than
    assert new_state != stack.stack_state
    assert filtered_H2_mfr < H2_mfr

    new_state2, filtered_H2_mfr2 = stack.update_dynamics(H2_mfr, new_state)

    # should ramp up if we feed the same MFR again
    assert new_state2 != new_state
    assert H2_mfr > filtered_H2_mfr2 > filtered_H2_mfr


def test_calc_state_space(stack: Stack):
    """Should produce a discretized state space system representing stack operation."""
    ss_d = stack.calc_state_space()

    assert len(ss_d) == 4

    expected = [[[0.81873075]], [[0.90634623]], [[0.2]], [[0.0]]]
    assert_array_almost_equal(ss_d, expected)
    assert_array_almost_equal(stack.DTSS, expected)


def test_update_status(stack: Stack):
    """Should turn a stack on if it has waited long enough."""

    # Should be off to start
    stack.stack_waiting = True
    stack.update_status()
    assert stack.stack_on is False

    # initiate startup
    stack.turn_stack_on()
    # fast forward one step past waiting time
    stack.time = stack.turn_on_delay + stack.dt
    stack.update_status()
    assert stack.stack_on is True


def test_turn_stack_off(stack: Stack):
    """Should turn stack off."""
    stack.time = 100
    # should not do anything if stack is on
    stack.turn_stack_off()
    assert stack.turn_off_time != stack.time
    assert stack.cycle_count == 0

    stack.stack_on = True
    stack.turn_stack_off()
    assert stack.turn_off_time == stack.time
    assert stack.stack_on is False
    assert stack.stack_waiting is False
    assert stack.cycle_count == 1
    assert stack.wait_time == 0


def test_turn_stack_on(stack: Stack):
    """Should initiate the stack startup process."""
    stack.turn_stack_on()

    assert stack.turn_on_time == stack.time
    assert stack.stack_waiting is True
    assert stack.wait_time == stack.turn_on_delay

    # should simply return if already on
    stack.time = 500
    stack.stack_on = True
    assert stack.turn_on_time != stack.time


def test_calc_stack_power(stack: Stack):
    """Should calculate stack power for a given DC current."""
    assert stack.calc_stack_power(stack.max_current) == stack.stack_rating_kW


def test_calc_electrolysis_efficiency(stack: Stack):
    """
    Should calculate values of electrolysis efficiency for given DC Power input and MFR.
    """
    H2_mfr = stack.cell.calc_mass_flow_rate(stack.max_current * 0.8) * stack.n_cells
    eta_values = stack.calc_electrolysis_efficiency(
        stack.stack_rating_kW, H2_mfr * 3600
    )

    assert len(eta_values) == 3

    # efficiency should decrease as we approach max current due to overpotentials
    assert eta_values[0] > 80  # highest efficiency around 80% capacity
    H2_mfr2 = stack.cell.calc_mass_flow_rate(stack.max_current) * stack.n_cells
    eta_values2 = stack.calc_electrolysis_efficiency(
        stack.stack_rating_kW, H2_mfr2 * 3600
    )
    assert eta_values2[0] < eta_values[0]
