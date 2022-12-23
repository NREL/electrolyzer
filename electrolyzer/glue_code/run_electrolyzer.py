"""
This module is responsible for running electrolyzer models based on YAML configuration
files.
"""

import numpy as np
import pandas as pd

import electrolyzer.inputs.validation as val
from electrolyzer import Supervisor


def run_electrolyzer(fname_input_modeling, power_signal):
    """
    Runs an electrolyzer simulation based on a YAML configuration file and power
    signal input.

    Args:
        fname_input_modeling (`str`): filepath specifying the YAML config file
        power_signal (`list`): An array representing power input

    Returns:
        `Supervisor`: the instance used to run the simulation
        `pandas.DataFrame`: a `DataFrame` representing the time series output
            for the system, including values for each electrolyzer stack
    """
    # Validate yaml configuration
    modeling_options = val.load_modeling_yaml(fname_input_modeling)

    # Initialize system
    elec_sys = Supervisor.from_dict(modeling_options["electrolyzer"])

    # Define output variables
    kg_rate = np.zeros((elec_sys.n_stacks, len(power_signal)))
    degradation = np.zeros((elec_sys.n_stacks, len(power_signal)))
    curtailment = np.zeros((len(power_signal)))
    tot_kg = np.zeros((len(power_signal)))
    cycles = np.zeros((elec_sys.n_stacks, len(power_signal)))
    uptime = np.zeros((elec_sys.n_stacks, len(power_signal)))
    p_in = []

    # Run electrolyzer simulation
    for i in range(len(power_signal)):
        # if (i % 1000) == 0:
        #     print('Progress', i)
        # print(i)
        loop_H2, loop_h2_mfr, loop_power_left, curtailed = elec_sys.run_control(
            power_signal[i]
        )
        p_in.append(power_signal[i] / elec_sys.n_stacks / 1000)

        tot_kg[i] = np.copy(loop_H2)
        curtailment[i] = np.copy(curtailed) / 1000000
        for j in range(elec_sys.n_stacks):
            kg_rate[j, i] = loop_h2_mfr[j]
            degradation[j, i] = elec_sys.stacks[j].V_degradation
            cycles[j, i] = elec_sys.stacks[j].cycle_count
            uptime[j, i] = elec_sys.stacks[j].uptime

    # Collect results into a DataFrame
    results_df = pd.DataFrame(index=range(len(power_signal)))

    results_df["power_signal"] = power_signal
    results_df["curtailment"] = curtailment
    results_df["kg_rate"] = tot_kg

    for i, stack in enumerate(elec_sys.stacks):
        id = i + 1
        results_df[f"stack_{id}_deg"] = degradation[i, :]
        results_df[f"stack_{id}_fatigue"] = stack.fatigue_history
        results_df[f"stack_{id}_cycles"] = cycles[i, :]
        results_df[f"stack_{id}_uptime"] = uptime[i, :]
        results_df[f"stack_{id}_kg_rate"] = kg_rate[i, :]

    return elec_sys, results_df
