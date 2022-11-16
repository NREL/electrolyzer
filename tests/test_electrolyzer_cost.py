
import pytest
import numpy as np

import electrolyzer

def test_cost_init():
    # inputs
    lifetime=30
    lcoe = np.ones(lifetime)*0.1 # $/kw
    energy = np.ones(lifetime)*1E3 # kw
    hydrogen_produced = np.ones(lifetime)*10.0 #kg
    electrolyzer_type="PEM"

    # initialize cost class
    costing = electrolyzer.ElectrolyzerCost(lcoe, energy, hydrogen_produced, electrolyzer_type, lifetime)

    # check if instance was successfully created
    isinstance(costing, electrolyzer.ElectrolyzerCost)

def test_cost_lcoh():

    # inputs
    lifetime=30
    lcoe = np.ones(lifetime)*0.1 # $/kw
    energy = np.ones(lifetime)*1E3 # kw
    hydrogen_produced = np.ones(lifetime)*100000.0 #kg
    electrolyzer_type="PEM"

    # initialize electrolyzer cost class
    costing = electrolyzer.ElectrolyzerCost(lcoe, energy, hydrogen_produced, electrolyzer_type, lifetime)

    # calculate lcoh
    lcoh = costing.levelized_cost_of_hydrogen()

    # check lcoh result #TODO correct expected result when model is ready
    assert lcoh == 3.0

