#!/usr/bin/env python

"""Tests for `electrolyzer` package."""

import pytest
import numpy as np

import electrolyzer


@pytest.fixture
def response():
    """Sample pytest fixture.

    See more at: http://doc.pytest.org/en/latest/fixture.html
    """
    # import requests
    # return requests.get('https://github.com/audreyr/cookiecutter-pypackage')


def test_content(response):
    """Sample pytest test function with the pytest fixture as an argument."""
    # from bs4 import BeautifulSoup
    # assert 'GitHub' in BeautifulSoup(response.content).title.string

def test_cost_init():
    # inputs
    lifetime=30
    lcoe = np.ones(lifetime)*0.1 # $/kw
    energy = np.ones(lifetime)*1E3 # kw
    hydrogen_produced = np.ones(lifetime)*10.0 #kg
    electrolyzer_type="PEM"

    # initialize cost class
    costing = electrolyzer.cost.cost(lcoe, energy, hydrogen_produced, electrolyzer_type, lifetime)

    # check if instance was successfully created
    isinstance(costing, electrolyzer.cost.cost)

def test_cost_lcoh():

    # inputs
    lifetime=30
    lcoe = np.ones(lifetime)*0.1 # $/kw
    energy = np.ones(lifetime)*1E3 # kw
    hydrogen_produced = np.ones(lifetime)*100000.0 #kg
    electrolyzer_type="PEM"

    # initialize electrolyzer cost class
    costing = electrolyzer.cost.cost(lcoe, energy, hydrogen_produced, electrolyzer_type, lifetime)

    # calculate lcoh
    lcoh = costing.levelized_cost_of_hydrogen()

    # check lcoh result
    assert lcoh == 3.0

