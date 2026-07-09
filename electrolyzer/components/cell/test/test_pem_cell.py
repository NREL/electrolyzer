from types import SimpleNamespace

import numpy as np
import pytest
import openmdao.api as om
from pytest import fixture

from electrolyzer.components.cell.pem_cell import PEMCell


@fixture
def plant_config():
    return {
        "simulation": {
            "n_timesteps": 20,  # unused by cell at the moment
            "dt": 3600,
        }
    }


@fixture
def pem_cell_config(kinetics_method, water_vapor_method, act_method, Urev0_method):
    config = {
        "A_cell": 1000,
        "membrane_thickness": 0.02,
        "temperature": 80.0,
        "P_anode": 1.0,
        "P_cathode": 1.0,
        "R_ohmic_elec": 50.0e-3,
        "f1": 250,
        "f2": 0.996,
        "i_0a": 4.0e-7,
        "i_0c": 4.0e-3,
        "alpha_a": 2.0,
        "alpha_c": 0.5,
        "water_vaporization_method": water_vapor_method,  # "antoine",  # "buck", "antoine"
        "activation_method": act_method,  # "ln",  # "ln", "log10", or "arcsinh"
        "kinetics_method": kinetics_method,  # "per_electrode", # "per_electrode" or "combined"
        "Urev0_calc_method": Urev0_method,  # "normal",  # "normal" or "temp_adjusted"
    }
    return config


@fixture
def cell_namespace_config(kinetics_method, water_vapor_method, act_method, Urev0_method):
    return SimpleNamespace(
        kinetics_method=kinetics_method,
        water_vaporization_method=water_vapor_method,
        activation_method=act_method,
        Urev0_calc_method=Urev0_method,
    )


@pytest.mark.parametrize(
    "kinetics_method,water_vapor_method,act_method,Urev0_method",
    [("per_electrode", "buck", "arcsinh", "normal")],
)
def test_Urev_buck_normal(cell_namespace_config, subtests):
    cell = object.__new__(PEMCell)
    cell.config = cell_namespace_config
    Urev0 = cell.calc_Urev0(None)
    with subtests.test("Urev0 - non-temperature dependent"):
        assert pytest.approx(1.229, rel=1e-6, abs=1e-3) == Urev0

    with subtests.test("Water vaporization pressure (25 C, buck)"):
        p_H2O_sat_25C = cell.water_vaporization_pressure_arden_buck(25.0)
        assert pytest.approx(0.031685314, rel=1e-6, abs=1e-5) == p_H2O_sat_25C

    with subtests.test("Water vaporization pressure (60 C, buck)"):
        p_H2O_sat_60C = cell.water_vaporization_pressure_arden_buck(60.0)
        assert pytest.approx(0.19945145, rel=1e-6, abs=1e-5) == p_H2O_sat_60C

    with subtests.test("Water vaporization pressure (80 C, buck)"):
        p_H2O_sat_80C = cell.water_vaporization_pressure_arden_buck(80.0)
        assert pytest.approx(0.47410267, rel=1e-6, abs=1e-5) == p_H2O_sat_80C

    with subtests.test("Urev - 25 C, 1 bar (buck)"):
        Urev_25C = cell.reversible_overpotential(25.0, 1.0, 1.0)
        assert pytest.approx(1.273133, rel=1e-6, abs=1e-5) == Urev_25C

    with subtests.test("Urev - 60 C, 1 bar (buck)"):
        Urev_60C = cell.reversible_overpotential(60.0, 1.0, 1.0)
        assert pytest.approx(1.24776, rel=1e-6, abs=1e-5) == Urev_60C

    with subtests.test("Urev - 80 C, 1 bar (buck)"):
        Urev_80C = cell.reversible_overpotential(80.0, 1.0, 1.0)
        assert pytest.approx(1.226098, rel=1e-6, abs=1e-5) == Urev_80C

    with subtests.test("Urev - 80 C, 30 bar cathode, 1 bar anode (buck)"):
        Urev_80C_30_1 = cell.reversible_overpotential(80.0, 30.0, 1.0)
        assert pytest.approx(1.287387, rel=1e-6, abs=1e-5) == Urev_80C_30_1

    with subtests.test("Urev - 80 C, 1 bar cathode, 30 bar anode (buck)"):
        Urev_80C_1_30 = cell.reversible_overpotential(80.0, 1.0, 30.0)
        assert pytest.approx(1.2567425, rel=1e-6, abs=1e-5) == Urev_80C_1_30

    with subtests.test("Urev - 80 C, 10 bar cathode, 30 bar anode (buck)"):
        Urev_80C_10_30 = cell.reversible_overpotential(80.0, 10.0, 30.0)
        assert pytest.approx(1.300818, rel=1e-6, abs=1e-5) == Urev_80C_10_30


@pytest.mark.parametrize(
    "kinetics_method,water_vapor_method,act_method,Urev0_method",
    [("per_electrode", "buck", "arcsinh", "normal")],
)
def test_pem_cell_default(plant_config, pem_cell_config, subtests):
    prob = om.Problem()
    comp = PEMCell(plant_config=plant_config, tech_config={"cell_parameters": pem_cell_config})
    prob.model.add_subsystem("cell", comp, promotes=["*"])
    prob.setup()
    prob.set_val("cell.I_in", np.full(20, 1.0 * 1000), units="A")
    prob.run_model()

    with subtests.test("Current density is 1 A/cm2"):
        J_cell = prob.get_val("cell.J_out", units="A/(cm**2)")
        assert np.all(J_cell == 1.0)

    with subtests.test("Cell voltage at 1 A/cm2"):
        assert (
            pytest.approx(prob.get_val("cell.V_cell", units="V")[0], rel=1e-6, abs=1e-5)
            == 1.983580501
        )

    with subtests.test("H2 Production at 1 A/cm2"):
        assert (
            pytest.approx(prob.get_val("cell.H2_produced", units="g/h")[0], rel=1e-6, abs=1e-5)
            == 37.45005976
        )

    with subtests.test("H2 Production Rate at 1 A/cm2"):
        assert (
            pytest.approx(prob.get_val("H2_out", units="g/h")[0], rel=1e-6, abs=1e-5) == 37.45005976
        )

    with subtests.test("O2 Production at 1 A/cm2"):
        assert (
            pytest.approx(prob.get_val("cell.O2_produced", units="g/h")[0], rel=1e-6, abs=1e-5)
            == 297.20412014327
        )

    with subtests.test("O2 Production Rate at 1 A/cm2"):
        assert (
            pytest.approx(prob.get_val("O2_out", units="g/h")[0], rel=1e-6, abs=1e-5)
            == 297.20412014327
        )

    power = prob.get_val("cell.V_cell", units="V") * prob.get_val("cell.I_in", units="A")
    eff = (power / 1e3) / (prob.get_val("H2_out", units="kg/h"))
    with subtests.test("H2 Conversion efficiency at 1 A/cm2"):
        assert pytest.approx(eff[0], rel=1e-6) == 52.96601698232939


# def test_pem_cell_om(pem_cell_config, subtests):

#     prob = om.Problem()

#     comp = PEMCell(
#         plant_config=plant_config,
#         tech_config=pem_cell_config,
#     )
#     prob.model.add_subsystem("cell", comp, promotes=["*"])
#     prob.setup()
#     prob.set_val("cell.I_in", current_vals, units="A")
#     prob.run_model()
