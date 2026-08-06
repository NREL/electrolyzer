import os
import copy

import numpy as np
import pytest

from electrolyzer import BERT_EXAMPLE_DIR
from electrolyzer.core.bert import BERT
from electrolyzer.core.file_utils import load_yaml


def test_example_00_no_controller(subtests):
    example_fpath = BERT_EXAMPLE_DIR / "example_00_refactor"
    os.chdir(example_fpath)

    config_fpath = example_fpath / "bert_config.yaml"
    config = load_yaml(config_fpath)
    config["simulation"]["n_timesteps"] = 20
    bert = BERT(config, make_n2=False)
    bert.run()

    scale_fac = bert.model.get_val("Cluster0.n_stacks", units="unitless") * bert.model.get_val(
        "Cluster0.n_cells", units="unitless"
    )

    p_cell_ref = bert.model.get_val("Cluster0.converter.ref_cell.P_cell_out", units="W")
    p_system_ref = p_cell_ref * scale_fac
    bert.model.set_val("controller.P_command", p_system_ref, units="W")
    bert.run()
    i_estimated = bert.model.get_val("Cluster0.translator.command_to_current.I_command", units="A")
    i_actual = bert.model.get_val("Cluster0.converter.I_ref_points", units="A")

    i_error = i_estimated - i_actual

    with subtests.test("100 cells per stack"):
        assert pytest.approx(100.0, rel=1e-6) == bert.model.get_val("Cluster0.n_cells")[0]

    with subtests.test("1 stack per cluster"):
        assert pytest.approx(1.0, rel=1e-6) == bert.model.get_val("Cluster0.n_stacks")[0]

    with subtests.test("I-V Curve fit error is less than 0.105 A"):
        assert np.all(np.abs(i_error) < 0.105)

    with subtests.test("Initial operating temperature"):
        assert pytest.approx(80.0, rel=1e-6) == bert.model.get_val(
            "operating_temperature", units="degC"
        )

    V_initial = copy.deepcopy(
        bert.model.get_val("Cluster0.simulation.cell_real.V_cell_out", units="V")
    )
    coeff_initial = copy.deepcopy(
        bert.model.get_val("Cluster0.translator.command_to_current.curve_coeffs", units="A/W")
    )
    V_ref_initial = copy.deepcopy(
        bert.model.get_val("Cluster0.converter.ref_cell.V_cell_out", units="V")
    )
    P_ref_initial = copy.deepcopy(
        bert.model.get_val("Cluster0.converter.ref_cell.P_cell_out", units="W")
    )
    with subtests.test("Initial reference rated power"):
        assert (
            pytest.approx(4467.1560, rel=1e-6)
            == bert.model.get_val("Cluster0.converter.ref_cell.P_cell_out", units="W")[-1]
        )
    with subtests.test("Initial curve coefficients"):
        expected_initial_coeff = np.array(
            [7.08472908e-10, -1.70727901e-05, 4.78002528e-01, 2.34225327e00, -1.42414827e01]
        )
        assert pytest.approx(expected_initial_coeff, rel=1e-6, abs=1e-8) == coeff_initial
    with subtests.test("Initial reference rated voltage"):
        assert (
            pytest.approx(2.233578003273652, rel=1e-6)
            == bert.model.get_val("Cluster0.converter.ref_cell.V_cell_out", units="V")[-1]
        )

    # Change one of the cell design parameters
    bert.model.set_val("operating_temperature", 60.0, units="degC")
    bert.run()
    V_new = bert.model.get_val("Cluster0.simulation.cell_real.V_cell_out", units="V")
    coeff_new = bert.model.get_val(
        "Cluster0.translator.command_to_current.curve_coeffs", units="A/W"
    )
    V_ref_new = bert.model.get_val("Cluster0.converter.ref_cell.V_cell_out", units="V")
    P_ref_new = bert.model.get_val("Cluster0.converter.ref_cell.P_cell_out", units="W")

    with subtests.test("Reference power points changed"):
        P_ref_diff = np.abs(P_ref_initial - P_ref_new)
        assert np.all(P_ref_diff < 91.0)
        assert np.all(P_ref_diff > 0.20)

    with subtests.test("Reference voltage points changed"):
        V_ref_diff = np.abs(V_ref_initial - V_ref_new)
        assert np.all(V_ref_diff < 0.046)
        assert np.all(V_ref_diff > 0.0004)

    with subtests.test("Real simulation voltage changed"):
        assert not all(k for k in np.isclose(V_new, V_initial, rtol=1e-6, atol=1e-6))

    with subtests.test("Curve coefficients changed"):
        expected_coeff = np.array(
            [8.30333109e-10, -1.92246906e-05, 4.75331901e-01, 2.52511351e00, -1.62209719e01]
        )
        assert pytest.approx(expected_coeff, rel=1e-6, abs=1e-8) == coeff_new

    with subtests.test("Cluster min voltage"):
        assert (
            pytest.approx(
                bert.model.get_val("Cluster0.classifier.cell_classifier.V_max", units="V"), rel=1e-6
            )
            == bert.model.get_val("Cluster0.classifier.V_max", units="V") / scale_fac
        )

    with subtests.test("Cell/stack/cluster max power"):
        cell_rated_power = bert.model.get_val(
            "Cluster0.classifier.cell_to_stack_ub.P_in", units="kW"
        )
        stack_rated_power = bert.model.get_val(
            "Cluster0.classifier.stack_to_cluster_ub.P_in", units="kW"
        )
        assert (
            pytest.approx(
                cell_rated_power * bert.model.get_val("Cluster0.n_cells", units="unitless"),
                rel=1e-6,
            )
            == stack_rated_power
        )
        assert pytest.approx(
            bert.model.get_val("Cluster0.classifier.P_max", units="kW"), rel=1e-6
        ) == stack_rated_power * bert.model.get_val("Cluster0.n_stacks", units="unitless")

    with subtests.test("Rated conversion efficiency"):
        assert pytest.approx(60.84498639, rel=1e-6) == bert.model.get_val(
            "Cluster0.classifier.efficiency_max", units="kW*h/kg"
        )

    with subtests.test("Min conversion efficiency"):
        assert pytest.approx(48.23406508, rel=1e-6) == bert.model.get_val(
            "Cluster0.classifier.efficiency_min", units="kW*h/kg"
        )

    with subtests.test("H2 rated production"):
        assert (
            pytest.approx(7.491416242830744, rel=1e-6)
            == bert.model.get_val("Cluster0.classifier.H2_max", units="kg/h")[0]
        )

    with subtests.test("Cluster H2 Production"):
        cell_h2 = bert.model.get_val(
            "Cluster0.simulation.cell_real.H2_cell_out", units="kg/h"
        ).sum()

        assert (
            pytest.approx(cell_h2 * scale_fac, rel=1e-6)
            == bert.model.get_val("Cluster0.simulation.Cluster_H2", units="kg/h").sum()
        )
    with subtests.test("Cluster H2 Production (value)"):
        assert (
            pytest.approx(81.49985544233095, rel=1e-6)
            == bert.model.get_val("Cluster0.simulation.Cluster_H2", units="kg/h").sum()
        )

    with subtests.test("Cluster Voltage"):
        cell_voltage = bert.model.get_val(
            "Cluster0.simulation.degradation_combiner.V_cell_total", units="V"
        )[-1]

        assert (
            pytest.approx(cell_voltage * scale_fac, rel=1e-6)
            == bert.model.get_val("Cluster0.simulation.Cluster_V", units="V")[-1]
        )
    with subtests.test("Cluster Voltage (value)"):
        cell_voltage = bert.model.get_val(
            "Cluster0.simulation.degradation_combiner.V_cell_total", units="V"
        )[-1]

        assert (
            pytest.approx(227.1918107704954, rel=1e-6)
            == bert.model.get_val("Cluster0.simulation.Cluster_V", units="V")[-1]
        )

    with subtests.test("Degradation power"):
        V_cell_bol = bert.model.get_val("Cluster0.simulation.cell_nominal.V_cell_out", units="V")
        V_cell_deg = bert.model.get_val(
            "Cluster0.simulation.degradation.V_cell_degraded", units="V"
        )
        I_deg = bert.model.get_val("Cluster0.simulation.degradation.I_actual", units="A")
        I_bol = bert.model.get_val("Cluster0.simulation.degradation.I_in", units="A")
        P_cell_bol = I_bol * V_cell_bol
        P_cell_deg = I_deg * (V_cell_deg + V_cell_bol)
        assert np.allclose(P_cell_bol, P_cell_deg)
        assert np.allclose(P_cell_bol * scale_fac, P_cell_deg * scale_fac)
