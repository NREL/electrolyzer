import pytest
from pytest import fixture

from electrolyzer.components.cell.pem_cell import PEMCellConfig


@fixture
def pem_cell_config():
    config = {
        "A_cell": 1000,
        "membrane_thickness": 0.02,
        "temperature": 80.0,
        "P_anode": 1.0,
        "P_cathode": 1.0,
        "R_ohmic_elec": 50.0e-3,
        "f1": 250,
        "f2": 0.996,
        "water_vaporization_method": "antoine",  # "buck", "antoine"
        "activation_method": "ln",  # "ln", "log10", or "arcsinh"
        # "kinetics_method": "per_electrode", # "per_electrode" or "combined"
        "Urev0_calc_method": "normal",  # "normal" or "temp_adjusted"
    }
    return config


def test_config_combined_kinetics(pem_cell_config, subtests):
    correct_inputs = {
        "kinetics_method": "combined",
        "b_combined": 0.045,
        "i0_combined": 2.0e-8,
        "i_0a": None,
        "i_0c": None,
    }

    config = PEMCellConfig.from_dict(pem_cell_config | correct_inputs)
    with subtests.test("Correct initialization"):
        assert config.b_combined == correct_inputs["b_combined"]
        assert config.i0_combined == correct_inputs["i0_combined"]

    with subtests.test("Other args are None"):
        assert config.i_0a is None
        assert config.i_0c is None

    # Test error if wrong method
    # wrong_method = {
    #     "kinetics_method": "per_electrode",
    #     "b_combined": 0.045,
    #     "i0_combined": 2.0e-8,
    # }

    # with subtests.test("incorrect parameters"):
    #     with pytest.raises(AttributeError) as excinfo:
    #         config = PEMCellConfig.from_dict(pem_cell_config | wrong_method)
    #     err = str(excinfo.value)
    #     assert "For kinetics_method of 'per_electrode', the inputs" in err

    # Test error if missing i0_combined
    missing_i0 = {
        "kinetics_method": "combined",
        "b_combined": 0.045,
    }
    with subtests.test("Missing input i0_combined"):
        with pytest.raises(AttributeError) as excinfo:
            config = PEMCellConfig.from_dict(pem_cell_config | missing_i0)
        err = str(excinfo.value)
        assert "Missing inputs (`i0_combined`)" in err

    # Test error if missing b_combined
    missing_b = {
        "kinetics_method": "combined",
        "i0_combined": 2.0e-8,
    }

    with subtests.test("Missing input b_combined"):
        with pytest.raises(AttributeError) as excinfo:
            config = PEMCellConfig.from_dict(pem_cell_config | missing_b)
        err = str(excinfo.value)
        assert "Missing inputs (`b_combined`)" in err

    # Test error if missing both inputs
    missing_both = {
        "kinetics_method": "combined",
    }

    with subtests.test("Missing both inputs"):
        with pytest.raises(AttributeError) as excinfo:
            config = PEMCellConfig.from_dict(pem_cell_config | missing_both)
        err = str(excinfo.value)
        assert "Missing inputs (`b_combined`, `i0_combined`)" in err

    # Test error if one extra arg given
    extra_arg = {
        "kinetics_method": "combined",
        "b_combined": 0.045,
        "i0_combined": 2.0e-8,
        "i_0a": 2.0e-7,
    }

    with subtests.test("Extraneous input"):
        with pytest.raises(AttributeError) as excinfo:
            config = PEMCellConfig.from_dict(pem_cell_config | extra_arg)
        err = str(excinfo.value)
        assert "Extraneous inputs (`i_0a`)" in err

    # Test error if multiple extra args given
    extra_args = {
        "kinetics_method": "combined",
        "b_combined": 0.045,
        "i0_combined": 2.0e-8,
        "i_0a": 2.0e-7,
        "i_0c": 2.0e-3,
    }

    with subtests.test("Extraneous inputs"):
        with pytest.raises(AttributeError) as excinfo:
            config = PEMCellConfig.from_dict(pem_cell_config | extra_args)
        err = str(excinfo.value)
        assert "Extraneous inputs (`i_0a`, `i_0c`)" in err
