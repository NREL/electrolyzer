import os

import pytest
from jsonschema.exceptions import ValidationError

from electrolyzer.inputs.validation import load_modeling_yaml


def test_basic_model():
    """A model config with all fields properly specified should pass validation."""
    fname_input_modeling = os.path.join(
        os.path.dirname(__file__), "test_modeling_options.yaml"
    )
    modeling_options = load_modeling_yaml(fname_input_modeling)

    # just make sure nothing is empty
    assert modeling_options["electrolyzer"]["supervisor"]
    assert modeling_options["electrolyzer"]["controller"]
    assert modeling_options["electrolyzer"]["stack"]
    assert modeling_options["electrolyzer"]["degradation"]
    assert modeling_options["electrolyzer"]["cell_params"]


def test_model_defaults():
    """A model config should be able to rely on defaults for some fields."""
    fname_input_modeling = os.path.join(
        os.path.dirname(__file__), "test_modeling_options_default.yaml"
    )
    modeling_options = load_modeling_yaml(fname_input_modeling)

    assert modeling_options["general"]["verbose"] is False
    assert modeling_options["electrolyzer"]["name"] == "electrolyzer_001"
    assert modeling_options["electrolyzer"]["description"] == "A PEM electrolyzer model"
    assert modeling_options["electrolyzer"]["dt"] == 1.0

    stack_opts = modeling_options["electrolyzer"]["stack"]
    assert stack_opts["include_degradation_penalty"] is True

    control_opts = modeling_options["electrolyzer"]["control"]
    assert control_opts["n_stacks"] == 1
    assert control_opts["control_type"] == "BaselineDeg"


def test_model_invalid_spec():
    """A model config that specifies invalid fields will fail validation."""
    fname_input_modeling = os.path.join(
        os.path.dirname(__file__), "test_modeling_options_invalid.yaml"
    )

    with pytest.raises(ValidationError):
        load_modeling_yaml(fname_input_modeling)
