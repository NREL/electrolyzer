import os

import numpy as np
import pytest

from electrolyzer import BERT_EXAMPLE_DIR
from electrolyzer.core.bert import BERT


def test_example_00_no_controller(subtests):
    example_fpath = BERT_EXAMPLE_DIR / "example_00_refactor"
    os.chdir(example_fpath)

    config_fpath = example_fpath / "bert_config.yaml"
    bert = BERT(config_fpath, make_n2=False)
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
