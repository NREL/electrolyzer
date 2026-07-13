import os
from pathlib import Path

from electrolyzer.core.bert import BERT


os.chdir(Path(__file__).parent)
config_fpath = Path(__file__).parent / "bert_config.yaml"
bert = BERT(config_fpath)
bert.run()

scale_fac = bert.model.get_val("Cluster0.n_stacks", units="unitless") * bert.model.get_val(
    "Cluster0.n_cells", units="unitless"
)

p_cell_ref = bert.model.get_val("Cluster0.converter.ref_cell.P_cell_out", units="W")
p_system_ref = p_cell_ref * scale_fac
bert.model.set_val("controller.P_command", p_system_ref, units="W")
bert.run()
i_out = bert.model.get_val("Cluster0.translator.command_to_current.I_command", units="A")
I_input = bert.model.get_val("Cluster0.converter.I_ref_points", units="A")
i_error = i_out - I_input
