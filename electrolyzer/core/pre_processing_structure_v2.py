from pathlib import Path

import numpy as np
import openmdao.api as om

from electrolyzer.components.building_blocks import (
    IJBounds,
    ScaleDown,
    SimulateCell,
    CellPowerToCurrent,
)


#


prob = om.Problem()
model = prob.model

# 1. Create necessary pre-processing pieces to convert controller signals to other things
cell_power_cmd = np.full(20, 40.0)
n_cells = 5.0
n_stacks = 2.0
cluster_power_cmd = cell_power_cmd * n_cells * n_stacks

ivc_comp = om.IndepVarComp(name="P_command", val=cluster_power_cmd, units="W")
prob.model.add_subsystem("ivc", ivc_comp)

cell_scale_down = ScaleDown(scaling_component="cells")
stack_scale_down = ScaleDown(scaling_component="stacks")

scale_down = prob.model.add_subsystem("scale_down", om.Group(), promotes=["n_stacks", "n_cells"])
scale_down.add_subsystem(
    "cluster_to_stack",
    stack_scale_down,
    promotes_inputs=["n_stacks", ("P_in", "P_cluster_in")],
    promotes_outputs=[("P_out", "P_stack_in")],
)
scale_down.add_subsystem(
    "stack_to_cell",
    cell_scale_down,
    promotes_inputs=["n_cells", ("P_in", "P_stack_in")],
    promotes_outputs=[("P_out", "P_cell_in")],
)

# Get reference points
pre_converter_grp = prob.model.add_subsystem(
    "converter", om.Group(), promotes=["A_cell", "I_min", "I_max"]
)
# IJBounds has inputs of A_cell, turndown ratio and J_max
# IJBounds outputs I_ref_points, J_ref_points, I_max, I_min, J_max
# CellPowerToCurrent has inputs of I_ref_points, P_ref_points, and P_command
# CellPowerToCurrent outputs I_command
# SimulateCell has inputs of A_cell, I_in
# SimulateCell has outputs of V_cell_out, P_cell_out, H2_cell_out and J_out


# Below we are connecting the reference outputs if IJref to the reference inputs of p2i
pre_converter_grp.add_subsystem(
    "IJ_ref",
    IJBounds(),
    promotes_inputs=["A_cell"],
    promotes_outputs=["I_ref_points", "I_min", "I_max"],
)
pre_converter_grp.add_subsystem("ref_cell", SimulateCell(), promotes_inputs=["A_cell"])
pre_converter_grp.add_subsystem("p2i", CellPowerToCurrent(), promotes_inputs=["I_ref_points"])

# Now, we have to make a new group for the cell
# converter_cell_grp = prob.model.add_subsystem("ref_cell", om.Group())


# TODO: add a scale-down component between ivc.P_command and converter.p2i.P_command
# OR add a scale-up component between ref_cell.P_cell_out and converter.P_ref_points

# Connect the reference points to the cell
prob.model.connect("converter.I_ref_points", "converter.ref_cell.I_in")
# Connect the power output from the cell to the power to current thing
prob.model.connect("converter.ref_cell.P_cell_out", "converter.p2i.P_ref_points")
# Connect the power from the "controller" to the CellPowerToCurrent
prob.model.connect("ivc.P_command", "scale_down.P_cluster_in")  # dont need because of promotion?
prob.model.connect("scale_down.P_cell_in", "converter.p2i.P_command")

prob.setup()

prob.set_val("scale_down.cluster_to_stack.n_stacks", n_stacks, units="unitless")
prob.set_val("scale_down.stack_to_cell.n_cells", n_cells, units="unitless")


prob.run_model()
prob.get_val("converter.p2i.I_command", units="A")
prob.get_val("converter.ref_cell.P_cell_out", units="W")

# Try to set power command as reference power
# prob.set_val(
# "converter.p2i.P_command", prob.get_val("converter.ref_cell.P_cell_out", units="W"),
#  units="W")
prob.set_val(
    "ivc.P_command",
    n_cells * n_stacks * prob.get_val("converter.ref_cell.P_cell_out", units="W"),
    units="W",
)
prob.run_model()
# check the error between the reference and actual
power_curve_fit_error = prob.get_val("converter.p2i.I_command", units="A") - prob.get_val(
    "converter.I_ref_points", units="A"
)


prob.setup()
om.n2(prob, outfile=str(Path(__file__).parent / "n2_preprocessing_structure_v4.html"))
prob.final_setup()
prob.check_config(checks=["unconnected_inputs"], out_file=None)
prob.run_model()

# prob.get_val("converter.I_ref_points", units="A")

# ref_pts_bnds = IJBounds()
# ref_cell = SimulateCell()
# p2i = CellPowerToCurrent()
# pre_converter_grp.add_subsystem("IVC1", h2s_ivc_comp, promotes=["*"])
