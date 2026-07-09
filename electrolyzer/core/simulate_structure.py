import numpy as np
import openmdao.api as om

from electrolyzer.components.building_blocks import (
    IJBounds,
    SimulateCell,
    CellDegradation,
    ClusterDynamics,
)


prob = om.Problem()
model = prob.model

# 1. Create necessary pre-processing pieces to convert controller signals to other things
ivc_comp = om.IndepVarComp(name="I_command", val=np.full(20, 40.0), units="A")
prob.model.add_subsystem("ivc", ivc_comp)

bnds = prob.model.add_subsystem("IJ_ref", IJBounds(), promotes=["*"])

# ClusterDynamics has inputs of I_in, I_min, I_max and outputs I_out and on_off_status
# CellDegradation has inputs of I_in, on_off_status, V_cell_nominal
# CellDegradation outputs V_cell_degraded and I_actual
# SimulateCell has inputs of A_cell, I_in
# SimulateCell has outputs of V_cell_out, P_cell_out, H2_cell_out and J_out
simulation = prob.model.add_subsystem(
    "simulation", om.Group(), promotes=["I_min", "I_max", "A_cell"]
)
simulation.add_subsystem("dynamics", ClusterDynamics(), promotes=["I_min", "I_max"])  # UNSURE
simulation.add_subsystem("cell_nominal", SimulateCell(), promotes=["A_cell"])
simulation.add_subsystem("degradation", CellDegradation())
simulation.add_subsystem("cell_real", SimulateCell(), promotes=["A_cell"])
prob.model.connect("ivc.I_command", "simulation.dynamics.I_in")

# connect dynamics current output to nominal cell current input
simulation.connect("dynamics.I_out", "cell_nominal.I_in")
# connect dynamics on/off status output to degradation on/off status input
simulation.connect("dynamics.on_off_status", "degradation.on_off_status")
# connect dynamics current output to the degradation nominal current input
simulation.connect("dynamics.I_out", "degradation.I_in")
# connect nominal cell voltage to the degradation
simulation.connect("cell_nominal.V_cell_out", "degradation.V_cell_nominal")
# connect the degraded current to the cell voltage
simulation.connect("degradation.I_out", "cell_real.I_in")

# r = om.SqliteRecorder('circuit.sqlite')
# prob.driver.add_recorder(r)

from pathlib import Path


prob.setup()

# prob.set_val("simulation.dynamics.I_in",np.full(20, 40.0), units="A")
om.n2(prob, outfile=str(Path(__file__).parent / "n2_simulate_structure.html"))
prob.final_setup()
prob.check_config(checks=["unconnected_inputs"], out_file=None)
prob.run_model()


# prob.model.connect("IJ_ref.I_max", "dynamics.I_max")
# prob.model.connect("IJ_ref.I_max", "dynamics.I_max")
# cluster: (dynamics)
# stack: (degradation)
# cell: (performance)
