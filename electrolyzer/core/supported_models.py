from electrolyzer.components.cell.pem_cell import PEMCell
from electrolyzer.components.cluster.simple_dynamics import SimpleDynamics
from electrolyzer.components.stack.simple_degradation import SimpleDegradation


supported_models = {
    # cell
    "PEMCell": PEMCell,
    # "AlkalineCell": "AlkalineCell",
    # stack
    "SimpleDegradation": SimpleDegradation,
    # cluster
    "SimpleDynamics": SimpleDynamics,
    # controller
}
