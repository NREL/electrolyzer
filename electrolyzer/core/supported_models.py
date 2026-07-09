from electrolyzer.components.stack.stack import StackBaseClass
from electrolyzer.components.cell.pem_cell import PEMCell
from electrolyzer.components.cluster.cluster import ClusterBaseClass


supported_models = {
    # cell
    "PEMCell": PEMCell,
    # "AlkalineCell": "AlkalineCell",
    # stack
    "StackBaseClass": StackBaseClass,
    # cluster
    "ClusterBaseClass": ClusterBaseClass,
    # controller
}
