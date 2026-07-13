from electrolyzer.connectors.scale_power import ScalePowerUp, ScalePowerDown
from electrolyzer.components.cell.pem_cell import PEMCell
from electrolyzer.connectors.bounds_baseclass import CurrentBoundsBase
from electrolyzer.components.cluster.simple_dynamics import SimpleDynamics
from electrolyzer.components.stack.simple_degradation import SimpleDegradation
from electrolyzer.translators.simple_power_translator import PowerToCurrentCurveFit
from electrolyzer.control.openloop.simple_openloop_control import OLBasicSplit


supported_models = {
    # cell
    "PEMCell": PEMCell,
    # "AlkalineCell": "AlkalineCell",
    # stack
    "SimpleDegradation": SimpleDegradation,
    # cluster
    "SimpleDynamics": SimpleDynamics,
    # controller
    "OLBasicSplit": OLBasicSplit,
    # TRANSLATORS
    "PowerToCurrentCurveFit": PowerToCurrentCurveFit,
    # CONNECTORS
    "ScalePowerUp": ScalePowerUp,
    "ScalePowerDown": ScalePowerDown,
    "CurrentBoundsBase": CurrentBoundsBase,
}
