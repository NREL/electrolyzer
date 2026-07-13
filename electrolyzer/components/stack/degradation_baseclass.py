import openmdao.api as om
from attrs import field, define

from electrolyzer.core.utilities import BaseConfig
from electrolyzer.tools.validators import contains


@define(kw_only=True)
class CellDegradationBaseConfig(BaseConfig):
    degradation_impact: str = field(default="hydrogen", validator=contains["hydrogen", "power"])


class CellDegradationBase(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("tech_config", types=dict, default={})
        self.options.declare("plant_config", types=dict, default={})

    def setup(self):
        self.add_input("I_in", val=0.0, shape_by_conn=True, units="A")
        self.add_input("on_off_status", val=0.0, copy_shape="I_in", units="unitless")
        self.add_input("V_cell_nominal", val=0.0, copy_shape="I_in", units="V")

        self.add_output("V_cell_degraded", val=0.0, copy_shape="I_in", units="V")

        # TODO: rename I_actual to I_out
        self.add_output("I_actual", val=0.0, copy_shape="I_in", units="A")

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        """
        Computation for the OM component.

        For a template class this is not implement and raises an error.
        """

        raise NotImplementedError("This method should be implemented in a subclass.")
