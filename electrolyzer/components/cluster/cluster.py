import numpy as np
import openmdao.api as om
from attrs import field, define, validators

from electrolyzer.core.utilities import BaseConfig
from electrolyzer.tools.validators import range_val


@define(kw_only=True)
class ClusterBaseConfig(BaseConfig):
    n_stacks: int = field(converter=int, validator=validators.gt(0.0))
    # j_max: float = field(validator=validators.ge(0.0))
    i_max: float = field(validator=validators.ge(0.0))
    turndown_ratio: float = field(validator=range_val(0.0, 1.0))
    include_startup_delay: bool = field(default=False)
    off_hours_cold_start: float = field(
        default=None, validator=validators.optional(validators.gt(0.0))
    )
    cold_start_delay_hours: float = field(
        default=None, validator=validators.optional(validators.gt(0.0))
    )


class ClusterBaseClass(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("plant_config", types=dict)
        self.options.declare("tech_config", types=dict)

    def setup(self):
        # self.n_timesteps = self.options["plant_config"]["simulation"]["n_timesteps"]
        self.dt = self.options["plant_config"]["simulation"]["dt"]
        self.config = ClusterBaseConfig.from_dict(self.options["tech_config"]["cluster_parameters"])

        # design variables
        self.add_input("n_stacks", val=self.config.n_stacks, units="unitless")

        self.add_input("current_cmd", val=0.0, shape_by_conn=True, units="A")

        self.add_output("on_off_status", val=0.0, copy_shape="current_cmd", units="unitless")
        self.add_output("current_out", val=0.0, copy_shape="current_cmd", units="A")

    def compute(self, inputs, outputs):
        # NOTE: need cell area as an input to use j_max as a bound
        i_min = self.config.turndown_ratio * self.config.i_max
        on_off_status = np.where(inputs["current_cmd"] < i_min, 0, 1)
        # i_out should also reflect warm-up losses or delays
        i_out = np.clip(inputs["current_cmd"], a_min=None, a_max=self.config.i_max)
        outputs["current_out"] = i_out
        outputs["on_off_status"] = on_off_status
