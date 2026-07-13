import numpy as np
from attrs import field, define, validators

from electrolyzer.core.utilities import BaseConfig
from electrolyzer.components.cluster.dynamics_baseclass import DynamicsBase


@define(kw_only=True)
class SimpleDynamicsConfig(BaseConfig):
    # n_stacks: int = field(converter=int, validator=validators.gt(0.0))
    # j_max: float = field(validator=validators.ge(0.0))
    # i_max: float = field(validator=validators.ge(0.0))
    # turndown_ratio: float = field(validator=range_val(0.0, 1.0))
    include_startup_delay: bool = field(default=False)
    off_hours_cold_start: float = field(
        default=None, validator=validators.optional(validators.gt(0.0))
    )
    cold_start_delay_hours: float = field(
        default=None, validator=validators.optional(validators.gt(0.0))
    )


class SimpleDynamics(DynamicsBase):
    def initialize(self):
        self.options.declare("plant_config", types=dict)
        self.options.declare("tech_config", types=dict)

    def setup(self):
        # self.n_timesteps = self.options["plant_config"]["simulation"]["n_timesteps"]
        self.dt = self.options["plant_config"]["simulation"]["dt"]
        self.config = SimpleDynamicsConfig.from_dict(
            self.options["tech_config"]["cluster_parameters"]
        )

        super().setup()
        # design variables

    def compute(self, inputs, outputs):
        on_off_status = np.where(inputs["I_in"] < inputs["I_min"], 0, 1)
        # i_out should also reflect warm-up losses or delays
        i_out = np.clip(inputs["I_in"], a_min=inputs["I_min"], a_max=inputs["I_max"])
        # TODO: add start-up delay
        outputs["I_out"] = i_out
        outputs["on_off_status"] = on_off_status
