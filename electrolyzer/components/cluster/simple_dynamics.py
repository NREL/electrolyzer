import numpy as np
from attrs import field, define, validators

from electrolyzer.core.utilities import BaseConfig
from electrolyzer.components.cluster.dynamics_baseclass import DynamicsBase


@define(kw_only=True)
class SimpleDynamicsConfig(BaseConfig):
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
        self.dt = self.options["plant_config"]["simulation"]["dt"]
        self.config = SimpleDynamicsConfig.from_dict(
            self.options["tech_config"].get("cluster_parameters", {})
        )

        super().setup()
        # design variables

    def compute(self, inputs, outputs):
        on_off_status = np.where(inputs["I_in"] < inputs["I_min"], 0, 1)
        # should I_out also reflect warm-up losses or delays?
        # I_out should reflect delays or losses that result in no production
        # But a separate output should be used to reflect show partial losses (warm-up delay)

        i_out = np.clip(inputs["I_in"], a_min=inputs["I_min"], a_max=inputs["I_max"])
        outputs["I_out"] = i_out * on_off_status
        outputs["on_off_status"] = on_off_status
