import numpy as np
import openmdao.api as om
from attrs import field, define, validators

from electrolyzer.core.utilities import BaseConfig
from electrolyzer.tools.validators import range_val


@define(kw_only=True)
class CurrentBoundsBaseConfig(BaseConfig):
    n_ref_points: int = field(converter=int, default=20)
    turndown_ratio: float = field(validator=range_val(0.0, 1.0))
    J_max: float = field(validator=validators.gt(0.0))
    A_cell: float = field(validator=(validators.gt(0.0)))


class CurrentBoundsBase(om.ExplicitComponent):
    """Provides the operational bounds of the electrolyzer via the operating current.
    Also provides reference points of the current within these bounds which is used for
    getting reference values of the cell performance (used for curve-fits) and
    classifying the system performance at beginning-of-life and at rated operating conditions.
    """

    def initialize(self):
        self.options.declare("tech_config", types=dict, default={})
        self.options.declare("cell_config", types=dict, default={})

    def setup(self):
        config_dict = self.options["tech_config"]["bounds_parameters"] | {
            "A_cell": self.options["cell_config"]["A_cell"]
        }
        self.config = CurrentBoundsBaseConfig.from_dict(config_dict)
        self.add_input("turndown_ratio", val=self.config.turndown_ratio, shape=1, units="unitless")
        self.add_input("J_max", val=self.config.J_max, shape=1, units="A/(cm**2)")
        self.add_input("A_cell", val=self.config.A_cell, shape=1, units="cm**2")

        self.add_output("I_max", val=0.0, shape=1, units="A")
        self.add_output("I_min", val=0.0, shape=1, units="A")
        # self.add_output("J_min", val=0.0, shape=1, units="A/(cm**2)")
        self.add_output("I_ref_points", val=0.0, shape=self.config.n_ref_points, units="A")
        # self.add_output("J_ref_points", val=0.0, shape=self.options["n_pts"], units="A/(cm**2)")

    def compute(self, inputs, outputs):
        outputs["I_max"] = inputs["J_max"] * inputs["A_cell"]
        outputs["I_min"] = outputs["I_max"] * inputs["turndown_ratio"]
        # outputs["J_min"] = outputs["I_min"] / inputs["A_cell"]

        outputs["I_ref_points"] = np.linspace(
            outputs["I_min"], outputs["I_max"], self.config.n_ref_points
        )
        # outputs["J_ref_points"] = np.linspace(
        #     outputs["J_min"], inputs["J_max"], self.options["n_pts"]
        # )
