import numpy as np
import openmdao.api as om


class CurrentBoundsBase(om.ExplicitComponent):
    """Used to run simulate cell performance"""

    def initialize(self):
        self.options.declare("tech_config", types=dict, default={})
        # TODO: make n_pts a config input
        self.options.declare("n_pts", types=int, default=20)

    def setup(self):
        self.add_input("turndown_ratio", val=0.1, shape=1, units="unitless")
        self.add_input("J_max", val=2.0, shape=1, units="A/(cm**2)")
        self.add_input("A_cell", val=40.0, shape=1, units="cm**2")

        self.add_output("I_max", val=0.0, shape=1, units="A")
        self.add_output("I_min", val=0.0, shape=1, units="A")
        # self.add_output("J_min", val=0.0, shape=1, units="A/(cm**2)")
        self.add_output("I_ref_points", val=0.0, shape=self.options["n_pts"], units="A")
        # self.add_output("J_ref_points", val=0.0, shape=self.options["n_pts"], units="A/(cm**2)")

    def compute(self, inputs, outputs):
        outputs["I_max"] = inputs["J_max"] * inputs["A_cell"]
        outputs["I_min"] = outputs["I_max"] * inputs["turndown_ratio"]
        # outputs["J_min"] = outputs["I_min"] / inputs["A_cell"]

        outputs["I_ref_points"] = np.linspace(
            outputs["I_min"], outputs["I_max"], self.options["n_pts"]
        )
        # outputs["J_ref_points"] = np.linspace(
        #     outputs["J_min"], inputs["J_max"], self.options["n_pts"]
        # )
