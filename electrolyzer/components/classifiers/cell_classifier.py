import numpy as np
import openmdao.api as om


class CellClassification(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("tech_config", types=dict, default={})
        self.options.declare("plant_config", types=dict, default={})

    def setup(self):
        self.add_input("I_max", val=0.0, shape=1, units="A")
        self.add_input("I_min", val=0.0, shape=1, units="A")
        self.add_input("I_ref_points", val=0.0, shape_by_conn=True, units="A")

        self.vars_to_units = {
            "J": "A/(cm**2)",
            "P": "kW",
            "H2": "kg/h",
            "O2": "kg/h",
            # "H2O": "kg/s",
            "V": "V",
        }

        ref_shape = "I_ref_points"
        for v, u in self.vars_to_units.items():
            self.add_input(f"{v}_in", val=0.0, copy_shape=ref_shape, units=u)
            self.add_output(f"{v}_min", val=0.0, shape=1, units=u)
            self.add_output(f"{v}_max", val=0.0, shape=1, units=u)

        # energy bounds
        self.add_output("efficiency_min", val=0.0, shape=1, units="kW*h/kg")
        self.add_output("efficiency_max", val=0.0, shape=1, units="kW*h/kg")
        # Should output:
        # - rated cell voltage
        # - rated current density
        # - rated power consumption
        # - rated h2 production rate
        # - rated efficiency
        # - rated o2 production rate
        # - rated water consumption rate

        # self.add_input("J_max", val=self.config.J_max, shape=1, units="A/(cm**2)")
        # self.add_input("J_min", val=self.config.J_max, shape=1, units="A/(cm**2)")

    def compute(self, inputs, outputs):
        idx_ref_min = np.argwhere(inputs["I_ref_points"] <= inputs["I_min"]).flatten()[-1]
        idx_ref_max = np.argwhere(inputs["I_ref_points"] >= inputs["I_max"]).flatten()[0]

        for v in list(self.vars_to_units.keys()):
            outputs[f"{v}_min"] = inputs[f"{v}_in"][idx_ref_min]
            outputs[f"{v}_max"] = inputs[f"{v}_in"][idx_ref_max]

        # kWh/kg
        efficiency = inputs["P_in"] / inputs["H2_in"]
        outputs["efficiency_min"] = efficiency[idx_ref_min]
        outputs["efficiency_max"] = efficiency[idx_ref_max]
