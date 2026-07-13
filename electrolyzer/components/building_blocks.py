import numpy as np
import scipy
import openmdao.api as om
from openmdao.utils import units
from scipy.constants import R, physical_constants

from electrolyzer.components.constants import H2_MW


F, _, _ = physical_constants["Faraday constant"]  # Faraday's constant [C/mol]


class SimulateCell(om.ExplicitComponent):
    """Used to run simulate cell performance"""

    def initialize(self):
        self.options.declare("tech_config", types=dict, default={})

    def setup(self):
        self.add_input("I_in", val=0.0, shape_by_conn=True, units="A")
        self.add_output("J_out", val=0.0, copy_shape="I_in", units="A/(cm**2)")
        self.add_output("V_cell_out", val=0.0, copy_shape="I_in", units="V")
        self.add_output("P_cell_out", val=0.0, copy_shape="I_in", units="W")
        self.add_output("H2_cell_out", val=0.0, copy_shape="I_in", units="kg/s")

        self.add_input("A_cell", val=40.0, shape=1, units="cm**2")

    def compute(self, inputs, outputs):
        temperature_celsius = 80.0
        alpha_a = 2.0
        alpha_c = 0.5
        i_0a = 2.0e-7
        i_0c = 2.0e-3
        temp_K = units.convert_units(temperature_celsius, "degC", "K")

        ba = R * temp_K / (alpha_a * F)
        bc = R * temp_K / (alpha_c * F)

        J_cell = inputs["I_in"] / inputs["A_cell"]

        anode_ratio = np.maximum(J_cell / i_0a, 1e-30)
        cathode_ratio = np.maximum(J_cell / i_0c, 1e-30)

        V_act = ba * np.log(anode_ratio) + bc * np.log(cathode_ratio)

        outputs["V_cell_out"] = 1.229 + V_act
        outputs["J_out"] = J_cell
        outputs["P_cell_out"] = outputs["V_cell_out"] * inputs["I_in"]
        outputs["H2_cell_out"] = H2_MW * (inputs["I_in"] / (2 * F)) / 1e3


class IJBounds(om.ExplicitComponent):
    """Used to run simulate cell performance"""

    def initialize(self):
        self.options.declare("tech_config", types=dict, default={})
        self.options.declare("n_pts", types=int, default=20)

    def setup(self):
        self.add_input("turndown_ratio", val=0.1, shape=1, units="unitless")
        self.add_input("J_max", val=2.0, shape=1, units="A/(cm**2)")
        self.add_input("A_cell", val=40.0, shape=1, units="cm**2")

        self.add_output("I_max", val=0.0, shape=1, units="A")
        self.add_output("I_min", val=0.0, shape=1, units="A")
        self.add_output("J_min", val=0.0, shape=1, units="A/(cm**2)")
        self.add_output("I_ref_points", val=0.0, shape=self.options["n_pts"], units="A")
        self.add_output("J_ref_points", val=0.0, shape=self.options["n_pts"], units="A/(cm**2)")

    def compute(self, inputs, outputs):
        outputs["I_max"] = inputs["J_max"] * inputs["A_cell"]
        outputs["I_min"] = outputs["I_max"] * inputs["turndown_ratio"]
        outputs["J_min"] = outputs["I_min"] / inputs["A_cell"]

        outputs["I_ref_points"] = np.linspace(
            outputs["I_min"], outputs["I_max"], self.options["n_pts"]
        )
        outputs["J_ref_points"] = np.linspace(
            outputs["J_min"], inputs["J_max"], self.options["n_pts"]
        )


class CellPowerToCurrent(om.ExplicitComponent):
    """Convert power command to current command"""

    def initialize(self):
        self.options.declare("tech_config", types=dict, default={})

    def setup(self):
        self.add_input("I_ref_points", val=0.0, shape_by_conn=True, units="A")
        # self.add_input("J_ref_points", val=0.0, copy_shape="I_ref_points", units="A/(cm**2)")
        self.add_input("P_ref_points", val=0.0, copy_shape="I_ref_points", units="W")
        # NOTE: could use V_ref_points instead and calculate power in compute()

        self.add_input("P_command", val=0.0, shape_by_conn=True, units="W")
        self.add_output("I_command", val=0.0, copy_shape="P_command", units="A")

    def compute(self, inputs, outputs):
        def power_to_i_3rd(pwr, p1, p2, p3, p4, p5, p6):
            i_stack = p1 * (pwr**3) + p2 * (pwr**2) + (p3 * pwr) + (p4 * pwr ** (1 / 2)) + p5
            return i_stack

        curve_coeff, curve_cov = scipy.optimize.curve_fit(
            power_to_i_3rd,
            inputs["P_ref_points"],
            inputs["I_ref_points"],
            p0=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        )

        current = power_to_i_3rd(inputs["P_command"], *curve_coeff)
        outputs["I_command"] = current


class CellDegradation(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("tech_config", types=dict, default={})
        self.options.declare("deg_impact", types=str, default="hydrogen")

    def setup(self):
        self.add_input("I_in", val=0.0, shape_by_conn=True, units="A")
        self.add_input("on_off_status", val=0.0, copy_shape="I_in", units="unitless")
        self.add_input("V_cell_nominal", val=0.0, copy_shape="I_in", units="V")

        self.add_output("V_cell_degraded", val=0.0, copy_shape="I_in", units="V")
        self.add_output("I_actual", val=0.0, copy_shape="I_in", units="A")

    def compute(self, inputs, outputs):
        steady_deg_per_dt = 1.41737929e-10 * inputs["V_cell_nominal"] * inputs["on_off_status"]
        V_cell_deg = np.cumsum(steady_deg_per_dt)

        if self.options["deg_impact"] == "hydrogen":
            eff_mult = np.nan_to_num(
                (inputs["V_cell_nominal"] + V_cell_deg) / inputs["V_cell_nominal"]
            )
            outputs["I_actual"] = inputs["I_in"] / eff_mult
        else:
            outputs["I_actual"] = inputs["I_in"]

        outputs["V_cell_degraded"] = V_cell_deg


class ClusterDynamics(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("tech_config", types=dict, default={})

    def setup(self):
        self.add_input("I_in", val=0.0, shape_by_conn=True, units="A")
        self.add_input("I_min", val=0.0, shape=1, units="A")
        self.add_input("I_max", val=0.0, shape=1, units="A")

        self.add_output("I_out", val=0.0, copy_shape="I_in", units="A")
        self.add_output("on_off_status", val=0.0, copy_shape="I_in", units="unitless")

    def compute(self, inputs, outputs):
        on_off_status = np.where(inputs["I_in"] < inputs["I_min"], 0, 1)
        i_out = np.clip(inputs["I_in"], a_min=0.0, a_max=inputs["I_max"])
        outputs["I_out"] = np.where(i_out < inputs["I_min"], 0.0, i_out)
        outputs["on_off_status"] = on_off_status


class ClusterToStack(om.ExplicitComponent):
    """Scale things down from cluster to stack"""

    def initialize(self):
        self.options.declare("tech_config", types=dict, default={})

    def setup(self):
        self.add_input("n_stacks", val=1.0, shape=1, units="unitless")

        vars_to_units = {"I": "A", "H2": "kg/s", "P": "W", "V_stack": "V", "V_deg_stack": "V"}

        ref_shape = None
        for v, u in vars_to_units.items():
            if ref_shape is None:
                self.add_input(f"{v}_in", val=0.0, shape_by_conn=True, units=u)
                self.add_output(f"{v}_out", val=0.0, copy_shape=f"{v}_in", units=u)
            else:
                self.add_input(f"{v}_in", val=0.0, copy_shape=ref_shape, units=u)
                self.add_output(f"{v}_out", val=0.0, copy_shape=ref_shape, units=u)

    def compute(self, inputs, outputs):
        for o_name in outputs.keys():
            in_name = o_name.replace("_out", "_in")
            if in_name == "I_in":
                outputs[o_name] = inputs[in_name]
            else:
                outputs[o_name] = inputs[in_name] / inputs["n_stacks"]


class ScaleUp(om.ExplicitComponent):
    """Scale things down"""

    def initialize(self):
        self.options.declare("tech_config", types=dict, default={})
        self.options.declare("scaling_component", types=str)

    def setup(self):
        self.add_input(f"n_{self.options['scaling_component']}", val=1.0, shape=1, units="unitless")
        vars_to_units = {
            # "I": "A",
            "P": "W",
            "H2": "kg/s",
            # "V_stack": "V",
            # "V_deg_stack": "V"
        }
        ref_shape = None
        for v, u in vars_to_units.items():
            if ref_shape is None:
                self.add_input(f"{v}_in", val=0.0, shape_by_conn=True, units=u)
                self.add_output(f"{v}_out", val=0.0, copy_shape=f"{v}_in", units=u)
            else:
                self.add_input(f"{v}_in", val=0.0, copy_shape=ref_shape, units=u)
                self.add_output(f"{v}_out", val=0.0, copy_shape=ref_shape, units=u)

    def compute(self, inputs, outputs):
        for o_name in outputs.keys():
            in_name = o_name.replace("_out", "_in")
            if in_name == "I_in":
                outputs[o_name] = inputs[in_name]
            else:
                outputs[o_name] = inputs[in_name] * inputs[f"n_{self.options['scaling_component']}"]


class ScaleDown(om.ExplicitComponent):
    """Scale things down"""

    def initialize(self):
        self.options.declare("tech_config", types=dict, default={})
        self.options.declare("scaling_component", types=str)

    def setup(self):
        self.add_input(f"n_{self.options['scaling_component']}", val=1.0, shape=1, units="unitless")
        vars_to_units = {
            # "I": "A",
            "P": "W",
            "H2": "kg/s",
            # "V_stack": "V",
            # "V_deg_stack": "V"
        }

        ref_shape = None
        for v, u in vars_to_units.items():
            if ref_shape is None:
                self.add_input(f"{v}_in", val=0.0, shape_by_conn=True, units=u)
                self.add_output(f"{v}_out", val=0.0, copy_shape=f"{v}_in", units=u)
                ref_shape = f"{v}_in"
            else:
                self.add_input(f"{v}_in", val=0.0, copy_shape=ref_shape, units=u)
                self.add_output(f"{v}_out", val=0.0, copy_shape=ref_shape, units=u)

    def compute(self, inputs, outputs):
        for o_name in outputs.keys():
            in_name = o_name.replace("_out", "_in")
            if in_name == "I_in":
                outputs[o_name] = inputs[in_name]
            else:
                outputs[o_name] = inputs[in_name] / inputs[f"n_{self.options['scaling_component']}"]


class Draft(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("tech_config", types=dict, default={})

    def setup(self):
        pass

    def compute(self, inputs, outputs):
        pass
