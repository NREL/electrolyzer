import scipy
import openmdao.api as om
from attrs import field, define

from electrolyzer.core.utilities import BaseConfig
from electrolyzer.tools.validators import contains
from electrolyzer.translators.curve_fits import cubic_with_sqrt_5coeffs


curve_shapes = {"cubic_with_sqrt_5coeffs": cubic_with_sqrt_5coeffs}


@define(kw_only=True)
class PIConfig(BaseConfig):
    curve_to_use: str = field(
        default="cubic_with_sqrt_5coeffs", validator=contains(["cubic_with_sqrt_5coeffs"])
    )

    def __attrs_post_init__(self):
        if self.curve_to_use not in curve_shapes:
            raise ValueError(f"{self.curve_to_use} not a valid curve")


# class PowerToCurrentCurveFit(PowerToCurrentBase):
#     def setup(self):
#         super().setup()

#         self.config = PIConfig.from_dict(self.options["tech_config"]["curve_fit_parameters"])

#     def compute(self, inputs, outputs):
#         p2i_func = curve_shapes[self.config.curve_to_use]
#         curve_coeff, curve_cov = scipy.optimize.curve_fit(
#             p2i_func,
#             inputs["P_ref_points"],
#             inputs["I_ref_points"],
#             p0=(1.0, 1.0, 1.0, 1.0, 1.0),
#         )

#         current = p2i_func(inputs["P_command"], *curve_coeff)
#         outputs["I_command"] = current


class PowerToCurrentCurveCoeff(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("tech_config", types=dict, default={})
        self.options.declare("plant_config", types=dict, default={})

    def setup(self):
        self.config = PIConfig.from_dict(self.options["tech_config"]["curve_fit_parameters"])
        n_coeffs = int(self.config.curve_to_use.split("coeffs")[0].split("_")[-1])

        self.add_input("I_ref_points", val=0.0, shape_by_conn=True, units="A")
        self.add_input("P_ref_points", val=0.0, copy_shape="I_ref_points", units="W")
        # NOTE: could use V_ref_points instead and calculate power in compute()

        self.add_output("curve_coeffs", val=0.0, shape=n_coeffs, units="A/W")
        self.inputs_hash = ""

    def compute(self, inputs, outputs):
        inputs_hash = inputs.get_hash()
        if inputs_hash != self.inputs_hash:
            p2i_func = curve_shapes[self.config.curve_to_use]
            curve_coeff, curve_cov = scipy.optimize.curve_fit(
                p2i_func,
                inputs["P_ref_points"],
                inputs["I_ref_points"],
                p0=(1.0, 1.0, 1.0, 1.0, 1.0),
            )

            outputs["curve_coeffs"] = curve_coeff

            self.inputs_hash = inputs_hash


class PowerToCurrent(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("tech_config", types=dict, default={})
        self.options.declare("plant_config", types=dict, default={})

    def setup(self):
        self.config = PIConfig.from_dict(self.options["tech_config"]["curve_fit_parameters"])
        n_coeffs = int(self.config.curve_to_use.split("coeffs")[0].split("_")[-1])

        self.add_input("curve_coeffs", val=0.0, shape=n_coeffs, units="A/W")
        self.add_input("P_command", val=0.0, shape_by_conn=True, units="W")
        self.add_output("I_command", val=0.0, copy_shape="P_command", units="A")

    def compute(self, inputs, outputs):
        p2i_func = curve_shapes[self.config.curve_to_use]
        curve_coeff = tuple(inputs["curve_coeffs"])
        current = p2i_func(inputs["P_command"], *curve_coeff)
        outputs["I_command"] = current
