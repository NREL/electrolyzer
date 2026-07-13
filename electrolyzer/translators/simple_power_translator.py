import scipy
from attrs import field, define

from electrolyzer.core.utilities import BaseConfig
from electrolyzer.tools.validators import contains
from electrolyzer.translators.curve_fits import cubic_with_sqrt_5coeffs
from electrolyzer.translators.power_translator_baseclass import PowerToCurrentBase


curve_shapes = {"cubic_with_sqrt_5coeffs": cubic_with_sqrt_5coeffs}


@define(kw_only=True)
class PIConfig(BaseConfig):
    curve_to_use: str = field(
        default="cubic_with_sqrt_5coeffs", validator=contains(["cubic_with_sqrt_5coeffs"])
    )


class PowerToCurrentCurveFit(PowerToCurrentBase):
    def setup(self):
        super().setup()

        self.config = PIConfig.from_dict(self.options["tech_config"]["curve_fit_parameters"])

    def compute(self, inputs, outputs):
        p2i_func = curve_shapes[self.config.curve_to_use]
        curve_coeff, curve_cov = scipy.optimize.curve_fit(
            p2i_func,
            inputs["P_ref_points"],
            inputs["I_ref_points"],
            p0=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        )

        current = p2i_func(inputs["P_command"], *curve_coeff)
        outputs["I_command"] = current
