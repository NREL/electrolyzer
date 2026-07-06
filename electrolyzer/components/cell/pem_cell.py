import numpy as np
from attrs import field, define, validators
from openmdao.utils import units
from scipy.constants import R, physical_constants

from electrolyzer.tools.validators import contains, range_val
from electrolyzer.components.cell.cell import CellBaseClass, CellBaseConfig
from electrolyzer.components.constants import gibbs


F, _, _ = physical_constants["Faraday constant"]  # Faraday's constant [C/mol]


@define(kw_only=True)
class PEMCellConfig(CellBaseConfig):
    f1: float = field(default=250, validator=validators.ge(0))
    f2: float = field(default=0.996, validator=range_val(0.5, 1.0))
    # i_0a: float = field(default=2.0e-7, validator=validators.optional(range_val(0.0, 1.0)))
    # i_0c: float = field(default=2.0e-3, validator=validators.optional(range_val(0.0, 1.0)))
    i_0a: float = field(default=None, validator=validators.optional(range_val(0.0, 1.0)))
    i_0c: float = field(default=None, validator=validators.optional(range_val(0.0, 1.0)))
    alpha_a: float = field(default=2.0, validator=range_val(0.25, 4.0))
    alpha_c: float = field(default=0.5, validator=range_val(0.25, 4.0))

    b_combined: float = field(default=None, validator=validators.optional(validators.ge(0.0)))
    i0_combined: float = field(default=None, validator=validators.optional(validators.ge(0.0)))

    water_vaporization_method: str = field(
        default="buck", converter=(str.lower, str.strip), validator=contains(["buck", "antoine"])
    )
    activation_method: str = field(
        default="arcsinh", validator=contains(["ln", "log10", "arcsinh"])
    )
    kinetics_method: str = field(
        default="per_electrode",
        converter=(str.lower, str.strip),
        validator=contains(["per_electrode", "combined"]),
    )

    Urev0_calc_method: str = field(
        default="normal", validator=contains(["normal", "temp_adjusted"])
    )

    def __attrs_post_init__(self):
        extra_inputs_msg = (
            "Extraneous inputs ({extraneous_attrs_msg}) for kinetics_method {kinetics_method}. "
        )
        missing_inputs_msg = (
            "Missing inputs ({required_attrs_msg}) for kinetics_method {kinetics_method}. "
        )

        if self.kinetics_method == "per_electrode":
            # check if theres missing info or extraneous info
            required_attributes = ["i_0a", "i_0c"]
            extreanous_attributes = ["b_combined", "i0_combined"]
            extraneous_attrs_msg = ", ".join(
                f"`{k}`" for k in extreanous_attributes if getattr(self, k, None) is not None
            )
            required_attrs_msg = ", ".join(
                f"`{k}`" for k in required_attributes if getattr(self, k, None) is None
            )
            txt = ""
            if len(extraneous_attrs_msg) > 0:
                txt += extra_inputs_msg.format(
                    extraneous_attrs_msg=extraneous_attrs_msg,
                    kinetics_method=self.kinetics_method,
                )

            if len(required_attrs_msg) > 0:
                txt += missing_inputs_msg.format(
                    required_attrs_msg=required_attrs_msg, kinetics_method=self.kinetics_method
                )
            if len(txt) > 0:
                raise AttributeError(txt)

        if self.kinetics_method == "combined":
            # check if theres missing info or extraneous info
            required_attributes = ["b_combined", "i0_combined"]
            extreanous_attributes = ["i_0a", "i_0c"]
            extraneous_attrs_msg = ", ".join(
                f"`{k}`" for k in extreanous_attributes if getattr(self, k, None) is not None
            )
            required_attrs_msg = ", ".join(
                f"`{k}`" for k in required_attributes if getattr(self, k, None) is None
            )
            txt = ""
            if len(extraneous_attrs_msg) > 0:
                txt += extra_inputs_msg.format(
                    extraneous_attrs_msg=extraneous_attrs_msg,
                    kinetics_method=self.kinetics_method,
                )

            if len(required_attrs_msg) > 0:
                txt += missing_inputs_msg.format(
                    required_attrs_msg=required_attrs_msg, kinetics_method=self.kinetics_method
                )

            if len(txt) > 0:
                raise AttributeError(txt)


class PEMCell(CellBaseClass):
    def setup(self):
        # self.n = 2  # number of electrons transferred in reaction

        self.config = PEMCellConfig.from_dict(self.options["tech_config"]["cell_parameters"])

        super().setup()

        # Design parameters
        if self.config.kinetics_method == "per_electrode":
            self.add_input("i_0a", val=self.config.i_0a, shape=1, units="A/(cm**2)")
            self.add_input("i_0c", val=self.config.i_0c, shape=1, units="A/(cm**2)")
            self.add_input("alpha_a", val=self.config.alpha_a, shape=1, units="unitless")
            self.add_input("alpha_c", val=self.config.alpha_c, shape=1, units="unitless")
        else:
            # b_combined is in V/decade
            self.add_input("b_combined", val=self.config.b_combined, shape=1, units="V")
            self.add_input("i_0combined", val=self.config.i0_combined, shape=1, units="A/(cm**2)")

        self.add_input("f1", val=self.config.f1, shape=1, units="(mA**2)/(cm**4)")
        self.add_input("f2", val=self.config.f2, shape=1, units="unitless")

    def calculate_current_density(self, I_cell, A_cell):
        return I_cell / A_cell

    def calculate_current(self, J_cell, A_cell):
        return J_cell * A_cell

    def calc_Urev0(self, temperature_celsius):
        E_cell0 = gibbs / (2 * F)  # 1.229  # [V]

        # Reversible potential at 25degC - Nerst Equation
        if self.config.Urev0_calc_method == "normal":
            return E_cell0
        if self.config.Urev0_calc_method == "temp_adjusted":
            temp_K = units.convert_units(temperature_celsius, "degC", "K")
            Urev0 = E_cell0 - (0.9 * 1e-3 * (temp_K - 298))
            return Urev0

    def reversible_overpotential(self, temperature_celsius, p_cathode_bar, p_anode_bar):
        if self.config.water_vaporization_method == "buck":
            p_H2O_sat = self.water_vaporization_pressure_arden_buck(temperature_celsius)
        elif self.config.water_vaporization_method == "antoine":
            p_H2O_sat = self.water_vaporization_pressure_antoine(temperature_celsius)

        temp_k = units.convert_units(temperature_celsius, "degC", "K")

        # Daltons law of partial pressures
        p_H2 = p_cathode_bar - p_H2O_sat
        p_O2 = p_anode_bar - p_H2O_sat

        # Nerst Equation
        Urev0 = self.calc_Urev0(temperature_celsius)

        E_cell = Urev0 + ((R * temp_k) / (2 * F)) * (np.log((p_H2 * np.sqrt(p_O2)) / p_H2O_sat))

        return E_cell

    def separate_activation_overpotential(
        self, J_cell, temperature_celsius, i_0a, i_0c, alpha_a, alpha_c
    ):
        temp_K = units.convert_units(temperature_celsius, "degC", "K")

        ba = R * temp_K / (alpha_a * F)
        bc = R * temp_K / (alpha_c * F)

        anode_ratio = np.clip(J_cell / i_0a, a_min=1e-30)
        cathode_ratio = np.clip(J_cell / i_0c, a_min=1e-30)

        if self.config.activation_method == "arcsinh":
            V_acta = ba * np.arcsinh(anode_ratio)
            V_actc = bc * np.arcsinh(cathode_ratio)
        if self.config.activation_method == "ln":
            V_acta = ba * np.log(anode_ratio)
            V_actc = bc * np.log(cathode_ratio)
        if self.config.activation_method == "log10":
            V_acta = ba * np.log(10) * np.log10(anode_ratio)
            V_actc = bc * np.log(10) * np.log10(cathode_ratio)
        return V_acta + V_actc

    def combined_kinetics_activation_overpotential(self, J_cell, b_combined, i_0combined):
        ratio = np.clip(J_cell / i_0combined, a_min=1e-30)
        if self.config.activation_method == "ln":
            V_act = b_combined * np.log(ratio)
        if self.config.activation_method == "log10":
            V_act = b_combined * np.log10(ratio)
        if self.config.activation_method == "arcsinh":
            # TODO: put this error raising in the config
            msg = "Cannot run combined kinetics with arcsinh activation method"
            raise ValueError(msg)

        return V_act

    def ohmic_overpotential(self, J_cell, temperature_celsius, delta_membrane, R_elec):
        sigma_membrane = self.membrane_conductivity(temperature_celsius)

        # ionic resistance [ohms*cm^2]
        R_membrane = delta_membrane / sigma_membrane

        R_tot = R_membrane + R_elec

        V_ohmic = J_cell * R_tot

        return V_ohmic

    def cell_voltage(self, inputs):
        temp_C = inputs["operating_temperature"]
        if "current_density_in" in inputs:
            J_cell = inputs["current_density_in"]
        elif "current_in" in inputs:
            J_cell = self.calculate_current_density(
                inputs["current_in"], inputs["cell_active_area"]
            )

        Urev = self.reversible_overpotential(
            temp_C, inputs["cathode_pressure"], inputs["anode_pressure"]
        )
        U_ohmic = self.ohmic_overpotential(
            J_cell, temp_C, inputs["membrane_thickness"], inputs["R_elec"]
        )

        if self.config.kinetics_method == "per_electrode":
            V_act = self.separate_activation_overpotential(
                J_cell, temp_C, inputs["i_0a"], inputs["i_0c"], inputs["alpha_a"], inputs["alpha_c"]
            )
        else:
            V_act = self.combined_kinetics_activation_overpotential(
                J_cell, inputs["b_combined"], inputs["i_0combined"]
            )

        V_cell = Urev + U_ohmic + V_act
        return V_cell

    def get_current_density(self, inputs):
        if "current_density_in" in inputs:
            return inputs["current_density_in"]
        if "current_in" in inputs:
            J_cell = self.calculate_current_density(
                inputs["current_in"], inputs["cell_active_area"]
            )
            return J_cell

    def get_current(self, inputs):
        if "current_in" in inputs:
            return inputs["current_in"]
        if "current_density_in" in inputs:
            I_cell = self.calculate_current(
                inputs["current_density_in"], inputs["cell_active_area"]
            )
            return I_cell

    def h2_production_rate(self, inputs):
        I_cell = self.get_current(inputs)
        J_cell = self.get_current_density(inputs)
        h2_grams_per_sec = self.calculate_h2_production_rate(
            J_cell, I_cell, inputs["f1"], inputs["f2"]
        )
        return h2_grams_per_sec

    def o2_production_rate(self, inputs):
        I_cell = self.get_current(inputs)
        J_cell = self.get_current_density(inputs)
        o2_grams_per_sec = self.calculate_o2_production_rate(
            J_cell, I_cell, inputs["f1"], inputs["f2"]
        )
        return o2_grams_per_sec

    def h2_production(self, inputs):
        h2_grams_per_sec = self.h2_production_rate(inputs)
        return h2_grams_per_sec * self.dt

    def o2_production(self, inputs):
        o2_grams_per_sec = self.o2_production_rate(inputs)
        return o2_grams_per_sec * self.dt

    def power_consumption_rate(self, inputs):
        V_cell = self.cell_voltage(inputs)
        I_cell = self.get_current(inputs)
        return V_cell * I_cell

    def energy_consumption(self, inputs):
        V_cell = self.cell_voltage(inputs)
        I_cell = self.get_current(inputs)
        return V_cell * I_cell * self.dt

    def conversion_efficiency(self, inputs):
        h2_grams_per_sec = self.h2_production_rate(inputs)
        power_W_per_sec = self.power_consumption_rate(inputs)
        return power_W_per_sec / np.max([1e-30, h2_grams_per_sec])

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        self.cell_voltage(inputs)
