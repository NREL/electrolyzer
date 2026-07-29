import numpy as np
import openmdao.api as om
from attrs import field, define, validators
from openmdao.utils import units
from scipy.constants import physical_constants

from electrolyzer.core.utilities import BaseConfig
from electrolyzer.components.constants import H2_MW, O2_MW


F, _, _ = physical_constants["Faraday constant"]  # Faraday's constant [C/mol]


@define(kw_only=True)
class CellBaseConfig(BaseConfig):
    A_cell: float = field(validator=(validators.gt(0.0)))
    membrane_thickness: float = field(validator=(validators.gt(0.0)))
    temperature: float = field(validator=(validators.ge(50.0)))
    P_anode: float = field(validator=(validators.ge(0.0)))
    P_cathode: float = field(validator=(validators.ge(0.0)))
    R_ohmic_elec: float = field(validator=(validators.ge(0.0)))


class CellBaseClass(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("plant_config", types=dict)
        self.options.declare("tech_config", types=dict)

    def setup(self):
        # self.n_timesteps = self.options["plant_config"]["simulation"]["n_timesteps"]
        self.dt = self.options["plant_config"]["simulation"]["dt"]

        # design variables
        self.add_input("A_cell", val=self.config.A_cell, units="cm**2")
        self.add_input("membrane_thickness", val=self.config.membrane_thickness, units="cm")
        self.add_input("operating_temperature", val=self.config.temperature, units="degC")
        self.add_input("anode_pressure", self.config.P_anode, units="bar")
        self.add_input("cathode_pressure", self.config.P_cathode, units="bar")
        self.add_input("R_elec", self.config.R_ohmic_elec, units="ohm*(cm**2)")

        # input profiles
        # TODO: make either current density or current
        self.add_input("I_in", val=0.0, shape_by_conn=True, units="A")  # OR current density?
        # self.add_input("J_in", val=0.0, shape_by_conn=True, units="A/(cm**2)")

        # output profiles
        self.add_output("V_cell_out", val=0.0, copy_shape="I_in", units="V")
        self.add_output("H2_produced", val=0.0, copy_shape="I_in", units=f"g/({self.dt}*s)")
        self.add_output("O2_produced", val=0.0, copy_shape="I_in", units=f"g/({self.dt}*s)")
        # self.add_output(
        #     "H2O_consumed", val=0.0, copy_shape="I_in", units=f"g/({self.dt}*s)"
        # )
        self.add_output("H2_cell_out", val=0.0, copy_shape="I_in", units="g/s")
        self.add_output("O2_cell_out", val=0.0, copy_shape="I_in", units="g/s")

        self.add_output("J_out", val=0.0, copy_shape="I_in", units="A/(cm**2)")
        self.add_output("P_cell_out", val=0.0, copy_shape="I_in", units="W")

        # output design variables
        # self.add_output("rated_V_cell", val=0.0, units="V")
        # self.add_output("rated_conversion_efficiency", val=0.0, units="W*h/kg")
        # self.add_output("rated_H2_production", val=0.0, units="g/s")
        # self.add_output("rated_O2_production", val=0.0, units="g/s")
        # self.add_output("rated_cell_power", val=0.0, units="W")

    def compute(self, inputs, outputs, discrete_inputs, discrete_outputs):
        """
        Computation for the OM component.

        For a template class this is not implement and raises an error.
        """

        raise NotImplementedError("This method should be implemented in a subclass.")

    def membrane_conductivity(self, temperature_celsius):
        temp_K = units.convert_units(temperature_celsius, "degC", "K")
        lambda_water_content = ((-2.89556 + (0.016 * temp_K)) + 1.625) / 0.1875

        # membrane proton conductivity [S/cm]
        sigma = ((0.005139 * lambda_water_content) - 0.00326) * np.exp(
            1268 * ((1 / 303) - (1 / temp_K))
        )
        return sigma

    def water_vaporization_pressure_antoine(self, temperature_celsius):
        # coefficient for Antoine formulas
        if temperature_celsius <= 100:
            A = 8.07131
            B = 1730.63
            C = 233.426
        elif temperature_celsius > 100 and temperature_celsius < 374:
            A = 8.14019
            B = 1810.94
            C = 224.485

        # vapor pressure of water in [mmHg] using Antoine formula
        # valid for T<283 Kelvin
        p_h2o_sat_mmHg = 10 ** (A - (B / (C + temperature_celsius)))

        # convert mmHg to bar
        p_h2O_sat = units.convert_units(p_h2o_sat_mmHg, "torr", "bar")
        return p_h2O_sat

    def water_vaporization_pressure_arden_buck(self, temperature_celsius):
        # Arden Buck equation T=C, https://www.omnicalculator.com/chemistry/vapour-pressure-of-water#vapor-pressure-formulas # noqa
        # Reasonable at temperatures between 0-100C
        # different coefficients if temperature < 0 degrees C
        p_h2o_sat_Pa = (
            0.61121
            * np.exp(
                (18.678 - (temperature_celsius / 234.5))
                * (temperature_celsius / (257.14 + temperature_celsius))
            )
        ) * 1e3  # (Pa)

        # convert Pa to bar
        p_h2o_sat = units.convert_units(p_h2o_sat_Pa, "Pa", "bar")
        return p_h2o_sat

    def calculate_faradaic_efficiency(self, J_cell, f1, f2):
        J_cell_mA_pr_cm2 = J_cell * 1e3

        n_F = ((J_cell_mA_pr_cm2**2) / (f1 + (J_cell_mA_pr_cm2**2))) * f2

        return n_F

    def calculate_h2_production_rate(self, J_cell, I_cell, f1, f2):
        n_F = self.calculate_faradaic_efficiency(J_cell, f1, f2)
        h2_production_mol_per_s = n_F * (I_cell / (2 * F))
        h2_production_g_per_s = h2_production_mol_per_s * H2_MW
        return h2_production_g_per_s

    def calculate_o2_production_rate(self, J_cell, I_cell, f1, f2):
        n_F = self.calculate_faradaic_efficiency(J_cell, f1, f2)
        o2_production_mol_per_s = n_F * (I_cell / (4 * F))
        o2_production_g_per_s = o2_production_mol_per_s * O2_MW
        return o2_production_g_per_s

    def calculate_h2_production(self, J_cell, I_cell, f1, f2):
        h2_g_per_s = self.calculate_h2_production_rate(J_cell, I_cell, f1, f2)
        return h2_g_per_s * self.dt

    def calculate_o2_production(self, J_cell, I_cell, f1, f2):
        o2_g_per_s = self.calculate_o2_production_rate(J_cell, I_cell, f1, f2)
        return o2_g_per_s * self.dt
