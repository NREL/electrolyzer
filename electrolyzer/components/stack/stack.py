import numpy as np
import rainflow
import openmdao.api as om
from attrs import field, define, validators

from electrolyzer.core.utilities import BaseConfig
from electrolyzer.tools.validators import contains


@define(kw_only=True)
class StackBaseConfig(BaseConfig):
    n_cells = field(converter=int, validator=validators.gt(0.0))
    include_degradation: bool = field()
    steady_degradation_rate: float = field(validator=validators.ge(0.0))
    cycle_degradation_rate: float = field(validator=validators.ge(0.0))
    fatigue_degradation_rate: float = field(validator=validators.ge(0.0))
    fatigue_degradation_calc_interval_hrs: int = field(converter=int, validator=validators.ge(0.0))
    degradation_impact_profile: str = field(validator=contains(["power", "hydrogen"]))


class StackBaseClass(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("plant_config", types=dict)
        self.options.declare("tech_config", types=dict)

    def setup(self):
        # self.n_timesteps = self.options["plant_config"]["simulation"]["n_timesteps"]
        self.dt = self.options["plant_config"]["simulation"]["dt"]

        self.config = StackBaseConfig.from_dict(self.options["tech_config"]["stack_parameters"])

        # design variables
        self.add_input("n_cells", val=self.config.n_cells, shape=1, units="unitless")
        self.add_input(
            "steady_degradation_rate", val=self.config.steady_degradation_rate, shape=1, units="V/s"
        )

        # input profiles
        self.add_input("current_in", val=0.0, shape_by_conn=True, units="A")
        self.add_input("on_off_status", val=0.0, copy_shape="current_in", units="unitless")
        self.add_input("cell_voltage_nominal", val=0.0, copy_shape="current_in", units="V")
        self.add_output("degradation_voltage", val=0.0, copy_shape="current_in", units="V")
        self.add_output("actual_current", val=0.0, copy_shape="current_in", units="A")

        # output profiles
        # self.add_output("power_consumed", val=0.0, copy_shape="current_in", units="W")
        # self.add_output("voltage_out", val=0.0, copy_shape="current_in", units="V")
        # self.add_output("degradation_voltage", val=0.0, copy_shape="current_in", units="V")
        # self.add_output(
        #     "hydrogen_produced", val=0.0, copy_shape="current_in", units=f"g/({self.dt}*s)"
        # )
        # self.add_output(
        #     "oxygen_produced", val=0.0, copy_shape="current_in", units=f"g/({self.dt}*s)"
        # )
        # self.add_output(
        #     "water_consumed", val=0.0, copy_shape="current_in", units=f"g/({self.dt}*s)"
        # )
        # constraints

    def adjust_current_from_degradation(self, V_cell_nom, V_cell_deg, I_nominal):
        """Reduce the current so that power consumed with degradation is equal
        to power consumed without

        Args:
            V_cell_nom (np.ndarray | float): cell voltage without degradation
            V_cell_deg (np.ndarray | float): cell degradation voltage
            I_nominal (np.ndarray | float): stack current without degradation

        Returns:
            np.ndarray | float: current to reflect degradation
        """
        eff_mult = np.nan_to_num((V_cell_nom + V_cell_deg) / V_cell_nom)
        I_actual = I_nominal / eff_mult
        return I_actual

    def steady_degradation(self, V_cell_nom, on_off_status, steady_degradation_rate):
        steady_deg_per_dt = self.dt * steady_degradation_rate * V_cell_nom * on_off_status
        return steady_deg_per_dt

    def on_off_degradation(self, on_off_status, onoff_deg_rate):
        change_stack = np.diff(on_off_status)
        cycle_cnt = np.where(change_stack < 0, -1 * change_stack, 0)
        cycle_cnt = np.array([0, *list(cycle_cnt)])
        stack_off_deg_per_dt = onoff_deg_rate * cycle_cnt

        return stack_off_deg_per_dt

    def fatigue_degradation(self, V_cell_nom, fatigue_degradation_rate, n_dt_fatigue_calc):
        v_max = np.max(V_cell_nom)
        v_min = np.min(V_cell_nom)
        V_fatigue_ts = np.zeros(len(V_cell_nom))

        if v_max == v_min:
            rf_sum = 0
            return V_fatigue_ts

        # NOTE: I'm not sure if below is flexible to varying DT
        t_calc = np.arange(0, len(V_cell_nom) + n_dt_fatigue_calc, n_dt_fatigue_calc)
        rf_cycles = rainflow.count_cycles(V_cell_nom, nbins=10)
        rf_sum = np.sum([pair[0] * pair[1] for pair in rf_cycles])
        rf_track = 0.0
        # lifetime_fatigue_deg = rf_sum * fatigue_degradation_rate
        for i in range(len(t_calc) - 1):
            voltage_signal_temp = V_cell_nom[np.nonzero(V_cell_nom[t_calc[i] : t_calc[i + 1]])]
            if np.size(voltage_signal_temp) == 0:
                rf_sum = 0
            else:
                v_max = np.max(voltage_signal_temp)
                v_min = np.min(voltage_signal_temp)

                if v_max == v_min:
                    rf_sum = 0
                else:
                    rf_cycles = rainflow.count_cycles(voltage_signal_temp, nbins=10)
                    # rf_cycles = rainflow.count_cycles(
                    #     voltage_signal[t_calc[i] : t_calc[i + 1]], nbins=10
                    # )
                    rf_sum = np.sum([pair[0] * pair[1] for pair in rf_cycles])
            rf_track += rf_sum
            V_fatigue_ts[t_calc[i] : t_calc[i + 1]] = rf_track * fatigue_degradation_rate
        return V_fatigue_ts

    def calculate_cell_degradation(self, inputs):
        if not self.config.include_degradation:
            return np.zeros(len(inputs["cell_voltage_nominal"]))

        V_deg_uptime = self.steady_degradation(
            inputs["cell_voltage_nominal"],
            inputs["on_off_status"],
            inputs["steady_degradation_rate"][0],
        )
        V_deg_onoff = self.on_off_degradation(
            inputs["on_off_status"], self.config.cycle_degradation_rate
        )
        n_dt_fatigue_calc = int(
            self.config.fatigue_degradation_calc_interval_hrs / (self.dt / 3600)
        )
        V_fatigue = self.fatigue_degradation(
            inputs["cell_voltage_nominal"], self.config.fatigue_degradation_rate, n_dt_fatigue_calc
        )
        deg_signal = np.cumsum(V_deg_uptime) + np.cumsum(V_deg_onoff) + V_fatigue

        return deg_signal

    def compute(self, inputs, outputs):
        V_cell_deg = self.calculate_cell_degradation(inputs)

        if self.config.degradation_impact_profile == "hydrogen":
            outputs["actual_current"] = self.adjust_current_from_degradation(
                inputs["cell_voltage_nominal"], V_cell_deg, inputs["current_in"]
            )
        else:
            outputs["actual_current"] = inputs["current_in"]
        outputs["degradation_voltage"] = V_cell_deg
