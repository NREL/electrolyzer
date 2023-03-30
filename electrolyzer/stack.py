"""This module defines a Hydrogen Electrolyzer Stack."""
import numpy as np
import scipy
import pandas as pd
import rainflow
from attrs import field, define
from scipy.signal import tf2ss, cont2discrete

from .cell import Cell, electrolyzer_model
from .type_dec import NDArrayFloat, FromDictMixin, array_converter


@define
class Stack(FromDictMixin):
    # Stack parameters #
    ####################
    cell_area: float
    max_current: float
    temperature: float
    n_cells: int
    dt: float

    min_power: float = None
    stack_rating_kW: float = None

    # initialized in __attrs_post_init
    cell: Cell = field(init=False)
    fit_params: NDArrayFloat = field(init=False)
    stack_rating: float = field(init=False)

    # Degradation state #
    #####################

    # [s] amount of time this electrolyzer stack has been running
    uptime: float = field(init=False, default=0)

    cell_voltage: float = field(init=False, default=0)

    # conversion factor from steady to degradation V
    rate_steady: float = field(init=False, default=1.41737929e-10)

    # [V] degradation penalty from steady operation only
    d_s: float = field(init=False, default=0)

    # fatigue value for tracking fatigue in terms of "stress cycles"
    # rainflow counting
    rf_track: float = field(init=False, default=0)

    # [V] running count of fatigue voltage penalty
    fatigue_history: float = field(init=False, default=0)

    hourly_counter: float = field(init=False, default=0)
    hour_change: bool = field(init=False, default=False)
    voltage_signal: NDArrayFloat = field(
        init=False, default=[], converter=array_converter
    )
    voltage_history: NDArrayFloat = field(
        init=False, default=[], converter=array_converter
    )

    # conversion factor from rf_track to degradation V
    rate_fatigue: float = field(init=False, default=3.33330244e-07)

    # [V] degradation from fluctuating power only
    d_f: float = field(init=False, default=0)

    # numer of times the stack has been turned off
    cycle_count: int = field(init=False, default=0)

    # conversion from cycle_count to degradation V
    rate_onoff: float = field(init=False, default=1.47821515e-04)

    # [V] degradation from on/off cycling only
    d_o: float = field(init=False, default=0)

    include_degradation_penalty: bool = True

    # [V] running degradation voltage penalty
    V_degradation: float = field(init=False, default=0)

    # Stack dynamics #
    ##################

    # Current (A)
    I: float = field(init=False, default=0.0)

    # 10 minute startup procedure
    stack_on: bool = field(init=False, default=False)
    stack_waiting: bool = field(init=False, default=False)

    # [s] 10 minute base turn on delay, for large time steps
    base_turn_on_delay: float = 600

    # [s] 10 minute time delay for PEM electrolyzer startup procedure
    # (set in __attrs_post_init__)
    turn_on_delay: float = field(init=False)

    # keep track of when the stack was last turned on
    turn_on_time: float = field(init=False, default=0)

    # keep track of when the stack was last turned off
    turn_off_time: float = field(init=False, default=-1000)

    # wait time for partial startup procedure (set in __attrs_post_init)
    wait_time: float = field(init=False)

    # [s] total time of simulation
    time: float = field(init=False, default=0)

    # [s] time constant https://www.sciencedirect.com/science/article/pii/S0360319911021380 section 3.4 # noqa
    tau: float = 5

    stack_state: float = field(init=False, default=0)

    # state space, (set in __attrs_post_init)
    DTSS: NDArrayFloat = field(init=False)

    # whether 1st order dynamics should be ignored according to dt size
    ignore_dynamics: bool = field(init=False, default=False)

    def __attrs_post_init__(self) -> None:
        # Stack parameters #
        ####################

        # TODO: let's make this more seamless
        self.cell = Cell.from_dict({"cell_area": self.cell_area})

        self.fit_params = self.create_polarization()

        # Stack dynamics #
        ##################

        # If the time step is bigger than the 1st order time constant, ignore dynamics
        if self.dt > self.tau:
            self.ignore_dynamics = True

        # Remove turn on delay for large time steps
        if self.dt > 2 * self.base_turn_on_delay:
            self.turn_on_delay = 0
        else:
            self.turn_on_delay = self.base_turn_on_delay

        self.turn_on_time = 0
        self.turn_off_time = -self.turn_on_delay

        self.wait_time = np.min(
            [
                (self.turn_on_time - self.turn_off_time),
                self.turn_on_delay,
            ]
        )

        # [kW] nameplate power rating
        self.stack_rating_kW = self.stack_rating_kW or self.calc_stack_power(
            self.max_current
        )

        self.stack_rating = self.stack_rating_kW * 1e3  # [W] nameplate rating

        # [W] cannot operate at less than 10% of rated power
        self.min_power = self.min_power or (0.1 * self.stack_rating)
        # self.h2_pres_out = 31  # H2 outlet pressure (bar)

        self.DTSS = self.calc_state_space()

    def run(self, P_in):
        """
        P_in [Wdc]: stack power input
        return :: H2_mfr [kg/s]: hydrogen mass flow rate
        return :: H2_mass_out [kg]: hydrogen mass
        return :: power_left [W]: difference in P_in and power consumed
        """
        self.update_status()

        I = electrolyzer_model((P_in / 1e3, self.temperature), *self.fit_params)
        V = self.cell.calc_cell_voltage(I, self.temperature)

        if self.stack_on:
            power_left = P_in

            self.I = I

            if self.include_degradation_penalty:
                V += self.V_degradation

            self.update_temperature(I, V)
            self.update_degradation()
            power_left -= self.calc_stack_power(I) * 1e3
            H2_mfr = self.cell.calc_mass_flow_rate(I) * self.n_cells
            self.stack_state, H2_mfr = self.update_dynamics(H2_mfr, self.stack_state)

            H2_mass_out = H2_mfr * self.dt
            self.uptime += self.dt

        else:
            if self.stack_waiting:
                self.uptime += self.dt
                self.I = I
                self.update_temperature(I, V)
                self.update_degradation()
                power_left = 0
            else:
                power_left = P_in
                V = 0

            H2_mfr = 0
            H2_mass_out = 0

        self.cell_voltage = V
        self.voltage_history = np.append(self.voltage_history, [V])

        # check if it is an hour to decide whether to calculate fatigue
        hourly_temp = self.hourly_counter
        self.time += self.dt
        self.hourly_counter = self.time // 3600
        if hourly_temp != self.hourly_counter:
            self.hour_change = True
            self.voltage_signal = self.voltage_history
            self.voltage_history = np.array([])
        else:
            self.hour_change = False

        return H2_mfr, H2_mass_out, power_left

    # ------------------------------------------------------------
    # Polarization model
    # ------------------------------------------------------------
    def create_polarization(self):
        interval = 10.0
        currents = np.arange(0, self.max_current + interval, interval)
        pieces = []
        prev_temp = self.temperature
        for temp in np.arange(40, 60 + 5, 5):
            self.temperature = temp
            powers = self.calc_stack_power(currents)
            tmp = pd.DataFrame({"current_A": currents, "power_kW": powers})
            tmp["temp_C"] = temp
            pieces.append(tmp)
        self.temperature = prev_temp
        df = pd.concat(pieces)

        # assign initial values and solve a model
        paramsinitial = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

        # use curve_fit routine
        fitobj, fitcov = scipy.optimize.curve_fit(
            electrolyzer_model,
            (df.power_kW.values, df.temp_C.values),
            df.current_A.values,
            p0=paramsinitial,
        )

        return fitobj

    def convert_power_to_current(self, Pdc, T):
        """
        Pdc [kW]: stack power
        T [degC]: stack temperature
        return :: Idc [A]: stack current
        """
        Idc = electrolyzer_model((Pdc, T), *self.fit_params)
        return Idc

    def curtail_power(self, P_in):
        """
        P_in [kWdc]: input power
        Curtail power if over electrolyzer rating:
        """
        return np.where(P_in > self.stack_rating_kW, self.stack_rating_kW, P_in)

    def calc_fatigue_degradation(self, voltage_signal):
        """
        voltage_signal: the voltage signal from the last 3600 seconds
        return:: voltage_penalty: the degradation penalty
        """
        # based off degradation due to square waves of different frequencies
        # from results in https://iopscience.iop.org/article/10.1149/2.0231915je

        # nonzero voltage signal so that it does not double count power cycling
        voltage_signal = voltage_signal[np.nonzero(voltage_signal)]

        # rainflow counting for fatigue
        rf_cycles = rainflow.count_cycles(voltage_signal, nbins=10)
        rf_sum = np.sum([pair[0] * pair[1] for pair in rf_cycles])
        self.rf_track += rf_sum  # running sum of the fatigue value

        return self.rate_fatigue * self.rf_track

    def calc_steady_degradation(self):
        # based off degradation due to steady operation
        # from results in https://iopscience.iop.org/article/10.1149/2.0231915je

        d_s = self.d_s + self.rate_steady * self.cell_voltage * self.dt

        self.d_s = d_s
        return d_s

    def calc_onoff_degradation(self):
        # degradation due to shut downs based off the results in
        # https://iopscience.iop.org/article/10.1149/2.0421908jes/meta

        d_o = self.rate_onoff * self.cycle_count
        self.d_o = d_o
        return d_o

    def update_degradation(self):
        if self.hour_change:  # only calculate fatigue degradation every hour
            # fatigue only counts the nonzero voltage fluctuations since transition to
            # and from V = 0 are captured with on/off cycles.
            voltage_signal_nz = self.voltage_signal[np.nonzero(self.voltage_signal)]

            # to avoid a divide by zero, only proceed if there are nonzero values in the
            # voltage signal.
            if len(voltage_signal_nz) > 0:
                voltage_perc = (max(voltage_signal_nz) - min(voltage_signal_nz)) / max(
                    voltage_signal_nz
                )

                # Only penalize if more than 5% difference in voltage
                if voltage_perc > 0.05:
                    self.fatigue_history = self.calc_fatigue_degradation(
                        self.voltage_signal
                    )

        self.d_f = self.fatigue_history

        self.V_degradation = (
            self.calc_steady_degradation()
            + self.calc_onoff_degradation()
            + self.fatigue_history
        )

    def update_temperature(self, I, V):
        # placeholder
        return self.temperature

    def update_dynamics(self, H2_mfr_ss, stack_state):
        """
        H2_mfr_ss: steady state mass flow rate
        stack_state: previous mfr state
        return :: next_state: next mfr state
        return :: H2_mfr_actual: actual mfr according to the filter

        This is really just a filter on the steady state mfr from time step to time step
        """
        x_k = stack_state
        x_kp1 = self.DTSS[0] * x_k + self.DTSS[1] * H2_mfr_ss
        y_kp1 = self.DTSS[2] * x_k + self.DTSS[3] * H2_mfr_ss
        next_state = x_kp1
        H2_mfr_actual = y_kp1

        return next_state, H2_mfr_actual

    def calc_state_space(self):
        """
        Initialize the state space matrices
        """
        tau = self.tau
        dt = self.dt
        num = [1]
        den = [tau, 1]
        ss_c = tf2ss(num, den)
        ss_d = cont2discrete((ss_c[0], ss_c[1], ss_c[2], ss_c[3]), dt, "zoh")
        return [ss_d[0], ss_d[1], ss_d[2], ss_d[3]]

    def update_status(self):
        # Change the stack to be truly on if it has waited long enough
        if self.stack_on:
            return

        if self.stack_waiting:
            if (self.turn_on_time + self.wait_time) < self.time:
                self.stack_waiting = False
                self.stack_on = True

    def turn_stack_off(self):
        if self.stack_on or self.stack_waiting:
            # record turn off time to adjust waiting period
            self.turn_off_time = self.time
            self.stack_on = False
            self.stack_waiting = False
            self.cycle_count += 1

            # adjust waiting period
            self.wait_time = np.max(
                [0, self.wait_time - (self.turn_off_time - self.turn_on_time)]
            )

    def turn_stack_on(self):
        if self.stack_on:
            return

        # record turn on time to adjust waiting period
        self.turn_on_time = self.time
        self.stack_waiting = True

        # adjust waiting period
        self.wait_time = np.min(
            [
                self.wait_time + (self.turn_on_time - self.turn_off_time),
                self.turn_on_delay,
            ]
        )

    def calc_stack_power(self, Idc):
        """
        Idc [A]: stack current
        return :: Pdc [kW]: stack power
        """
        Pdc = Idc * self.cell.calc_cell_voltage(Idc, self.temperature) * self.n_cells
        Pdc = Pdc / 1000.0  # [kW]

        return Pdc

    def calc_electrolysis_efficiency(self, Pdc, mfr):
        """
        Pdc [kW]: stack power
        mfr [kg/h]: mass flow rate
        return :: eta_kWh_per_kg, eta_hhv_percent, and eta_lhv_percent: efficiencies
        """
        eta_kWh_per_kg = Pdc / mfr
        eta_hhv_percent = self.cell.hhv / eta_kWh_per_kg * 100.0
        eta_lhv_percent = self.cell.lhv / eta_kWh_per_kg * 100.0

        return (eta_kWh_per_kg, eta_hhv_percent, eta_lhv_percent)
