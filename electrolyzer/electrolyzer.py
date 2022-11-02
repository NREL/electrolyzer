"""
This module defines the Hydrogen Electrolyzer Model
"""
# TODOs
# * refine calcCellVoltage(); compare with alkaline models
# * refine convertACtoDC(); compare with empirical ESIF model
# * refine calcFaradaicEfficiency(); compare with other model
# * add a separate script to show results

import math

import numpy as np
import scipy
import pandas as pd
import rainflow
from scipy.signal import tf2ss, cont2discrete
from scipy.constants import R, physical_constants, convert_temperature


def electrolyzer_model(X, a, b, c, d, e, f):
    P, T = X
    I = a * (P**2) + b * T**2 + c * P * T + d * P + e * T + f
    return I


# Constants #
#############
F, _, _ = physical_constants["Faraday constant"]  # Faraday's constant [C/mol]
P_ATMO, _, _ = physical_constants["standard atmosphere"]


class Electrolyzer:
    def __init__(self, n_cells, cell_area, temperature, stack_rating_kW=None, dt=1):

        # Standard state -> P = 1 atm, T = 298.15 K

        # Chemical Params #
        ###################

        # If we rework this class to be more generic, we can have these be specified
        # as configuration params

        self.n = 2  # number of electrons transferred in reaction
        # E_th_0 = 1.481  # thermoneutral voltage at standard state
        self.gibbs = 237.24e3  # Gibbs Energy of global reaction (J/mol)
        self.M = 2.016  # molecular weight [g/mol]
        self.lhv = 33.33  # lower heating value of H2 [kWh/kg]
        self.hhv = 39.41  # higher heating value of H2 [kWh/kg]

        # Degradation variables #
        #########################

        # This if a flag if voltage needs to be calculated without including degradation
        self.include_degradation_penalty = True

        # fatigue value for tracking fatigue in terms of "stress cycles"
        # rainflow counting
        self.rf_track = 0

        self.V_degradation = 0  # [V] running degradation voltage penalty
        self.uptime = 0  # [s] amount of time this electrolyzer stack has been running
        self.cycle_count = 0  # numer of times the stack has been turned off
        self.fatigue_history = 0  # [V] running count of fatigue voltage penalty
        self.hourly_counter = 0
        self.hour_change = False
        self.voltage_signal = []
        self.voltage_history = []

        # Stack parameters #
        ####################
        self.n_cells = n_cells  # Number of cells
        self.cell_area = cell_area  # [cm^2] Cell active area
        self.temperature = temperature  # [C] stack temperature
        self.max_current = 2 * self.cell_area  # [A/cm^2] current density

        if stack_rating_kW is None:
            # [kW] nameplate power rating
            self.stack_rating_kW = self.calc_stack_power(
                self.max_current, self.temperature
            )
        else:
            self.stack_rating_kW = stack_rating_kW
        self.stack_rating = self.stack_rating_kW * 1e3  # [W] nameplate rating

        # [W] cannot operate at less than 10% of rated power
        self.min_power = 0.1 * self.stack_rating
        # self.h2_pres_out = 31                     # H2 outlet pressure (bar)

        # 10 minute startup procedure
        self.stack_on = False
        self.stack_waiting = False  # going through startup procedure

        # [s] 10 minute time delay for PEM electrolyzer startup procedure
        self.turn_on_delay = 600
        self.turn_on_time = 0  # keep track of when the stack was last turned on
        self.turn_off_time = -1000  # keep track of when the stack was last turned off
        self.wait_time = self.turn_on_delay  # wait time for partial startup procedure

        # Stack dynamics #
        ##################
        self.dt = dt  # [s] simulation time step
        self.time = 0  # [s] total time of simulation
        self.tau = 5  # [s] time constant https://www.sciencedirect.com/science/article/pii/S0360319911021380 section 3.4 # noqa
        self.DTSS = self.calculate_state_space()
        self.stack_state = 0.0

        # create a polarization curve
        self.fit_params = self.create_polarization()

    def curtail_wind_power(self, P_in, T):
        """
        P_in [kWdc]: input power
        T [degC]: stack temperature
        Curtail Wind Power if over electrolyzer rating:
        """
        self.P_in = P_in
        self.P_in = np.where(
            self.P_in > self.stack_rating_kW, self.stack_rating_kW, self.P_in
        )

    # ------------------------------------------------------------
    # Polarization model
    # ------------------------------------------------------------
    def create_polarization(self):
        interval = 10.0
        currents = np.arange(0, self.max_current + interval, interval)
        pieces = []
        for temp in np.arange(40, 60 + 5, 5):
            powers = self.calc_stack_power(currents, temp)
            tmp = pd.DataFrame({"current_A": currents, "power_kW": powers})
            tmp["temp_C"] = temp
            pieces.append(tmp)
        df = pd.concat(pieces)

        fit_params = self.get_polarization_fits(
            df.power_kW.values, df.current_A.values, df.temp_C.values
        )

        return fit_params

    def get_polarization_fits(self, P, I, T):
        """
        P [kWdc]: stack power
        I [Adc]: stack current
        T [degC]: stack temperature
        return :: fitobj: fit object containing coefficients
        """

        # assign initial values and solve a model
        paramsinitial = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

        # use curve_fit routine
        fitobj, fitcov = scipy.optimize.curve_fit(
            electrolyzer_model, (P, T), I, p0=paramsinitial
        )

        return fitobj

    # ------------------------------------------------------------
    # H2 production
    # ------------------------------------------------------------

    def run(self, P_in):
        """
        P_in [Wdc]: stack power input
        return :: H2_mfr [kg/s]: hydrogen mass flow rate
        return :: H2_mass_out [kg]: hydrogen mass
        return :: power_left [W]: difference in P_in and power consumed
        """
        if self.stack_on:
            power_left = P_in

            I = electrolyzer_model((P_in / 1e3, self.temperature), *self.fit_params)
            V = self.calc_cell_voltage(I, self.temperature)
            self.temperature = self.update_temperature(I, V, self.temperature)
            self.update_degradation()
            power_left -= self.calc_stack_power(I, self.temperature) * 1e3
            H2_mfr = self.calc_mass_flow_rate(I)
            self.stack_state, H2_mfr = self.update_dynamics(H2_mfr, self.stack_state)

            H2_mass_out = H2_mfr * self.dt
            self.uptime += self.dt

        else:
            if self.stack_waiting:
                self.uptime += self.dt
                power_left = 0
            else:
                power_left = P_in

            H2_mfr = 0
            H2_mass_out = 0
            V = 0  # TODO: Should we adjust this for waiting period for degradation?

        self.voltage_history.append(V)

        # check if it is an hour to decide whether to calculate fatigue
        hourly_temp = self.hourly_counter
        self.time += self.dt
        self.hourly_counter = self.time // 3600
        if hourly_temp != self.hourly_counter:
            self.hour_change = True
            self.voltage_signal = np.squeeze(
                np.array(self.voltage_history, dtype="float")
            )
            self.voltage_history = []
        else:
            self.hour_change = False

        self.check_status()

        return H2_mfr, H2_mass_out, power_left

    def calc_reversible_voltage(self):
        """
        Calculates reversible cell voltage (open circuit voltage) at standard state.
        """
        return self.gibbs / (self.n * F)

    def calc_cell_voltage(self, I, T):
        """
        I [Adc]: stack current
        T [degC]: stack temperature
        return :: V_cell [Vdc/cell]: cell voltage
        """
        T_K = convert_temperature([T], "C", "K")

        # Cell reversible voltage:
        E_rev_0 = self.calc_reversible_voltage()
        p_anode = P_ATMO  # (Pa) assumed atmo
        p_cathode = P_ATMO

        # noqa: E501
        # Arden Buck equation T=C, https://www.omnicalculator.com/chemistry/vapour-pressure-of-water#vapor-pressure-formulas # noqa
        p_h2O_sat = (
            0.61121 * math.exp((18.678 - (T / 234.5)) * (T / (257.14 + T)))
        ) * 1e3  # (Pa)

        # General Nernst equation
        E_rev = E_rev_0 + ((R * T_K) / (self.n * F)) * (
            np.log(
                ((p_anode - p_h2O_sat) / P_ATMO)
                * math.sqrt((p_cathode - p_h2O_sat) / P_ATMO)
            )
        )

        # Activation overpotential:
        # Option 1:

        # constants below assumed from https://www.sciencedirect.com/science/article/pii/S0360319916318341?via%3Dihub # noqa

        # TODO: updated with realistic anode temperature? 70-80 C nominal operating
        # temperature 58C
        T_anode = T_K

        T_cathode = T_K  # TODO: updated with realistic anode temperature?

        # anode charge transfer coefficient TODO: is this a realistic value?
        alpha_a = 2

        # cathode charge transfer coefficient TODO: is this a realistic value?
        alpha_c = 0.5

        # anode exchange current density TODO: update to be f(T)?
        i_0_a = 2e-7

        # cathode exchange current density TODO: update to be f(T)?
        i_0_c = 2e-3

        i = I / self.cell_area

        # derived from Butler-Volmer eqs
        V_act_a = ((R * T_anode) / (alpha_a * F)) * np.arcsinh(i / (2 * i_0_a))
        V_act_c = ((R * T_cathode) / (alpha_c * F)) * np.arcsinh(i / (2 * i_0_c))

        # alternate equations for Activation overpotential
        # Option 2: Dakota: I believe this may be more accurate, found more
        # frequently in lit review
        # https://www.sciencedirect.com/science/article/pii/S0360319918309017

        # z_a = 4 # stoichiometric coefficient of electrons transferred at anode
        # z_c = 2 # stoichometric coefficient of electrons transferred at cathode
        # i_0_a = 10**(-9) # anode exchange current density TODO: update to be f(T)?
        # i_0_c = 10**(-3) # cathode exchange current density TODO: update to be f(T)?

        # V_act_a = ((R*T_anode)/(alpha_a*z_a*F)) * np.log(i/i_0_a)
        # V_act_c = ((R*T_cathode)/(alpha_c*z_c*F)) * np.log(i/i_0_c)

        # Ohmic overpotential:

        # pulled from https://www.sciencedirect.com/science/article/pii/S0360319917309278?via%3Dihub # noqa
        # TODO: pulled from empirical data, is there a better eq?
        lambda_nafion = ((-2.89556 + (0.016 * T_K)) + 1.625) / 0.1875
        t_nafion = 0.03  # (cm) TODO: confirm actual thickness?

        # TODO: confirm with Nel, is there a better eq?
        sigma_nafion = ((0.005139 * lambda_nafion) - 0.00326) * math.exp(
            1268 * ((1 / 303) - (1 / T_K))
        )
        R_ohmic_ionic = t_nafion / sigma_nafion

        # TODO: confirm realistic value with Nel https://www.sciencedirect.com/science/article/pii/S0378775315001901 # noqa
        R_ohmic_elec = 50e-3

        # Alternate R_ohmic_elec from https://www.sciencedirect.com/science/article/pii/S0360319918309017 # noqa
        # rho =  (ohm*m) material resistivity
        # l_path = (m) length of electron path
        # A_path = (m2) cross-sectional area of conductor path
        # R_ohmic_elec = ((rho*l_path)/A_path)

        V_ohmic = i * (R_ohmic_elec + R_ohmic_ionic)

        # Concentration overpotential:
        # TODO: complete this section
        # Option 1:
        # https://www.sciencedirect.com/science/article/pii/S0360319918309017
        # equations governing concentration losses / diffusion are pretty complex;
        # hoping to get an approx value or eqs from Kaz / Nel for concentration of O2
        # and H2 at electrodes else can add in equations from this paper or others to
        # get into diffusion.
        # C_an_mem_o2 = TODO: complete with equations or can we use approx values?
        # C_an_mem_o2_0 = TODO: complete with equations or can we use approx values?
        # C_cat_mem_h2 = TODO: complete with equations or can we use approx values?
        # C_cat_mem_h2_0 = TODO: complete with equations or can we use approx values?

        # V_con = ((((R*T_K)/(4*F))*np.log(C_an_mem_o2/C_an_mem_o2_0)) + (((R*T_K)/(4*F))*np.log(C_cat_mem_h2/C_cat_mem_h2_0))) # noqa

        # Option 2:
        # PEM Fuel Cell Modeling and simulation using MATLAB ISBN 978-0-12-374259-9
        # (saved in H2@scale teams>Lit Review>Fuel Cells folder)

        # Similar relationship with -log(1-i/i_L) found here
        # https://doi.org/10.1016/j.jclepro.2020.121184

        # i_L = #limiting current density TODO: get value or eq from Nel / Kaz?
        # V_con = ((R*T_K)/(self.n*F))*np.log((i_L/(i_L-i)))

        # Cell / Stack voltage:
        V_cell = E_rev + V_act_a + V_act_c + V_ohmic

        if self.include_degradation_penalty:
            V_cell += self.V_degradation

        return V_cell

    def calc_stack_power(self, Idc, T):
        """
        Idc [A]: stack current
        T [degC]: stack temperature
        return :: Pdc [kW]: stack power
        """
        Pdc = Idc * self.calc_cell_voltage(Idc, T) * self.n_cells  # [W]
        Pdc = Pdc / 1000.0  # [kW]

        return Pdc

    def convert_power_to_current(self, Pdc, T):
        """
        Pdc [kW]: stack power
        T [degC]: stack temperature
        return :: Idc [A]: stack current
        """
        Idc = electrolyzer_model(
            (Pdc, T),
            self.fit_params[0],
            self.fit_params[1],
            self.fit_params[2],
            self.fit_params[3],
            self.fit_params[4],
            self.fit_params[5],
        )
        return Idc

    # ------------------------------------------------------------
    # Post H2 production
    # ------------------------------------------------------------
    def calc_faradaic_efficiency(self, I):
        """
        I [A]: stack current
        return :: eta_F [-]: Faraday's efficiency
        Reference: https://res.mdpi.com/d_attachment/energies/energies-13-04792/article_deploy/energies-13-04792-v2.pdf
        """  # noqa
        f_1 = 250  # (mA2/cm4)
        f_2 = 0.996
        i_cell = I * 1000

        eta_F = (
            ((i_cell / self.cell_area) ** 2) / (f_1 + ((i_cell / self.cell_area) ** 2))
        ) * f_2

        return eta_F

    def calc_electrolysis_efficiency(self, Pdc, mfr):
        """
        Pdc [kW]: stack power
        mfr [kg/h]: mass flow rate
        return :: eta_kWh_per_kg, eta_hhv_percent, and eta_lhv_percent: efficiencies
        """
        eta_kWh_per_kg = Pdc / mfr
        eta_hhv_percent = self.hhv / eta_kWh_per_kg * 100.0
        eta_lhv_percent = self.lhv / eta_kWh_per_kg * 100.0

        return (eta_kWh_per_kg, eta_hhv_percent, eta_lhv_percent)

    def calc_mass_flow_rate(self, Idc, dryer_loss=6.5):
        """
        Idc [A]: stack current
        dryer_loss [%]: loss of drying H2
        return :: mfr [kg/s]: mass flow rate
        """
        eta_F = self.calc_faradaic_efficiency(Idc)
        mfr = (
            eta_F
            * Idc
            * self.M
            * self.n_cells
            / (self.n * F)
            * (1 - dryer_loss / 100.0)
        )  # [g/s]
        # mfr = mfr / 1000. * 3600. # [kg/h]
        mfr = mfr / 1e3  # [kg/s]
        return mfr

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
        if not self.stack_on:
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

    def check_status(self):
        # Change the stack to be truly on if it has waited long enough
        if self.stack_on:
            return
        else:
            if self.stack_waiting:
                if (self.turn_on_time + self.wait_time) < self.time:
                    self.stack_waiting = False
                    self.stack_on = True

    def update_temperature(self, I, V, temp):
        # placeholder
        return temp

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

    def calculate_state_space(self):
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

    def calculate_fatigue_degradation(self, voltage_signal):
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

        # below: these numbers are the lowest cataylst loading (most fragile)
        # A = 0.2274592919410412
        # B = 0.10278876731577287

        # below: these numbers are the highest catalyst loading (least fragile)
        # A = 0.22746397778732894
        # B = 0.06116270762621622

        # below: these numbers are based off a 40000 hr lifetime of steady operation
        # decreased by 1/3 due to renewable energy signal
        A = 0.22746397778732844
        B = 0.0070957975662638405

        voltage_penalty = max([0, B * self.rf_track**A])

        return voltage_penalty

    def calculate_steady_degradation(self):
        # https://www.researchgate.net/publication/263092194_Investigations_on_degradation_of_the_long-term_proton_exchange_membrane_water_electrolysis_stack # noqa
        # steady_deg_rate = 35.5e-6 # (microvolts / hr)

        # (volts / hr) most fragile membrane loading from Alia et al 2019
        # steady_deg_rate = 0.000290726819047619

        # this is set as constant now but should be changed dynamically in the future
        operating_voltage = 2

        # lowest catalyst loading steady degradation rate
        # steady_deg_rate = 2.80278563e-08 * operating_voltage

        # highest catalyst loading steady degradation rate [V/(s V)*(V)]
        steady_deg_rate = 1.12775521e-09 * operating_voltage
        # ^ these are in units of [V/s]

        # return steady_deg_rate*self.uptime/(60*60)
        return steady_deg_rate * self.uptime

    def calculate_onoff_degradation(self):
        # This is a made up number roughly equal to operating at steady for 1 day
        # onoff_rate = 0.006977443657142856 # (volts / cycle)

        # This is a made up number roughly equal to operating at steady for 1 day for
        # highest catalyst loading steady
        onoff_rate = 0.00019487610028800001  # (volts / cycle)

        return onoff_rate * self.cycle_count

    def update_degradation(self):
        # scaling factors to manually change the rates of degradation
        steady_factor = 1
        onoff_factor = 1
        fatigue_factor = 1

        if self.hour_change:  # only calculate fatigue degradation every hour
            voltage_perc = (max(self.voltage_signal) - min(self.voltage_signal)) / max(
                self.voltage_signal
            )
            # Don't penalize more than 5% difference in voltage
            if voltage_perc > 0.05:
                # I think this should be just a normal = not a +=
                self.fatigue_history = self.calculate_fatigue_degradation(
                    self.voltage_signal
                )

        self.V_degradation = (
            steady_factor * self.calculate_steady_degradation()
            + onoff_factor * self.calculate_onoff_degradation()
            + fatigue_factor * self.fatigue_history
        )
