# required external packages
import numpy as np

class cost():
    """
    Keyword arguments:
    lcoe : array of length `lifetime`
        levelized cost of energy provided to the electrolyzer for each year (USD/kW)
    energy : array of length `lifetime`
        the amount of energy supplied to the electrolyzer (kW)
    hydrogen_produced : array of length `lifetime`
        amount of hydrogen produced by the electrolyzer (kg)
    lifetime : int
        the life of the project (years)
    """
    def __init__(self, lcoe, energy, hydrogen_produced, electrolyzer_type="PEM", lifetime=30):
        self.electrolyzer_type = electrolyzer_type # right now just "PEM", but could include other types in the future
        self.energy = energy # energy supplied to the electrolyzer in kW
        self.lcoe = lcoe # levelized cost of energy
        self.hydrogen_produced = hydrogen_produced # kg
        self.lifetime = lifetime # lifetime of the plant
        self.discount_rate = 0.05
        self.rated_power_electrolyzer = 750.0 # kw - rated power of electrolyzer
        self.efficiency_electrolyzer = 0.9 # efficiency of electrolyzer
        self.power_electrolyzer = self.rated_power_electrolyzer*self.efficiency_electrolyzer # actual power of electrolyzer (kW)
        self.referene_cost_electrolyzer = 600.0*0.98 # euro/kw to USD/kw
        self.install_frac_elec = 0.33 # fraction of plant for electrolyzer
        self.Offshore = 1 # whether or not the plant is offshore (1 for offshore or in-turbine, 0 for onshore)
        self.SF_elec = -0.21 # Scale factor
        self.capex = self.capital_expenditure_electrolyzer() # capex
        self.opex = self.operational_expenditure_electrolyzer() # opex
        self.lcoh = self.levelized_cost_of_hydrogen() #lcoh

    def capital_expenditure_electrolyzer(self):
        """ Returns CAPEX for electrolyzer equipment based on Singlitico 2021 """
        capex_elec = self.power_electrolyzer*self.referene_cost_electrolyzer*(1+self.install_frac_elec*self.Offshore)* \
            (self.power_electrolyzer/self.rated_power_electrolyzer)**self.SF_elec
        return capex_elec

    def operational_expenditure_electrolyzer(self):
        opex_eq = self.operational_expenditure_electrolyzer_equipment() # for equipment
        opex_sr = 0.0 # stack replacement needs to be added
        opex_neq = 0.0 # non-equipment (e.g. land lease, tax, etc) needs to be added
        opex_electrolyzer = opex_eq + opex_sr + opex_neq # total capex for electrolyzer
        return opex_electrolyzer

    def operational_expenditure_electrolyzer_equipment(self):
        """ Returns OPEX for electrolyzer equipment based on Singlitico 2021 """
        return self.capex*0.0344*(self.power_electrolyzer)**(-0.155) # equipment costs

    def levelized_cost_of_hydrogen(self):
        """Returns the levelized cost of hydrogen (LCOH) for the given system."""

        # these should be able to be varied per year, but I have not put that in
        lcoe = np.ones(self.lifetime)*self.lcoe
        energy = np.ones(self.lifetime)*self.energy
        capex = np.ones(self.lifetime)*self.capex
        opex = np.ones(self.lifetime)*self.opex
        hydrogen_produced = np.ones(self.lifetime)*self.hydrogen_produced

        # set numerator and denominator to zero
        numerator = 0
        denominator = 0

        # sum over the life of the plant
        for i in np.arange(0, self.lifetime):
            numerator += (lcoe[i]*energy[i] + capex[i] + opex[i])/((1.0 + self.discount_rate)**i)
            denominator += hydrogen_produced[i]/((1.0 + self.discount_rate)**i)

        # get lcoh from sum ratio
        lcoh = numerator/denominator

        return lcoh