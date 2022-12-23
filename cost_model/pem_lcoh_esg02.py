
import sys
from distutils.util import execute
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import math
import scipy
from mpl_toolkits import mplot3d
#from electrolyzer_supervisor import ElectrolyzerSupervisor
'''
Sources: 
- DTU 2021 Report: https://www.sciencedirect.com/science/article/pii/S2667095X21000052
- IEA 3.4 Spec Sheet: https://www.nrel.gov/docs/fy19osti/73492.pdf 
- Joule TEA 2021 Supplemental: https://ars.els-cdn.com/content/image/1-s2.0-S2542435121003068-mmc1.pdf 
- PNNL 2020 : https://www.pnnl.gov/sites/default/files/media/file/Hydrogen_Methodology.pdf 
- IRENA 2020 : https://www.irena.org/publications/2020/Dec/Green-hydrogen-cost-reduction
- Hydrogen Council 2020: https://hydrogencouncil.com/wp-content/uploads/2020/01/Path-to-Hydrogen-Competitiveness_Full-Study-1.pdf
- DRO 2021: https://dro.dur.ac.uk/33454/1/33454.pdf
- ICCT 2020: https://theicct.org/wp-content/uploads/2021/06/final_icct2020_assessment_of-_hydrogen_production_costs-v2.pdf 
- NREL 2019: https://www.nrel.gov/docs/fy19osti/73481.pdf
- LCOH with PV: https://onlinelibrary.wiley.com/doi/full/10.1002/solr.202100482
'''
class PEM_lcoh():

    def __init__(self, annual_data_dict): #,electrolyzer_perf_dict, turb_dict):
# # -------------- Total Plant Info ------------- #
        self.stack_rating_kW = annual_data_dict['Single Stack Size [kW]']  #[kW] single stack rating
        self.n_stacks = annual_data_dict['Number of Stacks'] #number of stacks

        #self.stack_min_pwr_kW=electrolyzer_specs['stack_min_pwr_kW'] #REMOVEmin power per stack
        #self.cntrl_strat=electrolyzer_specs['control_strat'] #REMOVE unused right now


        self.tot_h2_prod=annual_data_dict['Total H2 Production [kg]'] #kg_tot_sum -> single val
        self.pem_location=annual_data_dict['PEM Location']
        #option 1
        self.turb_COE=annual_data_dict['Cost of Energy [$/kWh]']
        self.PEM_energy_consump=annual_data_dict['Plant Energy Consumption [kWh/year]']
        #option 2
        self.elec_feedstock_cost=annual_data_dict['Total Annual Cost of Energy [$]']

        #option 1
        self.all_stack_op_hrs=annual_data_dict['Annual Equivalent Hours of Operation for all Stacks']
        #option 2
        self.num_stack_replacements=annual_data_dict['Total number of stack replacements']

        self.assumed_constants()
        self.prelim_calc()
    def assumed_constants(self):
        #self.hrs_per_year=8760
        self.electrolyzer_elec_consump=55.5 #[kWh/kg] assumed from ... HFTO? Also NREL 2019 Slide 12
        self.dr = 0.04 # Discount Rate - from LCOH with PV, sometimes assumed to be 0.05 (DTU 2021)
        self.stack_life_hrs = 80000 # ASSUMPTION [hrs] ranges between 80000 and 85000 for current PEM
        self.IF=0.33 #[%] Install Factor from DTU 2021 and others...
        self.plant_life_hrs=25*8760 #assuming 25 year operational time TODO: make this a minimum of turb life or BOP life
        
    def prelim_calc(self):
        # TODO estimate annual % of stack operation based on not year-long data
        #self.active_stack_perc=self.uptime_per_stack/self.data_length
        self.active_stack_perc=self.all_stack_op_hrs/(self.n_stacks*8760) #[%] hourly operation per stack
        self.operational_hours_life=max(self.active_stack_perc)*self.stack_life_hrs #based on max for all stacks
        #^ would like to update this to be per-stacks

        self.plant_rating = self.stack_rating_kW*self.n_stacks # electrolyzer plant rating[kW]

        if self.pem_location == 'onshore':
            self.DTU_os=0
        else:
            self.DTU_os=1
        
        if len(self.elec_feedstock_cost)==0:
            self.elec_feedstock_cost=self.turb_COE*self.PEM_energy_consump
        else:
            self.elec_feedstock_cost=self.elec_feedstock_cost #probably repetitive
        
        

        
  
    def PEM_CapEx_calc(self): 
        #This method is from DTU 2021 Report [see appendix]
        learning_rate=0.13 #FROM DTU 2021, Hydrogen Council 2020, DRO 2021, 
        #CapEx_IF=0.33 # [%] Install Factor from DTU 2021 and others...
        self.b_capex=np.log2(1-learning_rate) #b = -0.2009

        Ref_Cost_BOP=747 # [$/kW] for 200 kW electrolyzer BOP breakdown cost from DRO 2021.  IRENA has in range 300-1000 $/kW for 1MW electrolyzer
        Ref_Power_BOP=200 #[kW]

        self.Ref_Cost_PEM=695 # [$/kW] for 2MW PEM from DRO 2021.  IRENA has 400 $/kW for 2020 1MW electrolyzer
        self.Ref_Power_PEM=2000 #[kW]

        CapEx_BOP=self.plant_rating * Ref_Cost_BOP * (1+self.IF*self.DTU_os) * ((self.plant_rating/Ref_Power_BOP)**self.b_capex) # [$]
        CapEx_PEM=self.plant_rating * self.Ref_Cost_PEM * (1+self.IF*self.DTU_os) * ((self.plant_rating/self.Ref_Power_PEM)**self.b_capex) # [$]

        CapEx_Tot=CapEx_BOP + CapEx_PEM #[$]
        return CapEx_Tot
    def PEM_OpEx_calc_I(self):
        # This method is from the DTU 2021 Report [see appendix]
        # Includes equipment costs, non-equipment costs, and stack replacement cost estimation
        # TODO check hard-coded numbers with U.S. data
        b_opex=-0.155 #from DTU 2021
        CapEx_Tot=self.PEM_CapEx_calc()
        #Equipment OpEx: DTU 2021 - 0.0344 is a percent, unsure where from
        OpEx_Equip=CapEx_Tot*(1-self.IF*(1+self.DTU_os))*0.0344*(self.plant_rating**b_opex) 

        #other opex related to site management, land cost + taxes, site maintenance, etc
        OpEx_NEquip=0.04*CapEx_Tot*self.IF*(1+self.DTU_os) # IS this like simple cash annuals?

        Sf_Sr_0=0.11 #average scale factor
        P_stack_max=2000 #[kW] Max Stack Size
        Ref_Cost_OpEx=0.41 #[%]
        Ref_Power_OpEx=5000 #[kW] = 5MW
        Sf_Sr=1-(1-Sf_Sr_0)*math.exp(-self.plant_rating/P_stack_max) #scale factor
        Rc_Sr=Ref_Cost_OpEx*self.Ref_Cost_PEM*(1-self.IF)*((Ref_Power_OpEx/self.Ref_Power_PEM)**self.b_capex) #ref cost share

        #OH/OH_Max where OH is the total number of operational hours of the lectrolyzer over the lifetime of the plant
        #OH_Max is stack operating time

        #make this per stack then sum it up instead of taking the max
         #TODO FIX THIS -> move to prelim calc
        OpEx_StackReplacement=self.plant_rating*Rc_Sr*((self.plant_rating/Ref_Power_OpEx)**Sf_Sr)*(self.operational_hours_life/self.stack_life_hrs) #TODO Fix the hours_op/stack_life_hrs 
        OpEx_tot=OpEx_Equip+OpEx_NEquip+OpEx_StackReplacement

        return OpEx_tot

    def PEM_OpEx_calc_II(self):    
        
                # ------------------- Alternative OpEx ----------------------- #
        #OpEx = water + electricity + O&M_Gen + StackReplacementCost

        Fixed_OM = 12.8 #[2018$/kW-year] Joule TEA 2021 Supplemental Table S25
        #Fixed O&M occur regardless of production -> measured per unit of capacity
        #PNNL has fixed O&M as 14.5 $/kW-year fund in Table 2 that includes 0.8 $/MWh for part replacement
        Var_OM_MWh = 1.3 #[2018$/MWh] Joule TEA 2021 Supplemental Table S25
        # Variable O&M are per unit of energy outputs -> in this case it should be power consumed by stacks
        # Joule TEA 2021: "Replacement Costs of electrolyzers are incorporated into variable operating costs as an annualized cost that is a 
        # function of the stack durability given in Table S27 and the stack fraction of the total capital cost. Variable O&M costs in
        # Table S25 do not include these replacement costs"
        Var_OM=Var_OM_MWh*8760/1000 #[2018$/kW-year] 
        #hours_op=self.active_stack_perc*self.stack_life_hrs #TODO FIX THIS -> Move to prelim calc


        OM_year=self.plant_rating*(Fixed_OM + Var_OM*(self.operational_hours_life/self.stack_life_hrs)) # VarO&M based on operational hours
        
        water_used_kg=20*self.tot_h2_prod #20kg H20 per 1 kg H2 - average from Irena

        #0.8 $/m^3 of h20 from ICCT 2020 which cites: https://www.nature.com/articles/s41560-019-0326-1
        h20_feedstock_cost=water_used_kg*0.8/1000 #[$] = kg_h20 * $/m^3 * m^3/1000 kg_h20

        # PEM_elec_consump=self.turb_aep #[kWh] #TODO UPDATE THIS
        # elec_feedstock_cost=PEM_elec_consump*self.turb_COE #[$]
        #water_cost_per_kg_h2=0.08 #[$/kg of H2] from ICCT 2020
        #h20_feedstock_cost=water_cost_per_kg_h2*annual_h2_prod
        stack_replacement_cost=0.15*(self.PEM_CapEx_calc()/(1+self.IF*self.DTU_os))*self.num_stack_replacements
        #^^annual replacements based on degradation - maybe untrustworthy at the moment and very conservative
       
        OpEx_II=OM_year+h20_feedstock_cost+self.elec_feedstock_cost+stack_replacement_cost
        return OpEx_II


    def LCOH_Calc_I(self):
        # This uses OpEx_Calc_I which includes stack replacement costs, the cost of electricity to power electrolyzers is calculated here
        plant_life_years=self.plant_life_hrs/8760
        num_years=np.arange(0,math.floor(plant_life_years),1)
        CapEx=self.PEM_CapEx_calc()
        #print(CapEx)
        
        O_m_I=self.PEM_OpEx_calc_I()
        #print(O_m_I)
        #cost_per_year=np.zeros(25)
        yearly_opex_and_energy=np.zeros(math.floor(plant_life_years))
        yearly_h2=np.zeros(math.floor(plant_life_years))
        for y in num_years:
            yearly_opex_and_energy[y]=(self.elec_feedstock_cost + O_m_I)/((1+self.dr)**y)
            yearly_h2[y]=self.tot_h2_prod/((1+self.dr)**y)
            #cost_per_year[y]=num/den
            #print(cost_per_year)
        lcoh=(CapEx+sum(yearly_opex_and_energy))/sum(yearly_h2)

        return lcoh

    def LCOH_Calc_II(self):
         # This uses OpEx_Calc_II
         # The cost of electricity to power electrolyzers is calculated within the Opex_Calc_II function (electric_feedstock_cost)
        plant_life_years=self.plant_life_hrs/8760
        num_years=np.arange(0,math.floor(plant_life_years),1)
        CapEx=self.PEM_CapEx_calc()
        
        O_m_I=self.PEM_OpEx_calc_II()
        
        yearly_opex_and_energy=np.zeros(math.floor(plant_life_years))
        yearly_h2=np.zeros(math.floor(plant_life_years))
        for y in num_years:
            yearly_opex_and_energy[y]=( O_m_I)/((1+self.dr)**y)
            yearly_h2[y]=self.tot_h2_prod/((1+self.dr)**y)
            
        lcoh=(CapEx+sum(yearly_opex_and_energy))/sum(yearly_h2)

        return lcoh

    def run(self):
        if len(self.num_stack_replacements)==0:
            print("Since annual number of stack replacements wasn't specified \
                calculating LCOH using Opex_I calculation (dependent on operational hours).\n")
            lcoh=self.LCOH_Calc_I
            
        else:
            print("Since annual number of stack replacements was specified \
                calculating LCOH using Opex_II calculation (dependent on number of stack replacements).")
            lcoh=self.LCOH_Calc_II
    
        print("LCOH is estimated to be {} [$/kg H2]: ".format(lcoh))
        return lcoh
