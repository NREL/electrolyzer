
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

    def __init__(self, electrolyzer_specs,electrolyzer_perf_dict, turb_dict):
# # -------------- Total Plant Info ------------- #
        self.stack_rating_kW = electrolyzer_specs['stack_rating_kW']  #[kW] single stack rating
        self.n_stacks = electrolyzer_specs['n_stacks'] #number of stacks
        self.stack_min_pwr_kW=electrolyzer_specs['stack_min_pwr_kW'] #min power per stack
        self.cntrl_strat=electrolyzer_specs['control_strat'] #unused right now
#wind_in -out npz: TODO FIX THIS
         #or should this be time-series? Give user option to estimate

#kg_tot_sum (single val) -out npz
        self.tot_h2_prod=electrolyzer_perf_dict['tot_h2_prod'] #kg_tot_sum -> single val
        self.pwr_to_stacks_tot_kW=electrolyzer_perf_dict['tot_power_in_kw']
        self.stack_degradation_per_dt=electrolyzer_perf_dict['deg_per_stack']
        self.power_per_stack_W=electrolyzer_perf_dict['power_per_stack'] #/1000
        self.all_stack_info=electrolyzer_perf_dict['all_stack_info'] #unused right now

# # -------------- Individual Stack Info Per Simulation ------------- #
#uptime per stack [sec] -out npz
        self.uptime_per_stack=electrolyzer_perf_dict['uptime_per_stack_sec'] #in [s] as uptime per-stack over entire sim
        self.data_length=electrolyzer_perf_dict['sim_length_sec'] #actually in sec -> len(pz['wind_in']/(60*60))
        

        self.turb_rating_kw=turb_dict['turbine_rating_kW'] #could be added to a dict
        self.turb_name=turb_dict['turbname'] #TODO - ADD stuff for this to be exact
        self.num_turbs=turb_dict['num_turbs'] #calculated in ... control runs?
        #self.turb_aep=turb_dict['aep']
        #input a time-series power signal?
        
        self.pem_location=turb_dict['pem_loc']
        #TODO add capability to include weibull distribution params :)
        #TODO add output dict so the final numbers are accessible!

        self.turb_aep=turb_dict['aep']
        self.tot_h2_prod=electrolyzer_perf_dict['tot_h2_prod'] 
        self.assumed_constants()
        self.prelim_calc()
    def assumed_constants(self):
        #self.hrs_per_year=8760
        self.electrolyzer_elec_consump=55.5 #[kWh/kg] assumed from ... HFTO? Also NREL 2019 Slide 12
        self.dr = 0.04 # Discount Rate - from LCOH with PV, sometimes assumed to be 0.05 (DTU 2021)
        self.stack_life_hrs = 80000 # ASSUMPTION [hrs] ranges between 80000 and 85000 for current PEM
        self.IF=0.33 #[%] Install Factor from DTU 2021 and others...
        self.plant_life_hrs=25*8760 #assuming 25 year operational time TODO: make this a minimum of turb life or BOP life
        self.sec_per_yr=8760*3600
    def prelim_calc(self):
        # TODO estimate annual % of stack operation based on not year-long data
        self.active_stack_perc=self.uptime_per_stack/self.data_length
        self.operational_hours_life=max(self.active_stack_perc)*self.stack_life_hrs #based on max for all stacks
        #^ would like to update this to be per-stacks

        self.plant_rating = self.stack_rating_kW*self.n_stacks # electrolyzer plant rating[kW]
        #self.max_energy_gen=self.turb_rating_kw*self.num_turbs*8760 # [kWh]
        #self.TurbCapFac=self.turb_aep/self.max_energy_gen #This doesn't need to be used
        self.turb_COE= 44.18*(1/1000) #TODO -> MAKE THIS DYNAMIC AND TURB DEPENDENT

        #TODO Add capability to scale up production/consumption etc for not a full years worth of data

        if self.pem_location == 'onshore':
            self.DTU_os=0
        else:
            self.DTU_os=1
        
        
        if self.turb_name=='IEA':
            #IEA 3.4 MW Spec Sheet - Table 2 of
            self.dt_gen_eff=0.936 #drive-train efficiency
            self.turb_COE= 44.18*(1/1000) # [$/kWh] from 44.18 $/MWh 
            
            self.v_in=4 #[m/s] cut-in wind speed
            self.v_out=25 #[m/s] cut-out wind speed
            self.v_rated=9.8 #rated wind speed [m/s]
            self.cp_max=0.481 #max power coeff
            self.tsr_op=8.16 #optimal tip-speed ratio
            self.rot_rad=130/2 #[m]
            self.Weib_shape=1.85 #weibull shape parameter only used if no annual data is given
  

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

        PEM_elec_consump=self.turb_aep #[kWh] #TODO UPDATE THIS
        elec_feedstock_cost=PEM_elec_consump*self.turb_COE #[$]
        #water_cost_per_kg_h2=0.08 #[$/kg of H2] from ICCT 2020
        #h20_feedstock_cost=water_cost_per_kg_h2*annual_h2_prod
        stack_replacement_cost=self.estimate_stack_costs() 
        #^^annual replacements based on degradation - maybe untrustworthy at the moment and very conservative
       
        OpEx_II=OM_year+h20_feedstock_cost+elec_feedstock_cost+stack_replacement_cost
        return OpEx_II
    def estimate_stack_costs(self): 
        '''This is a linear approximation of stack replacement costs based on the assumption that 
            1. degradation rate is linear and calculated from time-series performance
            2. when total stack voltage degradation = 0.7 then stack needs to be replaced (based on info from ZT)
            3. Stack replacement is 15% of CapEx'''
        # NOTE Stack replacement should be 15% of installed CapEx (not uninstalled CapEx)

        # ESG TODO: estimate stack replacement based on data of wind variation, mean wind speed, and relationship w degradation
        #https://iopscience.iop.org/article/10.1149/2.0421908jes - degradation and cycling
        approx_deg_rate=np.zeros(self.n_stacks)
        hrs_til_dead=np.zeros(self.n_stacks)
        
        #deg_limit=0.71 #[V] point where stack has to be replaced based on info from Zack
        deg_limit=0.71*self.all_stack_info[1].n_cells #assuming all stacks have same number of cells
        for n in range(0,self.n_stacks,1):
           
           approx_deg_rate[n]=max(self.stack_degradation_per_dt[n])/self.uptime_per_stack[n]
           hrs_til_dead[n]=(1/approx_deg_rate[n])*deg_limit*(1/3600) #s/V * V = [s] * hr/13600sec= hr
           

        annual_replacement_per_stack=8760/hrs_til_dead[:-1]
        tot_annual_replacements=sum(annual_replacement_per_stack)
        avg_annual_replacement_plant=np.mean(annual_replacement_per_stack)
        annual_cost_for_replacement=0.15*(self.PEM_CapEx_calc()/(1+self.IF*self.DTU_os))*tot_annual_replacements #1 replacement = 15% of uninstalled CapEx
  
        lifetime_replacements_stack=sum(self.plant_life_hrs/hrs_til_dead[:-1])
        
        
        return annual_cost_for_replacement

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
            yearly_opex_and_energy[y]=(self.turb_COE*self.turb_aep + O_m_I)/((1+self.dr)**y)
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

    






if __name__=="__main__": #complete this based off of most recent info
    MyDir = os.getcwd()
    tempdr=MyDir + '/h2atscale-milestone-Q2_snapshot/analysis/Q2_milestone/results/IEA_6ms/baseline_deg/'
    elec_pickle=pd.read_pickle(tempdr+"elec_system")
    elec_npz=np.load(tempdr+"elec_outputs.npz")
    
    electrolyzer_specs={}

    electrolyzer_specs['stack_rating_kW'] = elec_pickle.stack_rating_kW #single stack rating
    electrolyzer_specs['n_stacks'] = elec_pickle.n_stacks #number of stacks
    electrolyzer_specs['stack_min_pwr_kW'] = elec_pickle.stack_min_power/1000 #[kW]
    electrolyzer_specs['control_strat'] = elec_pickle.control_type #for saving info :)
    #elec_pickle.stacks[1].V_degradation

    electrolyzer_perf_dict={}
    electrolyzer_perf_dict['tot_h2_prod']=elec_npz['kg_tot_sum']
    electrolyzer_perf_dict['uptime_per_stack_sec']=elec_npz['uptime'] #in [sec]
    electrolyzer_perf_dict['sim_length_sec'] =len(elec_npz['wind_in']) #
    electrolyzer_perf_dict['tot_power_in_kw']=(elec_npz['wind_in'] - elec_npz['wind_curtailment'])/1000
    electrolyzer_perf_dict['power_per_stack']=elec_pickle.P_indv_store #[W] rows are sec, columns are stack numbers
    electrolyzer_perf_dict['deg_per_stack']=elec_npz['degradation'] #[V] rows are sec, columns are stack numbers
    electrolyzer_perf_dict['all_stack_info']=elec_pickle.stacks

    turb_dict={}
    turb_dict['turbine_rating_kW']=3400
    turb_dict['turbname']='IEA'
    turb_dict['num_turbs'] = 1
    turb_dict['pem_loc'] ='onshore'#'onshore', 'offshore', 'inturbine'
    turb_dict['aep'] =3676806.633 #REMOVE
    turb_dict['scale-up']=False #intend for this to be added


    #------------------ If annual data --------- #
    electrolyzer_specs = {}
    electrolyzer_specs['stack_rating_kW']  = 1000
    electrolyzer_specs['n_stacks'] = 1
    
    electrolyzer_perf_dict={}
    electrolyzer_perf_dict['hr_h2_kg']=[]
    electrolyzer_perf_dict['tot_h2_prod']=61015.122
    electrolyzer_perf_dict['hr_count']=8760
    electrolyzer_perf_dict['uptime_per_stack']=7060 #temp num

    turb_dict={}
    turb_dict['turbine_rating']=1000
    turb_dict['turbname']='IEA' #TODO - ADD stuff for this to be exact
    turb_dict['num_turbs'] = 1
    turb_dict['aep'] =3676806.633 #[kWh]
    turb_dict['pem_loc']='onshore'

    cost=PEM_lcoh(electrolyzer_specs,electrolyzer_perf_dict,turb_dict)

    dtu_based_lcoh=cost.LCOH_Calc_I()
    print('DTU Based Method for LCOH [$/kg H2]:'.format(dtu_based_lcoh))
    conservative_replacement_cost_lcoh=cost.LCOH_Calc_II()
    print('Othjer Method for LCOH [$/kg H2]:'.format(conservative_replacement_cost_lcoh))

            #cost_per_year.append(self.turb_COE*self.turb_aep) 
        #assuming that stacks consume all turbine power -> TODO REMOVE THIS
        #Should it be power sent to stack or power consumed by stack?




# This is just a bunch-o random notes -----


'''
 NREL 2014: https://www.energy.gov/sites/prod/files/2014/08/f18/fcto_2014_electrolytic_h2_wkshp_colella1.pdf
   -> Uses 2012 USD for production costs [Slide 8]
   -> uses 2007 USD for some numbers
   -> 3% dryer loss [slide 15]
   -> info used here is for "current forecourt" cases so 2014 year and 1500kg H2/day plant capacity
   -> nominal efficiency is 55.5 kWh/kg
'''

'''''
DRO 2021: https://dro.dur.ac.uk/33454/1/33454.pdf
-> look at mass manufacturing of PEM to reduce capital cost
-> Assumed 20% BOP reduction for each 10x additional units
-> Looked at potential tech available/advancements for 2030
-> has detailed info on component costs for stack
'''

'''
PNNL https://www.pnnl.gov/sites/default/files/media/file/Hydrogen_Methodology.pdf
-> 100 MW HESS system
-> Table 4 and Table 6 has O&M costs
'''

'''
 NREL 2019: https://www.nrel.gov/docs/fy19osti/73481.pdf
   -> Electrolyzer cost is 737 $/kW [slide 12]
'''

'''
ICCT 2020: https://theicct.org/wp-content/uploads/2021/06/final_icct2020_assessment_of-_hydrogen_production_costs-v2.pdf
'''


# # IEA 3.4 MW turb Table 2 of -> https://www.nrel.gov/docs/fy19osti/73492.pdf

#DTU 2021 assumes water cost of 9.23 euro/m^3 with 60% discount rate ...
#electricity consumption and cost
#citation 11 from ICCT 2020 says OpEx = 1-3% of CapEx 
#citation 12 from ICCT 2020 says Fixed OpEx = $40/kW 
#ICCT 2020 says Variable OpEx = cost of electricity and cost of water - page 18

# --------- Replacement Costs Based on Degradation ---------
#need to determine how much energy is used for H2 production -> Power_left=Pin-Pconsumed
#need to determine when stack replacement has to happen based on total degradation [V]
#Does avg_power_kW_per_hr include how much power the electrolyzer is actually using (which would include voltage degradation?)

'''
From IRENA Page 40
IRENA assumes 20: 1 ratio of h20: h2
        # ratio of water_used:h2_kg_produced depends on power source
        # h20_kg:h2_kg with PV 22-126:1 or 18-25:1 without PV but considering water deminersalisation
        # stoichometrically its just 9:1 but ... theres inefficiencies in the water purification process
        max_water_feed_mass_flow_rate_kg_hr = 411  # kg per hour
https://www.researchgate.net/publication/323253474_GREEN_CHEMISTRY_AND_TECHNOLOGY_OF_PLANT_BIOMASS Table 2 has water feedstock cost as 0.7-0.9 $/m^3
1m^3 H20 = 1000 kg
'''

