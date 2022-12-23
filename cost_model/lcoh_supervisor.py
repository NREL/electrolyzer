
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
#from pem_lcoh_esg02 import PEM_lcoh
import pem_lcoh_esg02
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
#class PEM_lcoh():
def I_Have_Annual_Data():
    annual_data_dict={}
    #COE 1
    annual_data_dict['Cost of Energy [$/kWh]']= 44.18*(1/1000) #[$/kWh]
    annual_data_dict['Plant Energy Consumption [kWh/year]']= 5 #[kWh/year]
    #COE2
    annual_data_dict['Total Annual Cost of Energy[$]']=0.448* 5 #[$/kWh]*[kWh/year] 
    #if total annual cost of power isn't specified then it's calculated from cost of energy and plant energy consumption


    annual_data_dict['Total H2 Production [kg]']=40 #[kg] of H2 produced in a year
    annual_data_dict['Single Stack Size [kW]']=200 #single stack rating
    annual_data_dict['Number of Stacks']=3 # total number of stacks
    annual_data_dict['PEM Location']='onshore' # offshore, in-turbine
    
    #stack replacement 1 - uses Opex_II_calc. Leave empty if you want to use Opex_I calc instead
    annual_data_dict['Total number of stack replacements']=2 #annual number of stack replacements (it can be a fraction)
    #stack replacement 2 - uses Opex_I calc
    annual_data_dict['Annual Equivalent Hours of Operation for all Stacks']=2500 # [hrs] annual equivalent hours of operation 
    
    lcoh_calc=pem_lcoh_esg02.PEM_lcoh(annual_data_dict)
    lcoh=lcoh_calc.run()
    #def __init__(self, electrolyzer_specs,electrolyzer_perf_dict, turb_dict):
class Estimate_Annual_Data_From_TS():

    def __init__(self, electrolyzer_specs,electrolyzer_perf_dict, turb_dict):
        '''
        Inputs: 
        -what data needs estimating
        -what info is available

        Outputs:
        -annual estimations

        '''
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

    def estimate_aep(self):
        #estimate AEP from Weibull
        #may need double checked - Weibull doesn't sum to 1
        a=self.Weib_shape #weibull shape parameter -> to be input
        
        # v_in=4 #cut-in windspeed [m/s] -> to be input
        # v_out=25 #cut-out wind speed [m/s]
        # v_rated=9.8 #rated wind speed [m/s]
        
        n = self.v_rated#weibull scale parameter -> unsure
        wnd_vec=np.arange(self.v_in,self.v_out)
        weibull_pdf=(a/n) * (wnd_vec/n) **(a-1) * np.exp(-(wnd_vec/n)**a)
        power_approx=np.zeros(len(wnd_vec))
        idx=0
        rho=1.225 #[kg/m^3]
        
        K=(math.pi*rho*(self.rot_rad**5)*self.cp_max)/(2*(self.tsr_op**3)) #Quick n' dirty K for below-rated operation
        for v in wnd_vec:
            omega=(self.tsr_op*v/self.rot_rad)# rotor speed in rad/s
            if v<self.v_rated:
                power_approx[idx]=K*(omega**3)*self.dt_gen_eff*(1/1000) #[kW] P=torque*omega=K*omega^3
            else:
                power_approx[idx]=self.turb_rating_kw #above-rated operation, saturate to max value
            idx=idx+1
        self.power_pdf=np.array(weibull_pdf*power_approx) #use this to estimate down-time for stacks
        self.weibull=weibull_pdf
        self.ss_pwr=power_approx
        aep_approx=8760*sum(weibull_pdf*power_approx) #[kWh/year]
        return aep_approx #just for one turb


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
        
        


    def estimate_annual_h2(self): #if not given stack specific info
        taep=self.estimate_aep() #this may not be necessary?
        #use power_pdf is annual hourly data isn't given
        
        num_stacks=np.arange(1,self.n_stacks+1,1)
        #num_stacks=np.arange(self.n_stacks,0,-1)
        min_pwr_stack=0.1*self.stack_rating_kW 
        perc_p_down=[]
        stack_down_dict={}
        stack_down_val=np.zeros(self.n_stacks)
        stack_down_tot=[]
        stack_down_pwr=np.zeros(self.n_stacks)
        for n in num_stacks:
            
            min_pwr=min_pwr_stack*n
            #perc_p_down=[p for p in self.power_pdf if p<min_pwr] #not enough to power n stack(s)
            perc_p_down=[p for p in self.ss_pwr if p<min_pwr] #not enough to power n stack(s)
            idx=[]
            for esg in range(len(perc_p_down)):
                #idx.append(np.where(self.power_pdf==perc_p_down[esg]))
                idx.append(np.where(self.ss_pwr==perc_p_down[esg]))

            idx_new=[int(i[0]) for i in idx]
            pwrz=perc_p_down/self.weibull[idx_new]
            stack_down_perc=sum(self.weibull[idx_new])
            #perc_down2=sum(self.weibull)
            stack_down_dict['{} Stacks Down Percent'.format(n)]=sum(perc_p_down/pwrz)
            #stack_down_val.append(sum(perc_p_down/pwrz))
            stack_down_val[n-1]=(sum(perc_p_down/pwrz))
            stack_down_pwr
            #stack_down_tot.append(stack_down_val[-1]-stack_down_val[-2])
            #stack_down_val=sum(perc_p_down/pwrz)
            #stack_down_act=stack_down_val[0]-stack_down_val
        
        stack_down_indiv=np.append(stack_down_val[0],stack_down_val[-(self.n_stacks-1):]-stack_down_val[:(self.n_stacks-1)])
        
        for stacknum, stackdownperc in enumerate(stack_down_indiv,1):
            all_operational_perc=1-sum(stack_down_indiv)
            op_hrs_annual=all_operational_perc*8760

            operational_stacks=self.n_stacks-stacknum #number of stacks that aren't off
            self.electrolyzer_elec_consump

        dict_keys=list(stack_down_dict.keys())

        return stack_down_dict



