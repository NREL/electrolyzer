
import sys
from distutils.util import execute
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle

import matplotlib.pyplot as plt
import os
import math
import scipy
from mpl_toolkits import mplot3d
#from pem_lcoh_esg02 import PEM_lcoh
import lcoh_calcs
from electrolyzer import Supervisor
from electrolyzer.stack import Stack
from electrolyzer.cell import Cell, electrolyzer_model
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
    
    annual_data_dict['Method'] = 'Both' #options: 'LCOH-I','LCOH-II','Both'
    annual_data_dict['Cost of Energy [$/kWh]']= 44.18*(1/1000) #[$/kWh]
    annual_data_dict['Plant Energy Consumption [kWh/year]']= 3522238907 #[kWh/year]
     

    annual_data_dict['Annual H2 Production [kg]']=67273320 #[kg] of H2 produced in a year 67273320 for 1GW plant
    annual_data_dict['Single Stack Size [kW]']=1000 #single stack rating
    annual_data_dict['Number of Stacks']=1000 # total number of stacks
    
    
    #it can be a list or sum - whichever you want :)
    annual_data_dict['Annual number of stack replacements']=[2.5,2.5] 
    #it can be a list or sum - whichever you want :)
    annual_data_dict['Annual Hours of Operation for all Stacks']=[4000]*1000#[1250,1250] # [hrs] annual equivalent hours of operation 

    annual_data_dict['PEM Location']='onshore' # options: 'onshore', 'offshore', 'in-turbine'
    annual_data_dict['Total Annual Cost of Energy [$]']='None' #optional
    #if total annual cost of power isn't specified then it's calculated from cost of energy and plant energy consumption

    lcoh_calc=lcoh_calcs.PEM_lcoh(annual_data_dict)
    lcoh,outputs_dict=lcoh_calc.run()

    return lcoh, outputs_dict
    
    []
lcoh_vals,cost_breakdown_dict=I_Have_Annual_Data()
[]
    #def __init__(self, electrolyzer_specs,electrolyzer_perf_dict, turb_dict):
class Estimate_Annual_Data_From_TS:

    def __init__(self, elec_sys,supervisor_dict,estimated_annual_dict):
        '''

        '''
        #estimated_annual_dict={}
        estimated_annual_dict['Method'] = 'Both'
        estimated_annual_dict['Cost of Energy [$/kWh]']=44.18*(1/1000) #[$/kWh]
        estimated_annual_dict['PEM Location']='onshore'

        estimated_annual_dict['Number of Stacks']=supervisor_dict["n_stacks"] #elec_sys.n_stacks
        estimated_annual_dict['Single Stack Size [kW]']=supervisor_dict["stack"]["stack_rating_kW"] #elec_sys.stack_rating_kW
        estimated_annual_dict['Total Annual Cost of Energy [$]']='None'
        self.elec_sys=elec_sys
        estimation_dict=self.simple_estimation()
        estimated_annual_dict.update(estimation_dict)
        []
        # lcoh_calc=lcoh_calcs.PEM_lcoh(annual_data_dict)
        # lcoh,outputs_dict=lcoh_calc.run()


    
    def simple_estimation(self):
        simple_dict={}
        n_stacks=self.elec_sys.n_stacks
        h2_df=pd.DataFrame(self.elec_sys.H2_store) # [kg]
        power_df=pd.DataFrame(self.elec_sys.P_indv_store) #[W]
        sim_time=len(h2_df)/3600 #[hrs]
        d_eol=0.7212 # [V] end of life degradation voltage

        #estimate annual number of stack replacements from degradation
        d_sim=[self.elec_sys.stacks[i].V_degradation for i in range(n_stacks)]
        deg_ratio=(d_eol/np.array(d_sim)) #how many times it can run this simulation before death
        #if deg ratio = 100 then we could run that simulation 100x before the stack dies

        time_til_death=(d_eol/np.array(d_sim))*sim_time #[hrs]
        annual_stack_replacements=8760/time_til_death


        stack_uptime=[self.elec_sys.stacks[i].uptime for i in range (n_stacks)]
        # stack_fatigue=[self.elec_sys.stacks[i].fatigue_history for i in range (n_stacks)]
        # stack_cycles=[self.elec_sys.stacks[i].cycle_count for i in range(n_stacks)]

        
        #estimate annual h2 production
        stack_h2=[sum(h2_df[i]) for i in range(n_stacks)] #[kg]
        stack_h2_avg_rate=np.array(stack_h2)/np.array(stack_uptime) #[h2/sec-on]
        stack_perc_uptime=np.array(stack_uptime)/(sim_time*3600)
        annual_stack_on_sec=stack_perc_uptime*8760*3600 #[sec on per year]
        annual_h2=annual_stack_on_sec*stack_h2_avg_rate #[kg/year]

        stack_power=[sum(power_df[i]*(1/3600) ) for i in range(n_stacks)] #[Wh]
        stack_power_consumed_rate=np.array(stack_power)/np.array(stack_uptime) #[W/sec-on]
        annual_power=annual_stack_on_sec * stack_power_consumed_rate #[W/year]

        approx_operational_hours=stack_perc_uptime*8760
        #approx_operational_hours=annual_stack_replacements*8760

        simple_dict['Plant Energy Consumption [kWh/year]']=np.sum(annual_power)/1000
        simple_dict['Annual H2 Production [kg]']=np.sum(annual_h2)#/(np.array(d_sim)/d_eol))#np.sum(annual_h2)
        simple_dict['Annual number of stack replacements']=list(annual_stack_replacements) #kinda gross way to do it
        simple_dict['Annual Hours of Operation for all Stacks']=list(approx_operational_hours)
        return simple_dict

if __name__=="__main__":
    info_dict={}
    info_dict['stack']={'stack_rating_kW':500}
    info_dict["n_stacks"]=7
    output_dict={}
    parent_path = os.path.abspath('')
    elec_dir=parent_path + '/Results'
    
    controllers= [
    "BaselineDeg",]
    # "PowerSharingRotation",
    # "SequentialRotation",
    # "EvenSplitEagerDeg",
    # "EvenSplitHesitantDeg",
    # "SequentialEvenWearDeg",
    # "SequentialSingleWearDeg"]

    for c in controllers:
        saveme=False
        elec_sys=pd.read_pickle(elec_dir + c + '/' + 'elec_system')
        annual=Estimate_Annual_Data_From_TS(elec_sys,info_dict,output_dict)
        lcoh_calc=lcoh_calcs.PEM_lcoh(output_dict)
        lcoh,lcoh_dict=lcoh_calc.run()
        
        if saveme:
            #parent_path = os.path.abspath('')
            today=list(time.localtime())
            subdir='/results/M{}_D{}_Y{}'.format(today[1],today[2],today[0] + '/')
            saveme_path=parent_path + subdir
            if not os.path.isdir(saveme_path):
                os.mkdir(saveme_path)
            tot_index=['CapEx','OpEx-I','LCOH-I - Totals','LCOH-II - Totals']
            yearly_idx=['LCOH-I - Yearly','LCOH-II - Yearly']
            year_df=pd.DataFrame()
            final_df=pd.DataFrame()
            for k in tot_index:
                keys,values=zip(*lcoh_dict[k].items())
                df_temp=pd.DataFrame({k + ' Keys': list(keys),k + ' Values':list(values)})
                final_df=pd.concat([df_temp,final_df],ignore_index=False,axis=1)
            for y in yearly_idx:
                keys,values=zip(*lcoh_dict[y].items())
                for ki,key in enumerate(keys):
                    df_temp=pd.DataFrame({y+' ' + key: list(values[ki])})
                    year_df=pd.concat([df_temp,year_df],ignore_index=False,axis=1)
            final_df.to_csv(saveme_path + 'LCOH_Breakdown_' + c + '.csv')
            year_df.to_csv(saveme_path + 'LCOH_Yearly_' + c + '.csv')



        []









        
        
        

        
        
        
        
        
    
    