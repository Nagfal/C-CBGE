#python3  
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Nagfal 
# Created Date: 8/1/2023 
# version ='1.1'
# ---------------------------------------------------------------------------
""" run confidence-weighted collective Boltzmann-Gumbel exploration (C-CBGE) in Scenario 1 (Figure 1) or Scenario 2 (Figure 2)"""  

import env_CCBGE as env
import numpy
import data_rw

import numpy
if __name__ == "__main__":
    
   
    # the agent number
    n = 20
    
    # the difference parameter (eta  = 0.0 for Scenario 1 and eta = 0.5*(0.15)**2 for Scenario 2)
    eta = 0.0

    #file to store the data of each simulation
    dw = data_rw.data_writer('result_each_epoch')
    
    #file to store the data of the average and standard deviation
    dw2 = data_rw.data_writer('avg_standardDeviation_regret')

    avg_regret = [0.0]*(env.Horizion+1)
    avg_regret_r = []

    res = 0.0
    
    #repeat times
    repeat = 100
    
    #run a single simulation        
    for i in range(0,repeat):
        sb = env.bandit(n,10, sigma= 1.0)
        
        #reset the environment
        #to run C-CBGE, set social_network_mode='cycle' or social_network_mode='stochastic'
        #set eta  = 0.0 for Scenario 1 and eta = 0.5*(0.15)**2 for Scenario 2
        #to run C-CBGE for Fig.2, set sd =-1
        e = 0.5*(0.15)**2

        sb.reset(eta = e , social_network_mode='stochastic', exploration_constant = 0.01, sigma= 1.0 ,stochastic_network_prob=0.2, sd = -1)

        done = False
        regret_list = []
        while not done:
            current_avg_reward, done = sb.round()
            regret_list.append(sb.get_avg_regret())
            regret = sb.get_avg_regret()
            dw.single_data_w(regret,sb.time,i,'res')
        avg_regret_r.append(regret_list)
        print('repeat', i, '|| avg accumulated regret:', sb.get_avg_regret())
        res += sb.get_avg_regret()
        dw.save()    

    # calculate and store the average and standard deviation        
    c = numpy.array(avg_regret_r)
    for j in range(sb.horizion):
        res = []
        for r in range(repeat):
            avg_regret[j] += avg_regret_r[r][j]
            res.append(avg_regret_r[r][j])
        avg_regret[j]/=repeat
        dw2.single_data_w(avg_regret[j],j,0,'avg')
        dw2.single_data_w(numpy.std(res,ddof=1),j,0,'std')
    dw2.save()
                
    pass
