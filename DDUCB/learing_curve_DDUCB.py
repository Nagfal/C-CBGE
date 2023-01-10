#python3  
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Nagfal 
# Created Date: 8/1/2023 
# version ='1.1'
# ---------------------------------------------------------------------------
""" run onfidence-weighted collective Boltzmann-Gumbel exploration (C-CBGE) in Scenario 1 (Figure 1) or Scenario 2 (Figure 2)"""  

import env_DDUCB as env
import numpy
import data_rw

import numpy
if __name__ == "__main__":
    # envi = env.bandit(1,50,10,eta = 0.15, sigma= 1.0)
    
    # the agent number
    n = 20
    
    
    #file to store the data of each simulation
    dw = data_rw.data_writer('result_each_epoch')
    #file to store the data of the average and standard deviation
    dw2 = data_rw.data_writer('avg_standardDeviation_regret')
    
    #repeat times
    repeat = 100
    
    avg_regret = [0.0]*(env.Horizion+1)
    curve_list = []
    for i in range(repeat):
        envi = env.bandit(n,10, sigma= 1.0)
        
        #reset the environment
        #to run DDUCb set social_network_mode='cycle' or social_network_mode='stochastic'
        # set eta  = 0.0 for Scenario 1 and eta = 0.5*(0.15)**2 for Scenario 2
        envi.reset(eta = 0.0, social_network_mode='stochastic',stochastic_network_prob=0.4)
        done = False
        regret_list = []
        while not done:
            current_avg_reward, done = envi.round()
            regret = envi.get_avg_regret()

            regret_list.append(regret)

            dw.single_data_w(regret,envi.time,i,'eta0_n20')
        curve_list.append(regret_list.copy())
        dw.save()
        print('repeat: ' , i, '|||| accu regret: ' , regret)
    
    
    # calculate and store the average and standard deviation    
    for j in range(envi.horizion):
        res = []
        for r in range(0,repeat):
            avg_regret[j] += curve_list[r][j]
            res.append(curve_list[r][j])
        avg_regret[j]/=repeat
        dw2.single_data_w(avg_regret[j],j,0,'eta0_n20_avg')
        dw2.single_data_w(numpy.std(res),j,0,'eta0_n20_std')
    dw2.save()
    

    
    # base_list = numpy.linspace(-4, 0.5, 50).tolist()
    # C_list= numpy.power(10,base_list).tolist()
    # N_list = numpy.linspace(1,25,25).tolist()
    # # N_list = [1]

    # dw = data_rw.data_writer('C_N_full2')
    # for n in N_list:
    #     for c in C_list:
    #         res = 0.0
    #         repeat = 30
    #         for i in range(0,repeat):
    #             sb = env.bandit(1,int(n),10,eta = 0.0, sigma= 0.25)
    #             sb.reset(social_network_mode='full', exploration_constant = c, sigma= 0.25 ,eta = 0.0)
    #             done = False
    #             while not done:
    #                 current_avg_reward, done = sb.round()
    #             res += sb.get_avg_regret()
                
    #         res = res/repeat
    #         print('N', n, '||  C:', c, '|| avg accumulated regret:', res)
    #         dw.single_data_w(res,N_list.index(n),C_list.index(c),'sigma_0.5,delta_0.01_bonuli_2')
    #         dw.save()
    pass
