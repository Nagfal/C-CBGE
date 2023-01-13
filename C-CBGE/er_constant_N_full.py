#python3  
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Nagfal 
# Created Date: 8/1/2023 
# version ='1.1'
# ---------------------------------------------------------------------------
""" run the data for C-CBGE under various exploration constant and agent number"""  

import env_CCBGE as env
import numpy
import data_rw

import numpy
if __name__ == "__main__":
    
    
    ##the list of exploration constants
    base_list = numpy.linspace(-2, 2, 50).tolist()
    C_list= numpy.power(10,base_list).tolist()
    
    ##the list of the agent number
    N_list = numpy.linspace(1,25,25).tolist()


    dw = data_rw.data_writer('C_N')
    for n in N_list:
        for c in C_list:
            res = 0.0
            repeat = 10
            for i in range(0,repeat):
                sb = env.bandit(int(n),10,eta = 0.0, sigma= 1.0)
                sb.reset(eta = 0.5*(0.1)**2 , social_network_mode='full', exploration_constant = c, sigma= 1.0 , sd = -1)
                done = False
                while not done:
                    current_avg_reward, done = sb.round()
                res += sb.get_avg_regret()
                
            res = res/repeat
            print('N', n, '||  C:', c, '|| avg accumulated regret:', res)
            dw.single_data_w(res,N_list.index(n),C_list.index(c),'sigma_0.5,delta_0.01_bonuli_2')
            dw.save()
    pass
