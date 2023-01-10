#python3  
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Nagfal 
# Created Date: 8/1/2023 
# version ='1.1'
# ---------------------------------------------------------------------------
""" run the data for Fig. 3"""  

import env_CCBGE as env
import numpy
import data_rw

import numpy
if __name__ == "__main__":
    
    base_list = numpy.linspace(-2, 2, 50).tolist()
    
    ##the list of exploration constants
    C_list= numpy.power(10,base_list).tolist()

    ##the list of expected reward expectations
    sdu_list = numpy.linspace(0.00, 0.125, 25).tolist()
    
    # the file to record the data
    dw = data_rw.data_writer('accu_regret')
    
    for sdu in sdu_list:
        for c in C_list:
            res = 0.0
            repeat = 50
            for i in range(0,repeat):
                sb = env.bandit(20,10,eta = 0.0, sigma= 1.0)
                sb.reset( social_network_mode='stochastic', exploration_constant = c, sigma= 1.0, stochastic_network_prob = 0.2)
                
                done = False
                while not done:
                    current_avg_reward, done = sb.round()
                res += sb.get_avg_regret()
                
            res = res/repeat
            print('sdu', sdu, '||  C:', c, '|| avg accumulated regret:', res)
            dw.single_data_w(res,sdu_list.index(sdu),C_list.index(c),'sigma_0.5,delta_0.01_bonuli_2')
            dw.save()
    pass

