#python3  
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Nagfal 
# Created Date: 22/11/2022 
# version ='1.0'
# ---------------------------------------------------------------------------
""" the code for DDUCB"""  
# more details about DDUCB can be found in [1]
# [1] Martínez-Rubio D, Kanade V, Rebeschini P. Decentralized cooperative stochastic bandits[J]. Advances in Neural Information Processing Systems, 2019, 32.

import numpy
import math
import sympy

#the exploration constant of DDUCB （eta_DDUCB>0）
eta_DDUCB = 0.2


class agent(object):
    
    # --------------------------------------------------------------------------------
    # function: __init__
    # initialization for DDUCB
    # input variables:
    # id : the id of the agent (id = 1,2,3,4,...)
    # eta : the difference parameter (eta >= 0)
    # num_arm : the number of the arms (num_arm = 10)
    # exploration_constant : the exploration constant (exploration_constant = 1.0)
    # return:
    # null
    #--------------------------------------------------------------------------------
    def __init__(self, id, num_arm, num_agent,  sigma = 1.0, network_type = 'full'):
        self.id = id
        # self.loc = loc
        self.num_arm = num_arm
        self.num_agent = num_agent

        self.second_engien_value = 0.5 

        self.arm_sigma = sigma

        self.last_action = -1
        self.last_reward = 0.0
        self.arm = -1
        self.accumulated_reward = 0.0
        self.accumulated_regret = 0.0

        self.estimates = [0.5] * num_arm
        self.sample_number = [0] * num_arm       
        self.constant_C = int( numpy.ceil( numpy.log((2*self.num_agent)/(1/22))/5*(numpy.sqrt(2*numpy.log(2))) ) ) 


        self.initial_pulls_rewards = numpy.array([0.0]* num_arm)
        self.initial_pulls_times = numpy.array([0]* num_arm)

        self.alpha = numpy.array([0.0]* num_arm)
        self.alpha_a = numpy.array([0.0]* num_arm)

        self.beta = numpy.array([0.0]* num_arm)
        self.beta_b = numpy.array([0.0]* num_arm)
        self.last_beta = numpy.array([0.0]* num_arm)
        self.last_beta_b = numpy.array([0.0]* num_arm)

        self.gamma = numpy.array([0.0]* num_arm)
        self.gamma_c = numpy.array([0.0]* num_arm)

        self.delta = numpy.array([0.0]* num_arm)
        self.delta_d = numpy.array([0.0]* num_arm)

        self.s = 0

        self.mix_time = 0

        self.omega_reward = numpy.zeros(3)
        self.omega_pulls = numpy.zeros(3)
        self.y_reward = numpy.zeros((3,self.num_arm))
        self.y_pulls = numpy.zeros((3,self.num_arm))

        self.styps = network_type

        if network_type == 'full':
            self.P = 1 / self.num_agent
        elif network_type == 'grid':
            self.P = 0.2 
        elif network_type == 'circle':
            self.P = 1.0/3.0 

        self.P_matrix = []

        self.neighbors = []
        pass


    # --------------------------------------------------------------------------------
    # function: set_neighbors
    # set the neighbor set of the agent
    # input variables:
    # neighbor_list: the neighbor list of the agent
    # return:
    # null
    #--------------------------------------------------------------------------------
    def set_neighbors(self,neighbor_list):
        self.neighbors = neighbor_list


    # --------------------------------------------------------------------------------
    # function: set_arm
    # set the chosen arm of the agent
    # input variables:
    # arm : the chosen arm of the agent (arm in [1,2,...,num_arm])
    # return:
    # null
    #--------------------------------------------------------------------------------
    def set_arm(self, arm):
        assert arm <self.num_arm
        self.arm = arm


    # --------------------------------------------------------------------------------
    # function: get_reward
    # observe the reward of the agent
    # input variables:
    # arm : the chosen arm of the agent (arm in [1,2,...,num_arm])
    # reward : the reward obtained by the agent 
    # return:
    # null
    #--------------------------------------------------------------------------------    
    def get_reward(self,arm,reward,time):
        self.accumulated_reward += reward
        # self.accumulated_regret += regret

        if time < self.num_arm:
            self.initial_pulls_rewards[time] = reward
            self.initial_pulls_times[time] = 1
            return

        # if self.mix_time < self.constant_C:
        self.gamma[arm] += reward
        self.gamma_c[arm] += 1

        self.alpha[arm] += reward/self.num_agent
        self.alpha_a[arm] += 1/self.num_agent
        self.s+=1

        self.last_beta = self.beta
        self.last_beta_b = self.beta_b
        self.beta = self.mix_reward(self.beta,self.mix_time)
        self.beta_b = self.mix_pulls(self.beta_b,self.mix_time)

        self.mix_time+=1

        if self.mix_time == self.constant_C-1:
            
            self.s = (time - self.constant_C) * self.num_agent
            # if self.s < 0:
            #     print('fuck')

            self.delta = self.delta +self.beta
            self.delta_d  = self.delta_d + self.beta_b

            self.alpha = self.delta
            self.alpha_a = self.delta_d


            self.last_beta = self.gamma
            self.last_beta_b = self.gamma_c
            self.beta = self.gamma
            self.beta_b = self.gamma_c

            self.gamma = numpy.array([0.0]* self.num_arm)
            self.gamma_c = numpy.array([0.0]* self.num_arm)

            self.mix_time = 0

        

        
    # --------------------------------------------------------------------------------
    # function: mix_reward
    # the implementation of Algorithm 2 in [1], mixing the reward observations
    # input variables:
    # y : the reward value that need to be mixed
    # r : the time of mix
    # return:
    # y_reward[1] : the mixed reward
    #--------------------------------------------------------------------------------
    def mix_reward(self,y, r):      
        if r == 0:
            y_current = 0.5 * y
            y_last1 = numpy.zeros(len(y))
            self.omega_reward[0] = 0.0
            self.omega_reward[1] = 0.5
            self.y_reward[1] = y_current
            self.y_reward[0] = y_last1

        y_sum = numpy.zeros(len(y))

        for i in self.neighbors:
            e = abs(self.second_engien_value)
            if self.styps == 'cycle':
                y_sum = y_sum + (2 * self.P_matrix[self.id][i.id] * i.last_beta) / e
            else:
                y_sum = y_sum + (2 * self.P * i.last_beta) / e
            # y_sum = y_sum + (2 * self.P * i.last_beta) / e

        self.omega_reward[2] = 2* self.omega_reward[1] / abs(self.second_engien_value) - self.omega_reward[0] 
        self.y_reward[2] = ( self.omega_reward[1] / self.omega_reward[2])*y_sum - ( self.omega_reward[0] / self.omega_reward[2])*self.y_reward[0]

        if  self.omega_reward[2]<0.0 or min(self.y_reward[2])<0.0:
            print('fuck')

        if r == 0:
            self.y_reward[1] = self.y_reward[1]*2
            self.omega_reward[1] = self.omega_reward[1] *2
        
        self.y_reward[0] = self.y_reward[1]
        self.y_reward[1] = self.y_reward[2]

        self.omega_reward[0] = self.omega_reward[1]
        self.omega_reward[1] = self.omega_reward[2]

        return self.y_reward[1]


    # --------------------------------------------------------------------------------
    # function: mix_pulls
    # the implementation of Algorithm 2 in [1], mixing the time of pulls
    # input variables:
    # y : the value that need to be mixed
    # r : the time of mix
    # return:
    # self.y_pulls[1] : the mixed number of pulls
    #--------------------------------------------------------------------------------
    def mix_pulls(self,y, r):      
        if r == 0:
            y_current = 0.5 * y
            y_last1 = numpy.zeros(len(y))
            self.omega_pulls[0] = 0.0
            self.omega_pulls[1] = 0.5
            self.y_pulls[1] = y_current
            self.y_pulls[0] = y_last1

        y_sum = numpy.zeros(len(y))

        for i in self.neighbors:
            e = abs(self.second_engien_value)
            if self.styps == 'grid' :
                y_sum = y_sum + (2 * self.P_matrix[self.id][i.id] * i.last_beta_b) / e
            else:
                y_sum = y_sum + (2 * self.P * i.last_beta_b) / e
            # y_sum = y_sum + (2 * self.P * i.last_beta) / e
        
        self.omega_pulls[2] = 2* self.omega_pulls[1] / abs(self.second_engien_value) - self.omega_pulls[0]
        
        self.y_pulls[2] = ( self.omega_pulls[1] / self.omega_pulls[2])*y_sum - ( self.omega_pulls[0] / self.omega_pulls[2])*self.y_pulls[0]

        if  self.omega_pulls[2]<0.0 or min(self.y_pulls[2])<0.0:
            print('fuck')

        if r == 0:
            self.y_pulls[1] = self.y_pulls[1]*2
            self.omega_pulls[1] = self.omega_pulls[1] *2
        
        self.y_pulls[0] = self.y_pulls[1]
        self.y_pulls[1] = self.y_pulls[2]

        self.omega_pulls[0] = self.omega_pulls[1]
        self.omega_pulls[1] = self.omega_pulls[2]

        

        return self.y_pulls[1]

        
    # --------------------------------------------------------------------------------
    # function: decision
    # get the decision of the agent
    # input variables:
    # time : current time that the game is played
    # return:
    # self.arm : the chosen arm
    #--------------------------------------------------------------------------------   
    def decision(self,time):
        self.P = 1/(len(self.neighbors)+1)
        if time < self.num_arm:
            self.arm = time
            return self.arm
        elif time == self.num_arm:
            self.alpha = self.initial_pulls_rewards / self.num_agent
            self.alpha_a = self.initial_pulls_times / self.num_agent

            self.beta = self.initial_pulls_rewards
            self.beta_b = self.initial_pulls_times

            self.s = self.num_arm

        # ucb_list = self.alpha / self.alpha_a + numpy.sqrt( (4 * (self.arm_sigma) * numpy.log(self.s) ) / (self.num_agent * self.alpha_a)   )
        for a in range(len(self.alpha_a)):
            if self.alpha_a[a] < 0:
                self.alpha_a[a] = 0.0
        ucb_list = (self.alpha / self.alpha_a) + numpy.sqrt( (2*eta_DDUCB* (self.arm_sigma**2) * numpy.log(self.s) ) / (self.num_agent * self.alpha_a)   )
        self.arm = numpy.argmax(ucb_list)
        return self.arm

        # if self.mix_time == 0:
        #     ucb_list = self.alpha / self.alpha_a + numpy.sqrt( (4 * (0.25) * numpy.log(self.s) ) / (self.num_agent * self.alpha_a)   )
        #     self.arm = numpy.argmax(ucb_list)
            
        #     return self.arm
        # else:
            
        #     return self.arm
        
        
        