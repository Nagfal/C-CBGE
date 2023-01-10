#python3  
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Nagfal 
# Created Date: 22/11/2022 
# version ='1.0'
# ---------------------------------------------------------------------------
""" The running environment for UCBGE and BGE"""  

import numpy
import sympy
import agent_UCBGE_BGE as agent

Horizion = 10000

class bandit(object):
    
    # --------------------------------------------------------------------------------
    # function: __init__
    # problem initialization
    # input variables:
    # agent_num : the number of the agent (agent_num = 1,2,3,4,...)
    # arm_num : the number of the arms (arm_num = 10)
    # eta : the difference parameter (eta = 0 for Scenario 1 and eta > 0 for Scenario 2 )
    # sigma : the variance of the reward (sigma >= 0)
    # exploration_constant : the exploration constant (exploration_constant = 1.0)
    # return:
    # null
    #-------------------------------------------------------------------------------- 
    def __init__(self, agent_num, arm_num, eta = 0.0, sigma = 1.0, exploration_constant = 1.0 ):
        self.agent_num = agent_num
        self.arm_num = arm_num

        self.agent_list = []
    
        self.eta = eta
        self.sigma = sigma

        self.exploration_constant = exploration_constant

        self.origin_expected_reward = numpy.array([0.8] * self.arm_num)
        self.origin_expected_reward[0] = 1.0
        self.origin_expected_reward_list= []
        for i in range(0,self.agent_num):
            self.origin_expected_reward_list.append(self.origin_expected_reward.copy())
        self.horizion = Horizion
        self.time = 0

        self.stochastic_network_prob = 0.0

        gamma = sympy.EulerGamma.evalf()

        self.purt = numpy.random.gumbel(gamma,0.2,size=[self.horizion, self.agent_num, self.arm_num])
        


    # --------------------------------------------------------------------------------
    # function: reset
    # reset the problem encironment
    # input variables:
    # eta : the difference parameter (eta = 0 for Scenario 1 and eta > 0 for Scenario 2 )
    # sigma : the variance of the reward (sigma >= 0)
    # exploration_constant : the exploration constant (exploration_constant = 1.0)
    # social_network_mode : the type of the social network ('none','cycle','stochastic')
    # stochastic_network_prob : the probability of two agents being the neighbors (only in stochastic)
    # return:
    # null
    #-------------------------------------------------------------------------------- 
    def reset(self,eta = 0.0, sigma = 1.0, exploration_constant = 1.0, social_network_mode = 'full',stochastic_network_prob = 0.2):
        assert social_network_mode in ['none','cycle','stochastic']
        self.eta = eta
        self.sigma = sigma

        self.stochastic_network_prob = stochastic_network_prob
        
        self.exploration_constant = exploration_constant

        self.social_network_init(social_network_mode)

        sd = (0.5*eta*(sigma**2))**0.5

        if eta > 0.0:
            for i in range(1,len(self.origin_expected_reward_list)):
                for j in range(0, len(self.origin_expected_reward_list[i])):
                    self.origin_expected_reward_list[i][j] = numpy.random.normal(self.origin_expected_reward_list[0][j],sd)
            for j in range(0, len(self.origin_expected_reward_list[0])):
                self.origin_expected_reward_list[0][j] = numpy.random.normal(self.origin_expected_reward_list[0][j],sd)

        pass
    
    # --------------------------------------------------------------------------------
    # function: social_network_init
    # initialize the social network
    # input variables:
    # social_network_mode : the type of the social network ('none','cycle','stochastic')
    # return:
    # null
    #-------------------------------------------------------------------------------- 
    def social_network_init(self, social_network_mode):
        # assert social_network_mode in ['cycle','stochastic']
        for i in range(0,self.agent_num):
            self.agent_list.append(agent.agent(i, self.arm_num, exploration_constant=self.exploration_constant))
        
        if social_network_mode == 'cycle':
            for a in self.agent_list:
                a.neighbors = [ self.agent_list[a.id - 1] ,  self.agent_list[(a.id + 1)%self.agent_num]]
        elif social_network_mode == 'stochastic':
            self.stochastic_network_init()
        pass
    
    
    # --------------------------------------------------------------------------------
    # function: stochastic_network_init
    # initialize the stochastic social network
    # input variables:
    # null
    # return:
    # null
    #-------------------------------------------------------------------------------- 
    def stochastic_network_init(self):
        for i in range(len(self.agent_list)-1):
            for j in range(i+1,len(self.agent_list)):
                ifedge = numpy.random.choice([True,False],p=[self.stochastic_network_prob , (1 - self.stochastic_network_prob)])
                if ifedge:
                    self.agent_list[i].neighbors.append(self.agent_list[j])
                    self.agent_list[j].neighbors.append(self.agent_list[i])    


    # --------------------------------------------------------------------------------
    # function: round
    # run a single round of the problem
    # input variables:
    # null
    # return:
    # avg_reward : the average reward of tha agents in this round 
    # if_done (bool): if the game is finished 
    #-------------------------------------------------------------------------------- 
    def round(self):
        if self.time < self.arm_num:
            for a in self.agent_list:
                a.set_arm(self.time)
        else:
            for a in self.agent_list:
                a.decision(self.purt[self.time][a.id])

        total_reward = 0.0
        for a in range(0,len(self.agent_list)):
            arm = self.agent_list[a].arm

            x=self.origin_expected_reward_list[a][arm]
            reward = numpy.random.normal(x,self.sigma)
            self.agent_list[a].get_reward(arm, reward)
            if self.eta != 0.0:
                optimal = numpy.max(self.origin_expected_reward_list[a])
            else:
                optimal = self.origin_expected_reward_list[a][0]
            self.agent_list[a].accumulated_regret += optimal - self.origin_expected_reward_list[a][arm]
            total_reward += reward
            

        
        avg_reward = total_reward/self.agent_num

        self.time += 1
        if self.time>=self.horizion:
            if_done = True
        else:
            if_done = False
        
        return avg_reward,if_done
    
    # --------------------------------------------------------------------------------
    # function: get_avg_reward
    # calculate the average reward
    # input variables:
    # null
    # return:
    # the average reward of tha agents 
    #-------------------------------------------------------------------------------- 
    def get_avg_reward(self):
        return sum([ x.accumulated_reward for x in self.agent_list ])/self.agent_num
    
    
    # --------------------------------------------------------------------------------
    # function: get_avg_reward
    # get the average regret
    # input variables:
    # null
    # return:
    # the average regret of tha agents 
    #-------------------------------------------------------------------------------- 
    def get_avg_regret(self):
        return sum([ x.accumulated_regret for x in self.agent_list ])/self.agent_num


# if __name__ == "__main__":
#     env = bandit(1,1,30,eta = 0.0, sigma= 0.5)
#     env.reset(social_network_mode='full')
    
#     while True:
#         current_avg_reward, done = env.round()
#         print('time', env.time, '||  current R:', current_avg_reward, '|| avg accumulated reward:', env.get_avg_reward(), '|| avg accumulated regret:', env.get_avg_regret())
        
#     pass





