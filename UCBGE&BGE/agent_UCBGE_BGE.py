#!/usr/bin/env python3  
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Nagfal 
# Created Date: 22/11/2022 
# version ='1.0'
# ---------------------------------------------------------------------------
""" the code for UCBGE & BGE"""  

import numpy
import math
import sympy
class agent(object):
    
    # --------------------------------------------------------------------------------
    # function: __init__
    # initialization for UCBGE or BGE
    # input variables:
    # id : the id of the agent (id = 1,2,3,4,...)
    # eta : the difference parameter (for UCBGE and BGE, eta should be 0)
    # num_arm : the number of the arms (num_arm = 10)
    # exploration_constant : the exploration constant (exploration_constant = 1.0)
    # return:
    # null
    #--------------------------------------------------------------------------------
    def __init__(self, id, num_arm, eta = 0.0, exploration_constant = 0.5):
        self.id = id
        # self.loc = loc
        self.num_arm = num_arm
        self.eta = eta

        self.last_action = -1
        self.arm = -1
        self.accumulated_reward = 0.0
        self.accumulated_regret = 0.0

        self.estimates = [0.5] * num_arm
        self.sample_number = [0] * num_arm
        self.accu_reward_arm = [0.0] * num_arm

        self.exploration_constant = exploration_constant
        
        self.confidence = 1.0   

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
    def get_reward(self,arm,reward):
        self.accumulated_reward += reward
        self.accu_reward_arm[arm] += reward
        # self.estimates[arm] = self.estimates[self.arm] * ( self.sample_number[self.arm]/ ( 1+ self.sample_number[self.arm] ) ) 
        # self.estimates[arm] += reward * ( 1/ ( 1+ self.sample_number[self.arm] ) )
        self.sample_number[arm] += 1


    # --------------------------------------------------------------------------------
    # function: get_neighbor_sample_num
    # get the times that the arms choosed by the neighbors
    # input variables:
    # null
    # return:
    # sn : a list of the times that the arms choosed by the neighbors
    #--------------------------------------------------------------------------------
    def get_neighbor_sample_num(self):
        sn = [0.0] * self.num_arm
        
        for i in range(len(sn)):
            for n in self.neighbors:
                sn[i] += n.sample_number[i] 
        return sn


    # --------------------------------------------------------------------------------
    # function: get_confidence_list
    # get the current confidence list of the agent
    # input variables:
    # neighbor_sample_nums : a list of the times that the arms chosen by the neighbors, i.e., the output of function "get_neighbor_sample_num"
    # return:
    # confidence_list : a list of the confidences
    #--------------------------------------------------------------------------------
    def get_confidence_list(self,neighbor_sample_nums):
        confidence_list = [0.0] * self.num_arm
        # if self.eta == 0.0:
        #     return confidence_list
        for i in range(len(confidence_list)):
            confidence_list[i] = (1+self.eta)* self.sample_number[i] / ( (1+self.eta)* self.sample_number[i] + neighbor_sample_nums[i] )
        return confidence_list


    # --------------------------------------------------------------------------------
    # function: decision
    # get the decision of the agent
    # input variables:
    # purt : a standard Gumbel distributed perturbation
    # return:
    # self.arm : the chosen arm
    #--------------------------------------------------------------------------------    
    def decision(self, purt):
        var_est = [0.0] * self.num_arm

        self.eta = 0
        

        sample_numbers = [0.0] * self.num_arm
        neighbor_sample_nums = self.get_neighbor_sample_num()

        for i in range(0,len(sample_numbers)):            
            sample_numbers[i] += (1+ self.eta) * self.sample_number[i] + neighbor_sample_nums[i]
        

        self.purt = purt

        for i in range(0,len(var_est)):
            x = (((1+ self.eta)) * self.exploration_constant )
            beta = numpy.sqrt(x / ( sample_numbers[i]  ) )
            var_est[i] = self.purt[i] * beta
        
        confidence_list = self.get_confidence_list(neighbor_sample_nums) 

        est_list = [0.0] * self.num_arm
        neighbor_estimates = [0.0] * self.num_arm
        for i in range(0,len(neighbor_estimates)):
            for n in self.neighbors:
                neighbor_estimates[i] += n.accu_reward_arm[i]  / neighbor_sample_nums[i] 
        

        for i in range(0,len(est_list)):
            # a = 1 - confidence_list[i]
            est_list[i] = confidence_list[i] * (self.accu_reward_arm[i] / self.sample_number[i]) + (1 - confidence_list[i]) * neighbor_estimates[i] + var_est[i]

        
        self.arm = numpy.argmax(est_list)
        return self.arm