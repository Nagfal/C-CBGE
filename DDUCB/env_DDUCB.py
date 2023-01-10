#python3  
# -*- coding: utf-8 -*- 
#----------------------------------------------------------------------------
# Created By  : Nagfal 
# Created Date: 22/11/2022 
# version ='1.0'
# ---------------------------------------------------------------------------
""" The running environment for DDUCB"""  

import numpy
import numpy.linalg as numpyl
import sympy
import agent_DDUCB as agent
# import agent


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
    def __init__(self, agent_num, arm_num, eta = 0.0,sigma = 1.0, exploration_constant = 0.5 ):
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

        self.stochastic_network_prob = 0.2

        self.P = []

        # gamma = sympy.EulerGamma.evalf()

        # self.purt = numpy.random.gumbel(gamma,0.2,size=[self.horizion, self.agent_num, self.arm_num])
        


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
    def reset(self,eta = 0.0, sigma = 1.0, exploration_constant = 0.5, social_network_mode = 'cycle',stochastic_network_prob=0.2):
        assert social_network_mode in ['cycle','stochastic']
        
        self.eta = eta
        self.sigma = sigma
        self.stochastic_network_prob = stochastic_network_prob
        self.exploration_constant = exploration_constant

        self.social_network_init(social_network_mode)

        sd = (0.5*eta*(sigma)**2)**0.5
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
        for i in range(0,self.agent_num):
            self.agent_list.append(agent.agent(i, self.arm_num, self.agent_num , network_type=social_network_mode))
            # self.agent_list.append(agent.agent(i,[i%self.agent_num_x, int(i/self.agent_num_x)] , self.arm_num,self.eta,exploration_constant=self.exploration_constant))
        
        if social_network_mode == 'cycle':
            for a in self.agent_list:
                a.neighbors = [ self.agent_list[a.id - 1] ,  self.agent_list[(a.id + 1)%self.agent_num]]
        elif social_network_mode == 'stochastic':
            self.stochastic_network_init()
            pass
        
        
        # get the P_matrix and the second large engien value
        P_matrix  = self.get_p_matrix(social_network_mode)
        ev = self.get_engien_values( P_matrix )
        lambda_2  = abs(ev)
        sorted_lam = numpy.sort(lambda_2)
        for i in self.agent_list:
            
            i.second_engien_value = sorted_lam[-2]
            if i.second_engien_value>=1:
                i.second_engien_value = 0.999
            i.P_matrix = P_matrix

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
                # numpy.random.seed(9)
                ifedge = numpy.random.choice([True,False],p=[self.stochastic_network_prob , (1 - self.stochastic_network_prob)])
                if ifedge:
                    self.agent_list[i].neighbors.append(self.agent_list[j])
                    self.agent_list[j].neighbors.append(self.agent_list[i])

        for i in self.agent_list:
            if len(i.neighbors) == 0:
                i.neighbors.append(numpy.random.choice(self.agent_list))


    # --------------------------------------------------------------------------------
    # function: get_p_matrix
    # get the P_matrix
    # input variables:
    # stype: the type of the social network
    # return:
    # jm : the P_matrix
    #
    # Note:
    # more details about the method to choose P_matrix can be found in [1][2]
    
    
    # [1] Duchi J C, Agarwal A, Wainwright M J. Dual averaging for distributed optimization: Convergence analysis and network scaling[J]. IEEE Transactions on Automatic control, 2011, 57(3): 592-606.
    # [2] Xiao L, Boyd S. Fast linear iterations for distributed averaging[J]. Systems & Control Letters, 2004, 53(1): 65-78.
    #-------------------------------------------------------------------------------- 
    def get_p_matrix(self,stype = 'cycle'):
        jm = numpy.zeros([self.agent_num,self.agent_num])

        if stype == 'stochastic' :
            A = []
            for i in self.agent_list:
                v = [0]*self.agent_num
                for nei in i.neighbors:
                    v[nei.id] = 1
                A.append( v  )
            a = numpy.array(A)
            delta  = numpy.array([ len(a.neighbors) for a in self.agent_list] )
            D_sqrt_inv = numpy.diag(1/numpy.sqrt(delta))
            Lap = numpy.eye(self.agent_num, dtype=float) - numpy.dot(numpy.dot(D_sqrt_inv, A), D_sqrt_inv)
            P = numpy.eye(self.agent_num, dtype=float) - 1/(delta.max()+1)*numpy.dot(numpy.dot(D_sqrt_inv, Lap), D_sqrt_inv)
            return P

        for i in range(self.agent_num):
            for n in self.agent_list[i].neighbors:
                jm[i][n.id] = 1/(len(self.agent_list[i].neighbors) + 1)
        for i in range(self.agent_num):
            jm[i][i] = 1/(len(self.agent_list[i].neighbors) + 1)
        return jm
    
    # --------------------------------------------------------------------------------
    # function: get_engien_values
    # calculate the engien values of the P_matrix
    # input variables:
    # P_matrix : the P matrix
    # return:
    # e_values : an array of the engien values of the P_matrix
    #-------------------------------------------------------------------------------- 
    def get_engien_values(self,P_matrix):

        numpy.nan_to_num(P_matrix)

        e_values, f_vector = numpyl.eig(P_matrix)
        return e_values


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

        for a in self.agent_list:
            a.decision(self.time)

        total_reward = 0.0
        for a in range(0,len(self.agent_list)):
            arm = self.agent_list[a].arm

            x=self.origin_expected_reward_list[a][arm]
            # numpy.random.seed(self.time*3)
            reward = numpy.random.normal(x,self.sigma)
            if reward <0:
                reward = 0.0
            self.agent_list[a].get_reward(arm, reward,self.time)

            if self.eta != 0.0:
                optimal = numpy.max(self.origin_expected_reward_list[a])
            else:
                optimal = self.origin_expected_reward_list[a][0]
            
            self.agent_list[a].accumulated_regret += optimal - self.origin_expected_reward_list[a][arm]
            # if optimal - self.origin_expected_reward_list[a][arm] == 0:
            #     print('yes')
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
    # get the average regret
    # input variables:
    # null
    # return:
    # the average regret of tha agents 
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
#     env = bandit(10,10,10,eta = 0.0, sigma= 0.25)
#     env.reset(eta = 0.0, social_network_mode='cycle')
    
#     done = False
#     while not done:
#         current_avg_reward, done = env.round()
#         print('time', env.time, '||  current R:', current_avg_reward, '|| avg accumulated reward:', env.get_avg_reward(), '|| avg accumulated regret:', env.get_avg_regret())
        
#     pass





