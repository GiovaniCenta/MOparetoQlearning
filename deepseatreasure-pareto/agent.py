from multiprocessing.resource_sharer import stop
from pathlib import Path
from xml.etree.ElementTree import tostring

import copy
import gym
import numpy as np
from scipy import rand
#from sympy import Q
import pygame
from gym.spaces import Box, Discrete
from pygmo import hypervolume
from metrics import metrics as met
import deepst
metrics = met([],[],[])
import wandb


class Pareto():
    def __init__(self, env,actionsMethods, choose_action, ref_point, nO=2,nS = 64, gamma=1.):
        self.env = env
        self.actionsMethods=actionsMethods
        self.choose_action = choose_action
        self.gamma = gamma

        self.ref_point = ref_point

        self.nS = nS
        
        self.nA = env.action_space.n
        env.nA = self.nA
        self.non_dominated = [[[np.zeros(nO)] for _ in range(self.nA)] for _ in range(self.nS)]
        self.avg_r = np.zeros((self.nS, self.nA, nO))
        self.n_visits = np.zeros((self.nS, self.nA))
        self.epsilon = 1
        self.epsilonDecrease = 0.99
        self.stateList = []

        
        self.polIndex = 0
        self.log = False
        self.qtable = np.zeros((self.nS, self.nA, nO),dtype=object)

        self.numberofeps = 0
        self.rewards0 = []
        self.rewards1 = []
        self.paretorew0 = []
        self.paretorew1 = []

        

    def initializeState(self):
        return self.flatten_observation(self.env.reset())

    def flatten_observation(self, obs):
        #print(obs[1])
        
        if type(obs[1]) is dict:
        	return int(np.ravel_multi_index((0,0), (11, 11)))
        #print(type(obs[1]))
           
        else:
            return int(np.ravel_multi_index(obs, (11, 11)))


    def train(self,max_episodes,max_steps,log,plot_paretofront):
        
        self.polDict = np.zeros((max_episodes,self.nS, self.nA, 2))
        self.log = log
        if log:
            metrics.setup_wandb("pql", "pql")

        numberOfEpisodes = 0
        episodeSteps = 0
        self.numberofeps = max_episodes

        #line 1 -> initialize q_set
        print("-> Training started <-")
        #line 2 -> for each episode
        while numberOfEpisodes  < max_episodes:

            acumulatedRewards = [0,0]
            episodeSteps = 0
            terminal = False
            #line 3 -> initialize state s
            s = self.initializeState()
            #print(s)
            
            
            #line 4 and 11 -> repeat until s is terminal:
            while terminal is not True and episodeSteps < max_steps:
                #env.render()
                s,terminal,reward = self.step(s)
                #print(s, episodeSteps)
                episodeSteps += 1
                acumulatedRewards[0] += reward[0]
                acumulatedRewards[1] += reward[1]


            metrics.rewards1.append(acumulatedRewards[0])
            metrics.rewards2.append(acumulatedRewards[1])
            metrics.episodes.append(numberOfEpisodes)
 
            
            
            #print()
            #self.qtable (estados,ações,objetivos)
            
            self.episodeqtable = copy.deepcopy(self.qtable)
            self.polDict[numberOfEpisodes] = self.episodeqtable
            self.polIndex +=1
            
            metrics.pdict = self.polDict
            numberOfEpisodes+=1

        self.rewards0= metrics.rewards1
        self.rewards1 = metrics.rewards2
        if plot_paretofront:
            self.pareto_front_training(log,max_episodes,max_steps)



    def pareto_front_training(self,log,max_episodes,max_steps):
            ######################### PARETO TRAINING ################

        print(self.polDict.shape)
        
        self.currentepisode = 0
        q_set = self.polDict
        numberOfEpisodes = 0
        episodeSteps = 0

        #line 1 -> initialize q_set
        print("-> Pareto Training started <-")

        #line 2 -> for each episode
        
        while self.currentepisode  < max_episodes:
            print("Pareto ep:" + str(self.currentepisode))
            ParetoacumulatedRewards = [0,0,0]
            episodeSteps = 0
            


            #line 3 -> initialize state s
            s = self.initializeState()
            
            terminalpareto = False
            
            #line 4 and 11 -> repeat until s is terminal:
            while terminalpareto is not True and episodeSteps < max_steps:
                s,terminalpareto,reward = self.stepPareto(s)
                #print(s, episodeSteps)
                episodeSteps += 1
                ParetoacumulatedRewards[0] += reward[0]
                ParetoacumulatedRewards[1] += reward[1]
                
            


            metrics.paretor0.append(ParetoacumulatedRewards[0])
            metrics.paretor1.append(ParetoacumulatedRewards[1])



            self.currentepisode+=1

        self.paretorew1 = metrics.paretor1
        self.paretorew0 = metrics.paretor0
        if log:
            self.send_wandb_metrics()
            data = [[x, y] for (x, y) in zip(metrics.paretor0, metrics.paretor1)]
            table = wandb.Table(data=data, columns = ["x", "y"])
            wandb.log({"my_custom_plot_id" : wandb.plot.scatter(table, "x", "y", title="Custom Y vs X Scatter Plot")})


            metrics.close_wandb()

    def send_wandb_metrics(self):
        for e in range(self.numberofeps):
            
            log_dict3 = {
                    "rewards0":metrics.rewards1[e],
                    "rewards1":metrics.rewards2[e],
                    "paretoreward0":metrics.paretor0[e],
                    "paretoreward1":metrics.paretor1[e],
                    "episode":e



                }
           
        wandb.log(log_dict3)

    def stepPareto(self,s):
        
        self.actionsMethods.epsilon = 0 
        
        
                
            
                
        q_set = self.polDict[self.currentepisode][s]
        
        q_set = q_set[:, np.newaxis,: ]
                
        action = self.choose_action(s, q_set)
        next_state, reward, terminal, _ = self.env.step(action)
        next_s = ''.join(str(next_state))
        if next_s not in self.stateList:
            self.stateList.append(next_s)
        next_s = self.stateList.index(next_s)
        nd = self.update_non_dominated(s, action, next_s)
        self.n_visits[s, action] += 1
        self.avg_r[s, action] += (reward - self.avg_r[s, action]) / self.n_visits[s, action] 
        return next_s,terminal,reward
        
        










        


    def step(self,s):
        
        
        

        #line 5 -> Choose action a from s using a policy derived from the Qˆset’s
        
        q_set = self.compute_q_set(s)
        action = self.choose_action(s, q_set)
        
        #metrics for pareto plot
        self.qcopy = copy.deepcopy(q_set)



        #line 6 ->Take action a and observe state s0 ∈ S and reward vector r ∈ R
        next_state, reward, terminal, _ = self.env.step(action)
        next_s = self.flatten_observation(next_state)
        
        
        
        
        nd = self.update_non_dominated(s, action, next_s)
        
        
        #line 9 -> Update avg immediate reward
        self.n_visits[s, action] += 1

        self.avg_r[s, action] += (reward - self.avg_r[s, action]) / self.n_visits[s, action]

        self.actionsMethods.epsilon *= self.actionsMethods.epsilonDecrease
        

        return next_s,terminal,reward

    
    def compute_q_set(self, s):
        q_set = []
        for a in range(self.env.nA):
            nd_sa = self.non_dominated[s][a]
            rew = self.avg_r[s, a]
            
          
            q_set.append([rew + self.gamma*nd for nd in nd_sa])
        return np.array(q_set)


    def update_non_dominated(self, s, a, next_state):
        q_set_n = self.compute_q_set(next_state)
        # update for all actions, flatten
        solutions = np.concatenate(q_set_n, axis=0)

        # compute pareto front
        
        self.non_dominated[s][a] = self.actionsMethods.get_non_dominated(solutions)
        return self.non_dominated[s][a]

class actionMethods():
    def __init__(self,epsilon,epsilonDecrease):
        self.epsilon = epsilon
        self.epsilonDecrease = epsilonDecrease

    

    def get_action(self,s, q,env):
        q_values = self.compute_hypervolume(q, q.shape[0], ref_point)

        if np.random.rand() >= self.epsilon:
            
            return np.random.choice(np.argwhere(q_values == np.amax(q_values)).flatten())
        else:
            
            return env.action_space.sample()

    def compute_hypervolume(self,q_set, nA, ref):
        q_values = np.zeros(nA)
        for i in range(nA):
            # pygmo uses hv minimization,
            # negate rewards to get costs
            points = np.array(q_set[i]) * -1.
            hv = hypervolume(points)
            # use negative ref-point for minimization
            q_values[i] = hv.compute(ref*-1)
        return q_values


    def get_non_dominated(self,solutions):
        is_efficient = np.ones(solutions.shape[0], dtype=bool)
        
        for i, c in enumerate(solutions):
            if is_efficient[i]:
                # Remove dominated points, will also remove itself
                is_efficient[is_efficient] = np.any(solutions[is_efficient] > c, axis=1)
                # keep this solution as non-dominated
                is_efficient[i] = 1

        return solutions[is_efficient]

if __name__ == '__main__':
    #envinronment variables
    import gym
    from gym import wrappers
    from deepst import DeepSeaTreasure
    env = DeepSeaTreasure()
    numberOfStates = 121
    numberOfObjectives = 2
    epsilon = 1
    epsilonDecrease = 0.9997
    acMeth = actionMethods(epsilon,epsilonDecrease)
    ref_point = np.array([0, -25])
    gamma = 0.94

    #agent call
    agent = Pareto(env,acMeth, lambda s, q: acMeth.get_action(s, q, env), ref_point, nO=numberOfObjectives,nS = numberOfStates, gamma=gamma)
    #          (episodes,steps,wandblog,paretofrontier plot)
    agent.train(3500,1500,log = True,plot_paretofront=False)

    #metrics
    metrics.plotGraph()
    metrics.plot_pareto_frontier()

    print("-> Done <-")