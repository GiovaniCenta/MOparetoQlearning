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

metrics = met([],[],[],[],[],[],[])

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

        self.polDict = {}
        self.polIndex = 0
        

    def initializeState(self):
        state = self.env.reset()

        s = ''.join(str(state))

        if s not in self.stateList:
            self.stateList.append(s)
        
        s = self.stateList.index(s)
        
        return {'observation':s,'terminal':False}


    def train(self,max_episodes,max_steps):
        numberOfEpisodes = 0
        episodeSteps = 0

        #line 1 -> initialize q_set
        print("-> Training started <-")
        #line 2 -> for each episode
        while numberOfEpisodes  < max_episodes:

            acumulatedRewards = [0,0,0,0,0,0]
            episodeSteps = 0

            #line 3 -> initialize state s
            s = self.initializeState()
            #print(s)
            
            
            #line 4 and 11 -> repeat until s is terminal:
            while s['terminal'] is not True and episodeSteps < max_steps:
                #env.render()
                s = self.step(s)
                #print(s, episodeSteps)
                episodeSteps += 1
                acumulatedRewards[0] += s['reward'][0]
                acumulatedRewards[1] += s['reward'][1]
                acumulatedRewards[2] += s['reward'][2]
                acumulatedRewards[3] += s['reward'][3]
                acumulatedRewards[4] += s['reward'][4]
                acumulatedRewards[5] += s['reward'][5]

            metrics.rewards1.append(acumulatedRewards[0])
            metrics.rewards2.append(acumulatedRewards[1])
            metrics.rewards3.append(acumulatedRewards[2])
            metrics.rewards4.append(acumulatedRewards[3])
            metrics.rewards5.append(acumulatedRewards[4])
            metrics.rewards6.append(acumulatedRewards[5])
            metrics.episodes.append(numberOfEpisodes)
            numberOfEpisodes+=1
            #print(numberOfEpisodes)
        
        metrics.pdict = self.polDict


    def step(self,state):
        s = state['observation']
        
        

        #line 5 -> Choose action a from s using a policy derived from the Qˆset’s
        
        q_set = self.compute_q_set(s)
        action = self.choose_action(s, q_set)
        
        #metrics for pareto plot
        self.qcopy = copy.deepcopy(q_set)
        self.polDict[self.polIndex] = self.qcopy
        self.polIndex +=1


        #line 6 ->Take action a and observe state s0 ∈ S and reward vector r ∈ R
        next_state, reward, terminal, _ = self.env.step(action)

        
        #transform state to string
        next_s = ''.join(str(next_state))

        if next_s not in self.stateList:
            self.stateList.append(next_s)
        next_s = self.stateList.index(next_s)
        
        #line 8 -> . Update ND policies of s' in s
        nd = self.update_non_dominated(s, action, next_s)
        metrics.ndPoints.append(nd)
        
        #line 9 -> Update avg immediate reward
        self.n_visits[s, action] += 1

        self.avg_r[s, action] += (reward - self.avg_r[s, action]) / self.n_visits[s, action]

        self.actionsMethods.epsilon *= self.actionsMethods.epsilonDecrease

        return {'observation': next_s,
                'terminal': terminal,
                'reward': reward}

    
    def compute_q_set(self, s):
        q_set = []
        for a in range(self.env.nA):
            nd_sa = self.non_dominated[s][a]
            rew = self.avg_r[s, a]
            q_set.append([rew + self.gamma*nd for nd in nd_sa])
        return np.array(q_set)


    def update_non_dominated(self, s, a, s_n):
        q_set_n = self.compute_q_set(s_n)
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
    from fruit_tree import FruitTreeEnv
    env = FruitTreeEnv()
    numberOfStates = 762
    numberOfObjectives = 6
    epsilon = 1
    epsilonDecrease = 0.999
    acMeth = actionMethods(epsilon,epsilonDecrease)
    ref_point = np.array([0, -1,-50,-20,-100,-10])
    gamma = 0.9

    #agent call
    agent = Pareto(env,acMeth, lambda s, q: acMeth.get_action(s, q, env), ref_point, nO=numberOfObjectives,nS = numberOfStates, gamma=gamma)
    agent.train(2000,400)

    #metrics
    metrics.plotGraph()
    metrics.plot_pareto_frontier()
    print("-> Done <-")