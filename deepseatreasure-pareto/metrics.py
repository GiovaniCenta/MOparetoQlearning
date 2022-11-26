from msilib.schema import Class
from queue import Empty
from re import X
import matplotlib.pyplot as plt
import datetime
import os
import wandb
from torch.utils.tensorboard import SummaryWriter

class metrics():
    def __init__(self, episodes, rewards1, rewards2):
        self.episodes = episodes
        self.rewards1 = rewards1
        self.rewards2 = rewards2
        self.paretor0 = []
        self.paretor1 = []
        self.nonDominatedPoints = []
        self.ndPoints =[]
        self.pdict = {}
        self.xA0 = []
        self.yA0 = []        
        self.xA1 = []
        self.yA1 = []
        self.xA2 = []
        self.yA2 = []
        self.xA3 = []
        self.yA3 = []
        self.count = 0
        self.path = ''
        self.createLogDir()

    def createLogDir(self):
        e = datetime.datetime.now()
        directory = e.strftime("%d#%m#%Y  %H-%M-%S")
        self.path = os.path.join(os.getcwd() + '\\log','log '+ directory)
        
        if not os.path.exists(self.path):
            os.mkdir(self.path)

    def plotGraph(self):
        
        
        fig, ax = plt.subplots()
        ax.plot(self.episodes, self.rewards1)
        ax.set_title('Treasure reward x Episodes')
        #plt.show()
        plt.savefig(self.path + '\\Treasure reward x Episodes')
        
        fig, ax2 = plt.subplots()
        ax2.plot(self.episodes, self.rewards2)
        ax2.set_title('Time penalty x Episodes')
        
        
        plt.savefig(self.path + '\\Time penalty x Episodes')
        plt.show()
    
    def plot_p_front(self,Xs,Ys,actionIndex,maxY = True,maxX = True):
        
        
        Xs = self.paretor0
        

        Ys = self.paretor1
        sorted_list = sorted([[Xs[i], Ys[i]] for i in range(len(Xs))], reverse=maxX)
        pareto_front = [sorted_list[0]]
        for pair in sorted_list[1:]:
            if maxY:
               
                if pair[1] >= pareto_front[-1][1]:
                    
                    pareto_front.append(pair)
            else:
                if pair[1] <= pareto_front[-1][1]:
                    pareto_front.append(pair)
       
        
       
       
       
        pf_X = []
        pf_Y = []
        best_y = [] 
        best_x = []
        for pair in pareto_front:
            
            
            if pair[1] not in pf_Y:
                best_y.append((pair[0],pair[1]))
                pf_Y.append(pair[1])
            else:
                pf_Y.append(pair[1])

            if pair[0] not in pf_X:
                best_x.append((pair[0],pair[1]))
                pf_X.append(pair[0])
            else:
                pf_X.append(pair[0])
            
        


        frontier = []
                
        for p in best_y:
            if p in best_x:
                frontier.append(p)      
            
        pf_X = [pair[0] for pair in frontier]
        pf_Y = [pair[1] for pair in frontier]    
        plt.scatter(Xs,Ys)
        plt.plot(pf_X, pf_Y)
        plt.xlabel("Treasure Reward  " )
        plt.ylabel("Time Penalty " )

        plt.show()
           
        



        
    def plot_pareto_frontier(self):
        '''Pareto frontier selection process'''
        
        #
        self.plot_p_front(self.paretor0,self.paretor1,3)
        
        
    def setup_wandb(self, project_name: str, experiment_name: str):
        self.experiment_name = experiment_name
        import wandb

        wandb.init(
            project=project_name,
            sync_tensorboard=True,
            config=self.get_config(),
            name=self.experiment_name,
            monitor_gym=True,
            save_code=True,

        )
        self.writer = SummaryWriter(f"{self.experiment_name}")
        # The default "step" of wandb is not the actual time step (gloabl_step) of the MDP
        wandb.define_metric("*", step_metric="global_step")

    def close_wandb(self):
        import wandb
        self.writer.close()
        wandb.finish()

    
    def get_config(self) -> dict:
        """Generates dictionary of the algorithm parameters configuration

        Returns:
            dict: Config
        """

    def plot_p_front2(self,Xs,Ys,actionIndex,maxY = True,maxX = True):
        import numpy as np
        """
        sorted_list = sorted([[Xs[i], Ys[i],Zs[i]] for i in range(len(Xs))], reverse=maxX)
        
        pareto_front = [sorted_list[0]]
        
        
        for pair in sorted_list[1:]:
            #pareto_front.append(pair)
            if maxY:
               
                if pair[1] >= pareto_front[-1][1]:
                    
                    pareto_front.append(pair)
            else:
                if pair[1] <= pareto_front[-1][1]:
                    pareto_front.append(pair)
     
        
        print(self.pdict)
        """
        frontier = []
        
        Xs = self.paretor0
        Ys = self.paretor1
        points = np.column_stack((Xs, Ys))
        uniques = np.unique(points,axis=0)
        
        inputPoints = uniques.tolist()
        paretoPoints, dominatedPoints = simple_cull(inputPoints, dominates)

        print ("*"*8 + " non-dominated answers " + ("*"*8))
        for p in paretoPoints:
            frontier.append(p)
            print (p)
        print ("*"*8 + " dominated answers " + ("*"*8))
        for p in dominatedPoints:
            pass
            #print (p)
        #print(arr)

        print(frontier)
        
  
        pf_Xx = [pair[0] for pair in frontier]
        pf_Yy = [pair[1] for pair in frontier]
          

        
      
        
        
        Xs = points[:,0]
        Ys = points[:,1]
        

        
        plt.plot(Ys,Xs)
        plt.scatter(pf_Yy,pf_Xx)
        plt.xlabel("Treasure Reward  " )
        plt.ylabel("Time Penalty " )
        

        
        
        
        
               
        plt.show()
def simple_cull(inputPoints, dominates): 
    paretoPoints = set()
    candidateRowNr = 0
    dominatedPoints = set()
    while True:
        candidateRow = inputPoints[candidateRowNr]
        inputPoints.remove(candidateRow)
        rowNr = 0
        nonDominated = True
        while len(inputPoints) != 0 and rowNr < len(inputPoints):
            row = inputPoints[rowNr]
            if dominates(candidateRow, row):
                # If it is worse on all features remove the row from the array
                inputPoints.remove(row)
                dominatedPoints.add(tuple(row))
            elif dominates(row, candidateRow):
                nonDominated = False
                dominatedPoints.add(tuple(candidateRow))
                rowNr += 1
            else:
                rowNr += 1

        if nonDominated:
            # add the non-dominated point to the Pareto frontier
            paretoPoints.add(tuple(candidateRow))

        if len(inputPoints) == 0:
            break
    return paretoPoints, dominatedPoints
def dominates(row, candidateRow):
    return sum([row[x] >= candidateRow[x] for x in range(len(row))]) == len(row) 