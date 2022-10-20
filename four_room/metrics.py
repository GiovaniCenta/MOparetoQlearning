from msilib.schema import Class
from queue import Empty
from re import X
import matplotlib.pyplot as plt
import datetime
import os
from mistune import markdown
import numpy as np
import pandas as pd
class metrics():
    def __init__(self, episodes, rewards1, rewards2,rewards3):
        self.episodes = episodes
        self.rewards1 = rewards1
        self.rewards2 = rewards2
        self.rewards3 = []
        self.nonDominatedPoints = []
        self.ndPoints =[]
        self.pdict = {}
        self.xA0 = []
        self.yA0 = []
        self.zA0 = []        
        self.xA1 = []
        self.yA1 = []
        self.zA1 = []  
        self.xA2 = []
        self.yA2 = []
        self.zA2 = [] 
        self.xA3 = []
        self.yA3 = []
        self.zA3 = [] 
        self.path = ''
        self.createLogDir()

    def createLogDir(self):
        e = datetime.datetime.now()
        directory = e.strftime("%d#%m#%Y  %H-%M-%S")
        self.path = os.path.join(os.getcwd() + '\\log','log '+ directory)
        os.mkdir(self.path)


    def plotGraph(self):
        
        
        fig, ax = plt.subplots()
        ax.plot(self.episodes, self.rewards1)
        ax.set_title('Enemy damage x Episodes')
        plt.savefig(self.path + '\\Enemy damage x Episodes')
        plt.show()
        
        fig, ax2 = plt.subplots()
        ax2.plot(self.episodes, self.rewards2)
        ax2.set_title('Gold gain x Episodes')
        plt.savefig(self.path + '\\Gold gain x Episodes')
        plt.show()
        
        fig, ax3 = plt.subplots()
        ax3.plot(self.episodes, self.rewards3)
        ax3.set_title('Gem gain x Episodes')
        plt.savefig(self.path + '\\Gem gain x Episodes')
        plt.show()
    
    def plot_p_front(self,Xs,Ys,Zs,actionIndex,obj1name,obj2name,obj3name,maxY = True,maxX = True):
        
        
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
     
        frontier = []
        
        points = np.column_stack((Xs, Ys, Zs))
        uniques = np.unique(points,axis=0)
        
        inputPoints = uniques.tolist()
        paretoPoints, dominatedPoints = simple_cull(inputPoints, dominates)

        print ("*"*8 + " non-dominated answers " + ("*"*8))
        for p in paretoPoints:
            frontier.append(p)
            print (p)
        print ("*"*8 + " dominated answers " + ("*"*8))
        for p in dominatedPoints:
            print (p)
        #print(arr)
        
       
        
        
        
        

        print(frontier)
        
        
            
            

        
        

    

        
        
     
        
        pf_Xx = [pair[0] for pair in frontier]
        pf_Yy = [pair[1] for pair in frontier]
        pf_Zz = [pair[2] for pair in frontier]    
        
        



        
      

        ax = plt.axes(projection='3d')
        
      
        ax.plot3D(pf_Xx, pf_Yy, pf_Zz,color= 'red',marker='o',label = 'pareto frontier for action '+ str(actionIndex))
        
        
        Xs = points[:,0]
        Ys = points[:,1]
        Zs = points[:,2]

        
        ax.scatter3D(Xs, Ys, Zs,color= 'blue',marker='x') 
        

        
        ax.legend()

        ax.set_xlabel(obj3name, fontsize=10, rotation=150)
        ax.set_ylabel(obj2name, fontsize=10, rotation=150)
        ax.set_zlabel(obj1name, fontsize=10, rotation=60)
        ax.yaxis._axinfo['label']['space_factor'] = 3.0
               
        plt.show()


          
    def plot_pareto_frontier(self):
 
        for v in self.pdict.values():
            #print(v)
            self.yA0.append(v[0][0][2])
            self.zA0.append(v[0][0][0])
            self.xA0.append(v[0][0][1])
        for v in self.pdict.values():
            self.yA1.append(v[1][0][2])
            self.zA1.append(v[1][0][0])
            self.xA1.append(v[1][0][1])
        for v in self.pdict.values():
            self.yA2.append(v[2][0][2])
            self.zA2.append(v[2][0][0])
            self.xA2.append(v[2][0][1])
        for v in self.pdict.values():
            self.yA3.append(v[3][0][2])
            self.zA3.append(v[3][0][0])
            self.xA3.append(v[3][0][1])
        
        #print(xA0)
        self.plot_p_front(self.xA0,self.yA0,self.zA0,0,"item1","item2","item3")
        self.plot_p_front(self.xA1,self.yA1,self.zA1,1,"item1","item2","item3")
        self.plot_p_front(self.xA2,self.yA2,self.zA2,2,"item1","item2","item3")
        self.plot_p_front(self.xA3,self.yA3,self.zA3,3,"item1","item2","item3")

# below code source: https://code.activestate.com/recipes/578287-multidimensional-pareto-front/
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
        
        
    