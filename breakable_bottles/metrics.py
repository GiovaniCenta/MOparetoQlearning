from msilib.schema import Class
from queue import Empty
import matplotlib.pyplot as plt
import datetime
import os

class metrics():
    def __init__(self, episodes, rewards1, rewards2,rewards3):
        self.episodes = episodes
        self.rewards1 = rewards1
        self.rewards2 = rewards2
        self.rewards3 = rewards3
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
        ax.set_title('Time penalty x Episodes')
        #plt.show()
        plt.savefig(self.path + '\\Time penalty x Episodes')
        
        fig, ax2 = plt.subplots()
        ax2.plot(self.episodes, self.rewards2)
        ax2.set_title('Bottles Delivered x Episodes')
        plt.savefig(self.path + '\\Bottles Delivered x Episodes')
        plt.show()

        fig, ax3 = plt.subplots()
        ax3.plot(self.episodes, self.rewards3)
        ax3.set_title('Potential x Episodes')
        plt.savefig(self.path + '\\Potential x Episodes')
        plt.show()
    
    def plot_p_front(self,Xs,Ys,obj1name,obj2name,actionIndex,maxY = True,maxX = True):
        
        
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
        if len(best_x) > len(best_y):
            for p in best_x:
                if p in best_y:
                    frontier.append(p)
        else:
             for p in best_y:
                if p in best_x:
                    frontier.append(p)    
            
        pf_X = [pair[0] for pair in frontier]
        pf_Y = [pair[1] for pair in frontier]    
        plt.scatter(Xs,Ys)
        plt.plot(pf_X, pf_Y)
        plt.xlabel(str(obj1name) + " for Action " + str(actionIndex) )
        plt.ylabel(str(obj2name) + " for Action " + str(actionIndex))
        plt.savefig(self.path + '//Pareto front - ' + str(obj1name) + ' x ' + str(obj2name) + " for action " + str(actionIndex))
        plt.show()
           
        



        
    def plot_pareto_frontier(self):
        '''Pareto frontier selection process'''
        
        #print(self.pdict)
        i = 0

        
        #print(self.pdict[i])
        c = self.pdict[i]

        
        
        for v in self.pdict.values():
            self.xA0.append(v[0][0][1])
            self.yA0.append(v[0][0][0])
        for v in self.pdict.values():
            self.xA1.append(v[1][0][1])
            self.yA1.append(v[1][0][0])
        for v in self.pdict.values():
            self.xA2.append(v[2][0][1])
            self.yA2.append(v[2][0][0])
        for v in self.pdict.values():
            self.xA3.append(v[2][0][1])
            self.yA3.append(v[2][0][2])
        
        #print(xA0)
        self.plot_p_front(self.xA0,self.yA0,"bottles delivered","time penalty",0)
        self.plot_p_front(self.xA1,self.yA1,"bottles delivered","time penalty",1)
        self.plot_p_front(self.xA2,self.yA2,"bottles delivered","time penalty",2)
        self.plot_p_front(self.xA3,self.yA3,"bottles delivered","potential",2)
        
        
        
    