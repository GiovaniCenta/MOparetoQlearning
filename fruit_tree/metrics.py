from msilib.schema import Class
from queue import Empty
from re import X
import matplotlib.pyplot as plt
import datetime
import os

class metrics():
    def __init__(self, episodes, rewards1, rewards2,rewards3, rewards4,rewards5, rewards6):
        self.episodes = episodes
        self.rewards1 = rewards1
        self.rewards2 = rewards2
        self.rewards3 = rewards3
        self.rewards4 = rewards4
        self.rewards5 = rewards5
        self.rewards6 = rewards6
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
        self.xA4 = []
        self.yA4 = []
        self.xA5 = []
        self.yA5 = []
        self.xA6 = []
        self.yA6 = []
        self.xA7 = []
        self.yA7 = []
        self.createLogDir()

    def createLogDir(self):
        e = datetime.datetime.now()
        directory = e.strftime("%d#%m#%Y  %H-%M-%S")
        self.path = os.path.join(os.getcwd() + '\\log','log '+ directory)
        
        if not os.path.exists(self.path):
            os.mkdir(self.path)


    def plotGraph(self):
        #nutrients = ["Protein","Carbs", "Fats", "Vitamins", "Minerals", "Water"]
        
        fig, ax = plt.subplots()
        ax.plot(self.episodes, self.rewards1)
        ax.set_title('Protein x Episodes')
        plt.savefig(self.path + '\\Protein x Episodes')
        plt.show()
        
        fig, ax2 = plt.subplots()
        ax2.plot(self.episodes, self.rewards2)
        ax2.set_title('Carbs x Episodes')
        plt.savefig(self.path + '\\Carbs x Episodes')
        plt.show()
        


        fig, ax3 = plt.subplots()
        ax3.plot(self.episodes, self.rewards3)
        ax3.set_title('Fats x Episodes')
        plt.savefig(self.path + '\\Fats x Episodes')
        plt.show()
        

        fig, ax4 = plt.subplots()
        ax4.plot(self.episodes, self.rewards4)
        ax4.set_title('Vitamins x Episodes')
        plt.savefig(self.path + '\\Vitamins x Episodes')
        plt.show()
        
        fig, ax5 = plt.subplots()
        ax5.plot(self.episodes, self.rewards5)
        ax5.set_title('Minerals x Episodes')
        plt.savefig(self.path + '\\Minerals x Episodes')
        plt.show()
        
        fig, ax6 = plt.subplots()
        ax6.plot(self.episodes, self.rewards6)
        ax6.set_title('Water x Episodes')
        plt.savefig(self.path + '\\Water x Episodes')
        plt.show()
        
    
    def plot_p_front(self,Xs,Ys,obj1,obj2,actionIndex,maxY = True,maxX = True):
        
        
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
        for p in best_x:
            if p in best_y:
                frontier.append(p)     
            
        pf_X = [pair[0] for pair in frontier]
        pf_Y = [pair[1] for pair in frontier]    
        plt.scatter(Xs,Ys)
        plt.plot(pf_X, pf_Y)
        plt.xlabel(str(actionIndex)  + " path for " + str(obj1))   
        plt.ylabel(str(actionIndex)  +  " path for " + str(obj2)) 
           
   
        plt.savefig(self.path + '//Pareto front - ' + str(obj1) + ' x ' + str(obj2) + " for action " + str(actionIndex))    

        plt.show()
           
        



        
    def plot_pareto_frontier(self):
        '''Pareto frontier selection process'''
        
        
        i = 0

        
        
        c = self.pdict[i]
        #nutrients = ["Protein","Carbs", "Fats", "Vitamins", "Minerals", "Water"]

        
        #v["direction", , "type"]
        for v in self.pdict.values():
            self.xA0.append(v[0][0][0])
            self.yA0.append(v[0][0][1])
            
        for v in self.pdict.values():
            self.xA1.append(v[1][0][0])
            self.yA1.append(v[1][0][1])
        for v in self.pdict.values():
            self.xA2.append(v[0][0][2])
            self.yA2.append(v[0][0][3])
        for v in self.pdict.values():
            self.xA3.append(v[1][0][2])
            self.yA3.append(v[1][0][3])
        for v in self.pdict.values():
            self.xA4.append(v[0][0][4])
            self.yA4.append(v[0][0][5])
        for v in self.pdict.values():
            self.xA5.append(v[1][0][4])
            self.yA5.append(v[1][0][5])
        
        for v in self.pdict.values():
            self.xA6.append(v[0][0][1])
            self.yA6.append(v[0][0][5])
        for v in self.pdict.values():
            self.xA7.append(v[1][0][1])
            self.yA7.append(v[1][0][5])
            
        
        
        self.plot_p_front(self.xA0,self.yA0,"Carbs","Proteins","left")
        self.plot_p_front(self.xA1,self.yA1,"Carbs","Proteins","rigth")
        self.plot_p_front(self.xA2,self.yA2,"Fats","Vitamins","rigth")
        self.plot_p_front(self.xA3,self.yA3,"Fats","Vitamins","left")
        self.plot_p_front(self.xA4,self.yA4,"Minerals", "Water","rigth")
        self.plot_p_front(self.xA5,self.yA5,"Minerals", "Water","left")
        self.plot_p_front(self.xA6,self.yA6,"Carbs", "Water","rigth")
        self.plot_p_front(self.xA7,self.yA7,"Carbs", "Water","left")
      
        
        
        
    