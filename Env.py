# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 10:37:23 2019

@author: vyas
Assignment 2 of RL
| X | 1 | 2 | 3 |
| 4 | 5 | 6 | 7 |
| 8 | 9 | 10| 11|
|12 |13 | 14| X |

"""
import numpy as np
import matplotlib.pyplot as plt


def IsTerminating(current_state):
    if (current_state == 0):
        return "yes"
    elif (current_state == 15):
        return "yes"
    else:
        return "no"

def  GetReward(previous_state, current_state):
    if (current_state == 0):
        return 0
    elif (current_state == 15):
        return 0
    else:
        return -1

class GridPolicyValue:
    def __init__ (self, name, discount):
        self.name = name
        self.mvalues_sum = np.zeros(16)
        self.mvalues_count = np.zeros(16)
        self.mvalues = np.zeros(16)
        self.fvalues_sum = np.zeros(16)
        self.fvalues_count = np.zeros(16)
        self.fvalues = np.zeros(16)
        self.discount = discount
        self.episode_samples = 0
        return
    
    def UpdateValueFunction(self, first_or_multi, episode):
        self.episode_samples += 1
        #self.value_function.append()
        e_values = episode.GetEpisode()
        e_values = np.array(e_values)
        e_values = e_values[:: -1] # Reverse time (T-1, T-2...)
        #print (e_values)
        self.found_states = np.zeros(16) # Identify 
        #print("length: " , len(e_values))
        G = 0
        for i in  np.arange(len(e_values)):
            G += float(self.discount* float(e_values[i][2]))
            current_state = int(e_values[i][0])
            if i != len(e_values):
                sequence = e_values[i+1:]  # Get from t .. 0
                sequence = sequence[:,0]
                sequence = sequence.astype(int)
                #print("current:", current_state,  "seq: ",sequence)
                if current_state in sequence:
                    if (first_or_multi == "first"):
                        #print("print.. first visit ignoring", current_state, self.fvalues_sum[current_state])

                        self.mvalues_count[current_state] += 1
                        self.mvalues_sum [current_state] += G
                        self.mvalues[current_state] = self.mvalues_sum[current_state]/self.mvalues_count[current_state]
                        #print("print.. first visit ignoring", current_state, self.mvalues_sum[current_state])
                    continue
                else:
                    #print("updating: ", current_state)
                    self.mvalues_count[current_state] += 1
                    self.mvalues_sum [current_state] += G
                    self.mvalues[current_state] = self.mvalues_sum[current_state]/self.mvalues_count[current_state]
           
                    self.fvalues_count[current_state] += 1
                    self.fvalues_sum [current_state] += G
                    self.fvalues[current_state] = self.fvalues_sum[current_state]/self.fvalues_count[current_state]

        #    current_state = e_value[0]
        #self.PrintValues()
        return   
        
    def PrintValues(self):
        print("Multi-visit")
        print(" |  ", round(self.mvalues[0], 2), " | ", round(self.mvalues[1], 2), 
              " | ", round(self.mvalues[2], 2), " | ", round(self.mvalues[3], 2))
        print(" | ", round(self.mvalues[4], 2), " | ", round(self.mvalues[5], 2), 
              " | ", round(self.mvalues[6], 2), " | ", round(self.mvalues[7], 2))
        print(" | ", round(self.mvalues[8], 2), " | ", round(self.mvalues[9], 2), 
              " | ", round(self.mvalues[10], 2), " | ", round(self.mvalues[11], 2))
        print(" | ", round(self.mvalues[12], 2), " | ", round(self.mvalues[13], 2), 
              " | ", round(self.mvalues[14], 2), " | ", round(self.mvalues[15], 2))
        print("First visit")
        print(" | ", round(self.fvalues[0], 2), " | ", round(self.fvalues[1], 2), 
              " | ", round(self.fvalues[2], 2), " | ", round(self.fvalues[3], 2))
        print(" | ", round(self.fvalues[4], 2), " | ", round(self.fvalues[5], 2), 
              " | ", round(self.fvalues[6], 2), " | ", round(self.fvalues[7], 2))
        print(" | ", round(self.fvalues[8], 2), " | ", round(self.fvalues[9], 2), 
              " | ", round(self.fvalues[10], 2), " | ", round(self.fvalues[11], 2))
        print(" | ", round(self.fvalues[12], 2), " | ", round(self.fvalues[13], 2), 
              " |  ", round(self.fvalues[14], 2), " | ", round(self.fvalues[15], 2)) 
        return

    def GetMultiVisitValues(self):
        return self.mvalues
    
    def GetFirstVisitValues(self):
        return self.fvalues
    
ActionState = [ [0, 0, 0, 0, 0], # CurrentStqte = 0
                [0, 2, 1, 5, 1], # CurrentState = 1
                [1, 3, 2, 6, 2],
                [2, 3, 3, 7, 3],
                
                [4, 5, 0, 8, 4], #S = 4
                [4, 6, 1, 9, 5],
                [5, 7, 2, 10, 6],
                [6, 7, 3, 11, 7],
                
                [8, 9, 4, 12, 8],
                [8, 10, 5, 13, 9],
                [9, 11, 6, 14, 10],
                [10, 11, 7, 15, 11],
                
                [12, 13, 8, 12, 12],
                [12, 14, 9, 13, 13],
                [13, 15, 10, 14, 14],
                [15, 15, 15, 15, 15]
                ]

valid_actions = ['L', 'R', 'U', 'D', "Stay"]

equi_probable_policy = [
                        [0, 0, 0, 0, 1],
                        [1/3, 1/3, 0, 1/3, 0],
                        [1/3, 1/3, 0, 1/3, 0],
                        [1/2, 0, 1/2, 0, 0],
                        
                        [0, 1/3, 1/3, 1/3, 0],
                        [1/4, 1/4, 1/4, 1/4, 0],
                        [1/4, 1/4, 1/4, 1/4, 0],
                        [1/3, 0, 1/3, 1/3, 0],
                        
                        [0, 1/3, 1/3, 1/3, 0],
                        [1/4, 1/4, 1/4, 1/4, 0],
                        [1/4, 1/4, 1/4, 1/4, 0],
                        [1/3, 0, 1/3, 1/3, 0],

                        [0, 1/2, 1/2, 0, 0],
                        [1/3, 1/3, 1/3, 0, 0],
                        [1/3, 1/3, 1/3, 0, 0],
                        [0, 0, 0, 0, 1],
                        ]
assignment_policy = [
                        [0, 0, 0, 0, 1],
                        [0.7, 0.1, 0, 0.1, 0.1],
                        [0.7, 0.1, 0, 0.1, 0.1],
                        [0.45, 0, 0.45, 0, 0.1],
                        
                        [0, 0.1, 0.7, 0.1, 0.1],
                        [0.4, 0.1, 0.1, 0.1, 0],
                        [0.1, 0.4, 0.1, 0.4, 0],
                        [0.1, 0, 0.1, 0.7, 0.1],
                        
                        [0, 0.1, 0.7, 0.1, 0.1],
                        [0.4, 0.1, 0.1, 0.4, 0],
                        [0.1, 0.4, 0.1, 0.4, 0],
                        [0.1, 0, 0.1, 0.7, 0.1],

                        [0, 0.45, 0.45, 0, 0.1],
                        [0.1, 0.7, 0.1, 0, 0.1],
                        [0.1, 0.7, 0.1, 0, 0.1],
                        [0, 0, 0, 0, 1],
                        ]


def GetNextRandomAction(policy, current_state):
    policy_row = policy[current_state]

    r = np.random.rand(1)
    csum = np.cumsum(policy_row)
    new_state_index = 0
    #print ("GetNextRandomAction: " , current_state, " policy_row: " , policy_row, "\ncsum: " , csum, "Random: " , r)

    for i in np.arange(len(csum)):
        if (r > csum[i]):
            new_state_index = i+1
        else:
            #print("GetNextRandomAction:", valid_actions[new_state_index])
            return valid_actions[new_state_index]

    #print("GetNextRandomAction:", 4) # last state - stay
    return valid_actions[4]

def GetNextState(current_state, action):
    #print("GetNextState: ", current_state, "Action: ", action)
    return ActionState[current_state][valid_actions.index(action)]
    
    
class Episode:
    def __init__ (self, name):
        self.name = name
        self.episode_result = []
        self.elements_in_episode = 0
        
    def updateEpisode(self,state, Action, reward):
        episode = [state, Action, reward]
        self.episode_result.append(episode)
        self.elements_in_episode +=1
        return
    
    def Print(self):
        for i in np.arange(len(self.episode_result)):
            print("S= ", self.episode_result[i][0], ", A=", self.episode_result[i][1], 
                  "R= ", self.episode_result[i][2])
        return
    
    def GetEpisode(self):
        return self.episode_result
    
class Grid:
    def __init__ (self, name, policy):
        self.name = name
        self.policy = policy
        self.episode_num = 0
        self.episodes = []
        str_val = self.name + "episode_num" + str(self.episode_num)
        self.current_episode = Episode(str_val)
        self.SetRandomStartState()
        self.grid_pv = GridPolicyValue(name, 0.95)
        self.mvalues = []
        self.fvalues = []
        for i in np.arange(16):
            m = []
            f = []
            self.mvalues.append(m)
            self.fvalues.append(f)
        
    def SetRandomStartState(self):
        self.start = np.ceil(np.random.rand(1)*14)
        self.current_state = int(self.start[0])
        return
    
    def SetStartState(self, state):
        self.start = state
        self.current_state = state
        return

    def RunNewEpisode(self, is_random_start, start_state):
        #Stash old episode
        self.episode_num += 1
        self.episodes.append(self.current_episode)
        str_val = self.name + "episode_num" + str(self.episode_num)
        self.current_episode = Episode(str_val)
        
        if (is_random_start == "random"):
            self.SetRandomStartState()
        else:
            self.SetStartState(start_state)
        
        while (IsTerminating(self.current_state) == "no"):
            #print("RunNewEpisode: ", self.current_state)
            action = GetNextRandomAction(self.policy, self.current_state)
            cstate = self.current_state
            self.current_state = GetNextState(self.current_state, action)
            reward = GetReward(cstate, self.current_state)
            
            self.current_episode.updateEpisode(cstate, action, reward)
        
        self.grid_pv.UpdateValueFunction("first", self.current_episode)
        gm = self.grid_pv.GetMultiVisitValues()
        gf = self.grid_pv.GetFirstVisitValues()
        for i in  np.arange(16):
            self.mvalues[i].append(gm[i])
            self.fvalues[i].append(gf[i])
        
        return
        
    def PrintCurrentEpisode(self):
        self.current_episode.Print()
        return
    
    def RunForNEpisodes(self, total, is_random_start, start_state):
        i = 0
        while (i < total):
            self.RunNewEpisode(is_random_start, start_state)
            i += 1
        return
    def PrintValues(self):
        print("Values for: ", self.name)
        self.grid_pv.PrintValues()


    def Assignment2plots(self):
        fig = plt.figure(1)
        fig.set_figheight(15)
        fig.set_figwidth(15)
        ax2_str = self.name + " first-visit simulation run" + "for " + str(len(self.mvalues[0])) + " samples"
        ax1_str = self.name + " every-visit simulation run" + "for " + str(len(self.mvalues[0])) + " samples"


     
        ax1 = fig.add_subplot(2,1,1)
        ax2 = fig.add_subplot(2,1,2)
        ax1.set_xlabel("Episode number")
        ax2.set_xlabel("Episode number")
        ax1.set_ylabel("Value function")
        ax2.set_ylabel("value function")
        ax2.set_title(ax2_str)
        ax1.set_title(ax1_str)
        for i in np.arange(14):
            #print(self.mvalues[i])
            label_str = "state: " + str(i+1)
            ax1.plot(self.mvalues[i+1], label= label_str)  
            ax2.plot(self.fvalues[i+1], label= label_str)  
        ax1.legend()
        ax2.legend()
        file_name = self.name + "-" + str(len(self.mvalues[0]))
        plt.savefig(file_name)
        plt.show()
    
def Test():
    g = Grid("Assignment Policy",assignment_policy)

    #g = Grid("test",equi_probable_policy)
    #g.RunNewEpisode("fixed", 12)
    #g.PrintCurrentEpisode()
    for j in np.arange(200):
        for i in np.arange(14):
            g.RunForNEpisodes(1, "fixed", i+1)
    g.PrintValues()
    
    #return
    g2 = Grid("equiprobable Policy",equi_probable_policy)
    #g2.RunNewEpisode("fixed", 12)
    #g2.PrintCurrentEpisode()
    for j in np.arange(5):
        for i in np.arange(14):
            g2.RunForNEpisodes(1, "fixed", i+1)
    g2.PrintValues()
    
    g.Assignment2plots()
    #g2.Assignment2plots()
    
if __name__ == "__main__":
    # execute only if run as a script
    Test()