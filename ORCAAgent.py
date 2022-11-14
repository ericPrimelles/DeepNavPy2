import numpy as np
from Env import DeepNav

class ORCAAgent:
    
    def __init__ (self, id : int, env : DeepNav):
        self.env = env
        self.id = id
    
    def act(self, s):
        pos = np.array(self.env.getAgentPos(self.id))
        goal = np.array(self.env.getAgentGoal(self.id))
        a = goal - pos
        norm = np.linalg.norm(a)
        
        return a * 1 / norm
    
if __name__ == '__main__':
    
    env = DeepNav(2, 0)
    
    a = ORCAAgent(0, env)
    
    print(a.act())