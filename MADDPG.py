from email import policy
from math import gamma
import tensorflow as tf
import numpy as np

import keras
from keras.losses import mean_squared_error
from keras.layers import Flatten
from keras.optimizers import Adam
from Env import DeepNav
#from DDPGAgent import MADDPGAgent
from replayBuffer import ReplayBuffer
from utils import flatten
from NNmodels import DDPGActor, DDPGCritic
from joblib import dump, load
import matplotlib.pyplot as plt
from DDPG_Agent import DDPGAgent

class MADDPG:
    
    def __init__(self, env : DeepNav,  n_epochs=1000, n_episodes=10, tau=0.005, 
                 gamma=0.99, l_r = 1e-5, bf_max_lenght=10000, bf_batch_size=64, path='models/DDPG/'):
        
        self.env = env
        self.obs_space = self.env.getStateSpec()
        self.action_space = self.env.getActionSpec()
        
        
        self.n_agents = env.n_agents
        self.n_epochs = n_epochs
        self.n_episodes = n_episodes
        self.bf_max_lenght = bf_max_lenght
        self.batch_size = bf_batch_size
        self.path = path
        #self.ou_noise = OUActionNoise(np)
        #self.agents = [MADDPGAgent(i, self.obs_space, self.action_space, gamma, l_r, tau)
                      # for i in range(self.n_agents)]
        self.agents = [
            DDPGAgent(agnt, self.obs_space, self.action_space, gamma, self.n_agents)
            for agnt in range(self.n_agents)
        ]
        self.rb = ReplayBuffer(env.getStateSpec(), env.getActionSpec(), self.n_agents,
                               self.bf_max_lenght, self.batch_size)
        self.gamma = gamma
        self.l_r = l_r
        self.tau = tau
        self.x_prev = np.zeros((self.n_agents, 2))
        
    def noise_reset(self):
        self.x_prev = np.zeros((self.n_agents, 2))
    
    @tf.function    
    def updateTarget(self, target_weights, weights, i):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * self.tau + a * (1 - self.tau))
            
       
    def save(self):
        for i in self.agents:
            _id = i.agent
            i.critic.save_weights(self.path + f'QNet_{_id}.h5')
            i.t_critic.save_weights(self.path + f'QTargetNet_{_id}.h5')
        
            i.actor.save_weights(self.path + f'ANet_{_id}.h5')
            i.t_actor.save_weights(self.path + f'ATargetNet_{_id}.h5')
            
    def load(self):
        for i in range(self.n_agents):
            _id = self.agents[i].agent
            self.agents[i].critic.load_weights(self.path + f'QNet_{_id}.h5')
            self.agents[i].t_critic.load_weights(self.path + f'QTargetNet_{_id}.h5')

            self.agents[i].actor.load_weights(self.path + f'ANet_{_id}.h5')
            self.agents[i].t_actor.load_weights(self.path + f'ATargetNet_{_id}.h5')
            
    
    def normalize(self, a):
        norm = np.linalg.norm(a)
        
        return a * 1 / norm    
    
    @tf.function
    def chooseAction(self, s : tf.Tensor, target : bool = False, training : bool = False):
            
            if s._rank() <= 2:
                s = tf.expand_dims(s, 0)
            
            if target:
                acts = tf.stack([
                self.agents[i].t_actor(s[:, i, :], training)
                for i in range(self.n_agents) 
                ])    
                return acts
            
            acts = tf.stack([
               self.agents[i].actor(s[:, i, :], training)
               for i in range(self.n_agents) 
            ])
            return acts
        
    def policy(self, s):
        a = np.squeeze(np.array(self.chooseAction(s)))
        
        x = (self.x_prev
            + 0.15 * (np.ones((self.n_agents, 2)) - self.x_prev) * 1e-2
            + np.full((self.n_agents, 2), 0.2) * np.sqrt(1e-2) * np.random.normal(size=(self.n_agents, 2)))
        
        self.x_prev = x
        
        
        return x + a    
        
    def Train(self):
        
        print('Starting Train')
        rwd = []
        for epoch in range(self.n_epochs):
            
            for episode in range(self.n_episodes):
                s = self.env.reset()
                
                reward = []
               
                ts = 0
                H=10000
                
                while 1:
                    
                    
                    a = self.policy(s)
                    #print(a, s)
                    #a = self.env.sample()
                    s_1, r, done = self.env.step(a)
                    
                    reward.append(r)
                    self.rb.store(s, a, r, s_1, done)
                    
                    if self.rb.ready:
                        s_s, a_s, r_s, s_1_s, dones_s = self.rb.sample()
                        
                        a_s = a_s.reshape((self.n_agents, 64, 2))
                        s_s = tf.convert_to_tensor(s_s)
                        #a_s = tf.convert_to_tensor(a_s)
                        r_s = tf.convert_to_tensor(r_s)
                        s_1_s = tf.convert_to_tensor(s_1_s)
                        dones_s = tf.convert_to_tensor(dones_s)
                        a_state = self.chooseAction(s_s, training=True)
                        t_a_state = self.chooseAction(s_1_s, True, True)
                        for i in range(self.n_agents):
                            self.agents[i].update(s_s, a_s, r_s, s_1_s, a_state, t_a_state)
                            self.updateTarget(self.agents[i].t_critic.variables, self.agents[i].critic.variables, i)
                            self.updateTarget(self.agents[i].t_actor.variables, self.agents[i].actor.variables, i)
                            
                        
                    s = s_1
                    ts +=1
                    
                    #fmt = '*' * int(ts*10/H)
                    #print(f'Epoch {i + 1} Episode {j + 1} |{fmt}| -> {ts}')
                    if done == 1 or ts > H:
                        
                        print(f'Epoch {epoch} Episode {episode} ended after {ts} timesteps Reward {np.mean(reward)}')
                        ts=0
                        rwd.append(np.mean(reward))
                        reward = []
                        self.noise_reset()
                        break
                    
                    
                
            self.save()
           
            dump(rwd, self.path + f'reward_epcohs_{i}.joblib')
                #print(f'Epoch: {i + 1} / {self.n_epochs} Episode {j + 1} / {self.n_episodes} Reward: {reward / ts}')        
                  
    
        

        
    def test(self):
        self.load()
        s = self.env.reset()
        ts = 0
        f = open(f'{self.path}/report.txt', 'w+')
        f.write('id,gid,x,y,dir_x,dir_y,radius,time\n')
        while 1:
            self.report(f)
            a = self.chooseAction(s)
            a = a.numpy()
            
            s, r, done = self.env.step(a)
            ts += 1
            
            if done or ts > 1000:
                f.close()
                break
    def plot(self, epoch=None):
        
        if epoch == None:
            epoch = self.n_epochs - 1
        rwds = load(f'{self.path}/reward_epcohs_{epoch}.joblib')
        
        plt.plot(rwds)
        plt.show()
    
    def report(self, f):   
        
        
        
        for i in range(self.n_agents):
                _id = self.agents[i].agent
                f.write(f'{_id},{self.env.getAgentPos(_id)[0]},{self.env.getAgentPos(_id)[1]},{self.env.getAgentVelocity(_id)[0]}, {self.env.getAgentVelocity(_id)[0]}, {self.env.radius}, {self.env.getGlobalTime()}\n')
            
           
        
        
if __name__ == '__main__':
    
     env = DeepNav(3, 0)


     p = MADDPG(env)
     p.Train()