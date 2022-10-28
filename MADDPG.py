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
            {   
                'id' : agnt,
                'a_n' : DDPGActor(self.obs_space[1], self.action_space[1]),
                'target_a_n' : DDPGActor(self.obs_space[1], self.action_space[1]),
                'q_n' : DDPGCritic(self.obs_space[0] * self.obs_space[1], self.action_space[0]),
                'target_q_n' : DDPGCritic(self.obs_space[0] * self.obs_space[1], self.action_space[0]),
                'loss_fn' : mean_squared_error
            }
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
        
    def updateTarget(self):
        
            
        for i in self.agents:
            a_w = np.array(i['a_n'].weights, dtype=object)
            q_w = np.array(i['q_n'].weights, dtype=object)
            a_t_w = np.array(i['target_a_n'].weights, dtype=object)
            q_t_w = np.array(i['target_q_n'].weights, dtype=object)
            
            
            i['target_a_n'].set_weights(a_t_w * self.tau + (1 - self.tau) * a_w)
            i['target_q_n'].set_weights(q_t_w * self.tau + (1 - self.tau) * q_w) 

    def save(self):
        for i in self.agents:
            _id = i['id']
            i['q_n'].save_weights(self.path + f'QNet_{_id}.h5')
            i['target_q_n'].save_weights(self.path + f'QTargetNet_{_id}.h5')
        
            i['a_n'].save_weights(self.path + f'ANet_{_id}.h5')
            i['target_a_n'].save_weights(self.path + f'ATargetNet_{_id}.h5')
            
    def load(self):
        for i in range(self.n_agents):
            _id = self.agents[i]['id']
            self.agents[i]['q_n'].load_weights(self.path + f'QNet_{_id}.h5')
            self.agents[i]['target_q_n'].load_weights(self.path + f'QTargetNet_{_id}.h5')

            self.agents[i]['a_n'].load_weights(self.path + f'ANet_{_id}.h5')
            self.agents[i]['target_a_n'].load_weights(self.path + f'ATargetNet_{_id}.h5')
            
    
    def normalize(self, a):
        norm = np.linalg.norm(a)
        
        return a * 1 / norm    
    
    
    def chooseAction(self, s : tf.Tensor, target : bool = False, training : bool = False):
        if s.ndim == 2:
            s = s.reshape((1, s.shape[0], s.shape[1]))
            
        actions = []
              
        
        for i in range(self.n_agents):
            x = s[:,i, :]
            #x = x.reshape((self.batch_size, 1, self.obs_space[1]))
            
            if not target:
            
                actions.append(self.normalize(self.agents[i]['a_n'](x, training)))    
            
            else:
                actions.append(self.normalize(self.agents[i]['target_a_n'](x, training)))
        return actions
    
    
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
        for i in range(self.n_epochs):
            for j in range(self.n_episodes):
                s = self.env.reset()
                
                reward = []
               
                ts = 0
                H=100
                
                while 1:
                    
                    
                    a = self.policy(s)
                    #print(a, s)
                    #a = self.env.sample()
                    s_1, r, done = self.env.step(a)
                    
                    reward.append(r)
                    self.rb.store(s, a, r, s_1, done)
                    
                    if self.rb.ready:
                        self._learn(self.rb.sample())
                        self.updateTarget()
                        
                    s = s_1
                    ts +=1
                    
                    #fmt = '*' * int(ts*10/H)
                    #print(f'Epoch {i + 1} Episode {j + 1} |{fmt}| -> {ts}')
                    if done == 1 or ts > H:
                        
                        print(f'Epoch {i + 1} Episode {j + 1} ended after {ts} timesteps Reward {np.mean(reward)}')
                        ts=0
                        rwd.append(np.mean(reward))
                        reward = []
                        self.noise_reset()
                        break
                    
                    
                
            self.save()
           
            dump(rwd, self.path + f'reward_epcohs_{i}.joblib')
                #print(f'Epoch: {i + 1} / {self.n_epochs} Episode {j + 1} / {self.n_episodes} Reward: {reward / ts}')        
                  
          
    def _learn(self, sampledBatch):
        s, a, r, s_1, dones = sampledBatch
        a = a.reshape((self.n_agents, 64, 2))
        s = tf.convert_to_tensor(s)
        #a = tf.convert_to_tensor(a)
        r = tf.convert_to_tensor(r)
        s_1 = tf.convert_to_tensor(s_1)
        dones = tf.convert_to_tensor(dones)
        
        
        for i in range(self.n_agents):
            s_agnt = s[:, i]
            a_agnt = a[i]
            r_agnt = r[:, i]
            s_1_agnt = s_1[:, i]
            dones_agnt = dones[i]
            agnt = self.agents[i]
            opt = Adam(self.l_r)
            with tf.GradientTape() as tape:
            
                acts = self.chooseAction(s_1, True, True)
                
                y = r_agnt + self.gamma * agnt['target_q_n']([flatten(s_1), [acc for acc in acts]])
                
                q_value = agnt['q_n']([flatten(s), [acc for acc in a]])
                q_loss = tf.math.reduce_mean(tf.math.square(y - q_value))   
                
            q_grad = tape.gradient(q_loss, agnt['q_n'].trainable_variables)
            opt.apply_gradients(zip(q_grad, agnt['q_n'].trainable_variables))
            
            acts = self.chooseAction(s, training=True)
            
            with tf.GradientTape(True) as tape:
                
                act = agnt['a_n'](s_agnt, training=True)
                q_values = agnt['q_n'](([
                    flatten(s),
                    [acc for acc in acts[0:i]],
                    act,
                    [acc for acc in acts[i+1:]],
                ]))
                loss = -tf.reduce_mean(q_values)
            a_grad = tape.gradient(loss, agnt['a_n'].trainable_variables)
            opt.apply_gradients(zip(a_grad, agnt['a_n'].trainable_variables))
            
                       
    def test(self):
        self.load()
        s = self.env.reset()
        ts = 0
       
        while 1:
            self.report()
            a = self.chooseAction(s)
            s, r, done = self.env.step(a[0])
            ts += 1
            
            if done or ts > 1000:
                
                break
    def plot(self, epoch=None):
        
        if epoch == None:
            epoch = self.n_epochs - 1
        rwds = load(f'{self.path}/reward_epcohs_{epoch}.joblib')
        
        plt.plot(rwds)
        plt.show()
    
    def report(self):   
        f = open(f'{self.path}/report.txt', 'w+')
        f.write('id,gid,x,y,dir_x,dir_y,radius,time\n')
        
        for i in range(self.n_agents):
                _id = self.agents[i]['id']
                f.write(f'{_id},{self.env.getAgentPos(_id)[0]},{self.env.getAgentPos(_id)[1]},{self.env.getAgentVelocity(_id)[0]}, {self.env.getAgentVelocity(_id)[0]}, {self.env.radius}, {self.env.getGlobalTime()}\n')
            
        f.close()   
        
        
if __name__ == '__main__':
    
     env = DeepNav(3, 0)


     p = MADDPG(env)
     p.Train()
     p.test()
     p.plot()