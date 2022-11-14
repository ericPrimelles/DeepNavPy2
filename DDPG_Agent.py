import tensorflow as tf
import numpy as np
import keras
from NNmodels import DDPGActor, DDPGCritic
from keras.optimizers import Adam
from keras.losses import mean_squared_error
from utils import flatten

class DDPGAgent:
    
    def __init__(self, agnt, obs_space, action_space, gamma, n_agents):
        self.agent = agnt
        self.obs_space = obs_space
        self.action_space = action_space
        self.actor : keras.Model = DDPGActor(self.obs_space[1], self.action_space[1])
        self.t_actor : keras.Model = DDPGActor(self.obs_space[1], self.action_space[1])
        self.critic : keras.Model = DDPGCritic(self.obs_space[0] * self.obs_space[1], self.action_space[0])
        self.t_critic : keras.Model = DDPGCritic(self.obs_space[0] * self.obs_space[1], self.action_space[0])
        self.a_opt : Adam = Adam(1e-05)
        self.q_opt : Adam = Adam(1e-04)
        self.gamma = gamma
        self.n_agents = n_agents
        
        # Set the initial weigths equally for actors and critics
        self.t_actor.set_weights(self.actor.get_weights())
        self.t_critic.set_weights(self.critic.get_weights())
        
    @tf.function
    def update(self, s, a, r, s_1, a_state, t_a_state):
        
                
        i = self.agent
        
        s_agnt = s[:, i]
        a_agnt = a[:, i]
        r_agnt = r[:, i]
        s_1_agnt = s_1[:, i]
        #agnt = self.agents[i]
        
        with tf.GradientTape() as tape:
        
            acts = t_a_state
            
            

            y = r_agnt + self.gamma * self.t_critic([flatten(s_1), [tf.squeeze(k) for k in tf.split(acts, self.n_agents)]])

            q_value = self.critic([flatten(s), [tf.squeeze(k) for k in tf.split(a, self.n_agents)]])
            q_loss = tf.math.reduce_mean(tf.math.square(y - q_value))   

        q_grad = tape.gradient(q_loss, self.critic.trainable_variables)
        self.q_opt.apply_gradients(zip(q_grad, self.critic.trainable_variables))
        
        #acts = self.chooseAction(s, training=True)

        with tf.GradientTape(True) as tape:

            #act = agnt['a_n'](s_agnt, training=True)
            acts = a_state
            act = self.actor(s_agnt, training=True)
            acts = tf.split(acts, self.n_agents)
            acts[self.agent] = act
            q_values = self.critic([flatten(s), [tf.squeeze(k) for k in acts]])
            loss = -tf.reduce_mean(q_values)
        a_grad = tape.gradient(loss, self.actor.trainable_variables)
        
        self.a_opt.apply_gradients(zip(a_grad, self.actor.trainable_variables))

        