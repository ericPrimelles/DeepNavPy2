from venv import create
import tensorflow as tf
import keras
from keras import layers
from keras.initializers import GlorotNormal
from keras.regularizers import l2

def qNetFC(input_shape, output_shape):
    
    inputs = layers.Input(shape=input_shape)
    
    # Hidden layers
    lyr1 = layers.Dense(32, activation='relu')(inputs)
    lyr1 = layers.Dropout(0.5)(lyr1)
    lyr1 = layers.BatchNormalization()(lyr1)
    lyr2 = layers.Dense(64, activation='relu')(lyr1)
    lyr2 = layers.Dropout(0.5)(lyr2)
    lyr2 = layers.BatchNormalization()(lyr2)
    lyr3 = layers.Dense(32, activation='relu')(lyr2)
    lyr3 = layers.Dropout(0.5)(lyr3)
    lyr3 = layers.BatchNormalization()(lyr3)
    
    #Output
    action = layers.Dense(output_shape, activation='linear')(lyr3)
    action = layers.BatchNormalization()(action)
    return keras.Model(inputs=inputs, outputs=action)
def DDPGActor(input_shape, output_shape):
    
    return qNetFC(input_shape=input_shape, output_shape=output_shape)

def DDPGCritic(input_obs, input_action):

    # State net
    
    state_input = layers.Input(input_obs)
    state_output = layers.Dense(8, activation='relu')(state_input)
    
    # Action Net
    act_inputs = [layers.Input(2) for i in range(input_action)]
    action_in = layers.Concatenate()(act_inputs)
    action_outputs = layers.Dense(8, activation='relu')(action_in)
    
    # Input
    inputs = layers.Concatenate()([state_output, action_outputs])
    lyr1 = layers.Dense(32, activation='relu')(inputs)
    lyr1 = layers.Dropout(0.5)(lyr1)
    lyr1 = layers.BatchNormalization()(lyr1)
    lyr2 = layers.Dense(64, activation='relu')(lyr1)
    lyr2 = layers.Dropout(0.5)(lyr2)
    lyr2 = layers.BatchNormalization()(lyr2)
    lyr3 = layers.Dense(32, activation='relu')(lyr2)
    lyr3 = layers.Dropout(0.5)(lyr3)
    lyr3 = layers.BatchNormalization()(lyr3)
    value = layers.Dense(1, activation='relu')(lyr3)
    
    return keras.Model(inputs=[state_input, act_inputs], outputs=value)
if __name__ == '__main__':
    
    x = DDPGActor(3, 3)
    
    print(type(x.variables))