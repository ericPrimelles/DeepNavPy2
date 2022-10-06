import tensorflow as tf
from keras.layers import Flatten
def flatten(x : tf.Tensor):
    
    return Flatten()(x)



if __name__ == '__main__':
    
    x = tf.ones((64, 3, 3))
    
    print(flatten(x))