from enum import Flag
import numpy as np
import time
from absl import flags
from absl import app
from MADQN import MADQN
from MADQNMix import MADQNMix
from MADDPG import MADDPG
from Env import DeepNav

Flags = flags.FLAGS

flags.DEFINE_string('alg', default='maddpg', help='Algorithm Selector')
flags.DEFINE_integer('n_agnts', default=10, help='No. of agents')



def main(argv):
    
    print(f'Training {Flags.alg} with {Flags.n_agnts} agents')
    env = DeepNav(Flags.n_agnts, 0)
    if Flags.alg == 'madqn':
        program = MADQN(env)
    
    elif Flags.alg == 'madqnmx':
        program = MADQNMix(env, 2)
    
    else:
        program = MADDPG(env)
    
    program.Train()


if __name__ == '__main__':
    
    np.random.seed(int(time.time()))
    app.run(main)