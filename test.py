from NNmodels import DDPGCritic
from keras. models import load_model

x = DDPGCritic(3, 2)
y = DDPGCritic(3, 2)

x.save_weights('kk.h5')
y.load_weights('kk.h5')