import os
from tensorflow import keras
from keras.layers import Dense

class ActorCriticNetwork(keras.Model):
    def __int__(self, n_actions, fc1_dims=1024, fc2_dims=512,
                name='actor_critic', chkpt_dir='tmp/actor_critic'):
        super(ActorCriticNetwork, self).__init__()
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_ac')

        #define the two common layers
        self.fc1 = Dense(self.fc1_dims, activation='relu') 
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        #specify the two independent outputs
        self.v = Dense(1, activation = None) #for value function
        self.pi = Dense(n_actions, activation = 'softmax') #for policy, need to sum up to 1 as they are probabilities

    def call(self, state):
        value = self.fc1(state)
        value = self.fc2(value)

        v = self.v(value)
        pi = self.pi(value)

        return v, pi