import random     
import matplotlib.pyplot as plt 
from collections import deque            
from keras.layers.merge import Add
from keras.models import Sequential, Model
from keras.layers.core import Dense, Flatten, Lambda
from keras.layers import Input
import tensorflow as tf
import numpy as np
import sys
import gym                               
env = gym.make('SpaceInvaders-ram-v0')   

program_, _model, _episodes, _batchsize, _epsilon, _frameSkip= sys.argv  

#_model = Linear, 2FC(Fully connected layers), 3FC_a(Fully connected layers), 3FC_b , DDQN .
#_episodes = # of episodes (default = 50000)
#_batchsize = tipically 32 or greater
#_epsilon = 0(greedy), 1(variant)
#_frameSkip = 2,3,4,5

def get_model(_model,*args):
  if _model == 'linear':
    return linear(*args)
  elif _model == '2FC':
    return two_fc(*args)
  elif _model == '3FC_a':
    return three_fc_a(*args)
  elif _model == '3FC_b':
    return three_fc_b(*args)
  elif _model == 'DDQN':
    return DDQN(*args)
  else:
    raise ValueError()
    
input_shape=(2,) + env.observation_space.shape
num_actions=env.action_space.n

def linear(*args):
    model = Sequential()
    model.add(Flatten())       # Flatten input so as to have no problems with processing
    model.add(Dense(env.action_space.n, init='uniform', activation='linear'))    # Same number of outputs as possible actions
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model

def two_fc(*args):
    model = Sequential()
    model.add(Dense(512, input_shape=(2,) + env.observation_space.shape, init='uniform', activation='relu'))
    model.add(Flatten())       # Flatten input so as to have no problems with processing
    model.add(Dense(env.action_space.n, init='uniform', activation='linear'))    # Same number of outputs as possible actions
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model

def three_fc_a(*args):
    model = Sequential()
    model.add(Dense(256, input_shape=(2,) + env.observation_space.shape, init='uniform', activation='relu'))
    model.add(Flatten())       # Flatten input so as to have no problems with processing
    model.add(Dense(128, init='uniform', activation='relu'))
    model.add(Dense(env.action_space.n, init='uniform', activation='linear'))    # Same number of outputs as possible actions
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model

def three_fc_b(*args):
    model = Sequential()
    model.add(Dense(512, input_shape=(2,) + env.observation_space.shape, init='uniform', activation='relu'))
    model.add(Flatten())       # Flatten input so as to have no problems with processing
    model.add(Dense(128, init='uniform', activation='relu'))
    model.add(Dense(env.action_space.n, init='uniform', activation='linear'))    # Same number of outputs as possible actions
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model

def DDQN(*args):
    inputs = Input(shape=input_shape)
    net = Dense(256, input_shape=(2,) + env.observation_space.shape, init='uniform', activation='relu')(inputs)
    net = Dense(128, init='uniform', activation='relu')(net)
    net = Flatten()(net)
    advt = Dense(256, activation='relu')(net)
    advt = Dense(env.action_space.n)(advt)
    value = Dense(256, activation='relu')(net)
    value = Dense(1)(value)
    # now to combine the two streams
    advt = Lambda(lambda advt: advt - tf.reduce_mean(advt, axis=-1, keep_dims=True))(advt)
    value = Lambda(lambda value: tf.tile(value, [1, env.action_space.n]))(value)
    final = Add()([value, advt])
    model = Model(inputs=inputs,outputs=final)
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model

model = get_model(_model)

def save_network(path):
    model.save(path)
    print("Successfully saved network.")

##################################PARAMETERS###########################################
D = deque()              # Register where the actions will be stored
path='/home/alessio94/Scrivania/Università/AI/AI2B/DQN_Final'
epsilon = _epsilon            # Probability of doing a random move
gamma = 0.9              # Discounted future reward. How much we care about steps further in time
mb_size = int(_batchsize)     # Learning minibatch size
episode = int(_episodes)      # Episodes
fs=int(_frameSkip)            # Frameskip
sum_reward = 0
max_tot_reward=0
cum_reward=0
Mean1=0
Mean2=0
Mean3=0
Mean4=0
x=[]
Y=[]
W=[]
T=[]
S=[]
q_val=[]
h_max=0
max_tot_reward=0

for z in range(1,episode+1):
    D=deque()
    observation = env.reset()                     # Game begins
    obs = np.expand_dims(observation, axis=0)     # (Formatting issues) Making the observation the first element of a batch of inputs 
    state = np.stack((obs, obs), axis=1)
    done = False
    epsilon = 1 - 0.95*(z/episode)
    tot_reward =0
    end=False
    tot_reward=0
    h=0
    while not(end) :#One single episode
        h+=1
        epsilon = 1 - 0.95*(z/episode)
        if z > 150000 and z < 180000 and epsilon == 1:
            if h < h_max-50:
                epsilon = (epsilon*0.5) + (epsilon*(h/h_max)*0.5)
        if z==1:
            Q = model.predict(state)
            Q[0]=0
        if h % fs == 0:#Frameskip
            if np.random.rand() <= epsilon :
                action = np.random.randint(0, env.action_space.n, size=1)[0]
            else:
                Q = model.predict(state)          # Q-values predictions
                action = np.argmax(Q)             # Move with highest Q-value is the chosen one            
            observation_new, reward, done, info = env.step(action)     # See state of the game, reward... after performing the action
            tot_reward+=reward
            obs_new = np.expand_dims(observation_new, axis=0)          # (Formatting issues)
            state_new = np.append(np.expand_dims(obs_new, axis=0), state[:, :1, :], axis=1)     # Update the input with the new state of the game
            D.append((state, action, reward, state_new, done))         # 'Remember' action and consequence
            state = state_new         # Update state
            if done:
                env.reset()                                   # Restart game if it's finished
                obs = np.expand_dims(observation, axis=0)     # (Formatting issues) Making the observation the first element of a batch of inputs 
                state = np.stack((obs, obs), axis=1)
                end = True
        if h > h_max:
            h_max=h
    if tot_reward > max_tot_reward:
        max_tot_reward = tot_reward
        Y.append(max_tot_reward)
        W.append(z)
    else:
        Y.append(max_tot_reward)
    if z > 3000:
        q_val.append(sum(Q[0])/6)
    S.append(tot_reward)
    if z == episode/4:
        Mean1 = (cum_reward/(episode/4))
        S=[]
    if z == episode/2:
        Mean2 = (sum(S))/(episode/4)
        S=[]
    if z == (episode/4 + episode/2):
        Mean3 = (sum(S))/(episode/4)
        S=[]
    if z == episode-2:
        Mean4 = (sum(S))/(episode/4)
        S=[]
    cum_reward+=tot_reward
    # SECOND STEP: Learning from the observations (Experience replay)
    
    if mb_size > len(D):
        mb_size = len(D) - 5
    minibatch = random.sample(D, mb_size)           # Sample some moves
    inputs_shape = (mb_size,) + state.shape[1:]
    inputs = np.zeros(inputs_shape)
    targets = np.zeros((mb_size, env.action_space.n))
    
    for i in range(0, mb_size):
        state = minibatch[i][0]
        action = minibatch[i][1]
        reward = minibatch[i][2]
        state_new = minibatch[i][3]
        done = minibatch[i][4]
        
    # Build Bellman equation for the Q function
        inputs[i:i+1] = np.expand_dims(state, axis=0)
        targets[i] = model.predict(state)
        Q_sa = model.predict(state_new)
        
        if done:
            targets[i, action] = reward
        else:
            targets[i, action] = reward + gamma * np.max(Q_sa)    
    # Train network to output the Q function
        model.train_on_batch(inputs, targets)
        
    print('Episode: '+str(z)+'  Reward: '+str(tot_reward)+' Max_tot_reward: '+str(max_tot_reward)+' Epsilon: '+str(epsilon))
    x.append(int(tot_reward))
    T.append(cum_reward)
c=[Mean1,Mean2,Mean3,Mean4]
save_network(path)
fig1=plt.figure()
plt.title('Rewards')
plt.plot(x,label='x=tot_reward')
plt.show(x)
fig1.savefig('Train_img/Reward_DQN.png')
fig2=plt.figure()
plt.title('Max reward during training')
plt.plot(Y)
plt.show(Y)
fig2.savefig('Train_img/Max_reward_DQN.png')
fig3=plt.figure()
plt.title('Cum_rewards')
plt.plot(T,label='Cumulative_reward')
plt.show(T)
fig3.savefig('Train_img/Cumulative_rew_DQN.png')
fig4=plt.figure()
plt.title('Q_val')
plt.plot(q_val)
plt.show(q_val)
fig4.savefig('Train_img/Q_val_DQN.png')
fig5=plt.figure()
plt.title('Mean')
plt.plot(c)
plt.show(c)
fig5.savefig('Train_img/Mean_DQN.png')
print('Mean1: '+str(Mean1)+' Mean2: '+str(Mean2)+' Mean3: '+str(Mean3)+' Mean4: '+str(Mean4))
print('Mb_size = ' + str(mb_size) + 'Frameskip = ' + str(fs))

