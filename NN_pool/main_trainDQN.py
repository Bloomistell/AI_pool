# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 19:43:19 2023

@author: Jolec
"""


import sys

import collisions
import event
import gamestate
import numpy as np
import gymnasium as gym
import seaborn as sns
import matplotlib.pyplot as plt 
from tqdm import tqdm 
import pickle 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import poolenv
import DQN 



CFG = {
    'CONTINUOUS_ACTION_SPACE' : False,
    'CYLINDRICAL' : False,
    'N_BINS_ANGLE' : 360,
    'N_BINS_FORCE' : 16,
    'ALPHA'        : 0,
    'BETA'         : 0,
    'DELTA'        : 20, 
    'LOST_REWARD'  : 0, 
    'NUM_BALLS'    : 2,
    'MAX_TURNS' : 5,
    'EPISODES' : 5000,
    'LR' : 0.005,
    'MEM_SIZE' : 100000,
    'BATCH_SIZE' : 64,
    'GAMMA' : 0.95,
    'EXPLORATION_MAX' : 1.0,
    'EXPLORATION_DECAY' : 0.995,
    'EXPLORATION_MIN' : 0.001,
    'SYNC_FREQ' : 50,
    'HIDDEN_LAYER_1_SIZE' : 512,
    'HIDDEN_LAYER_2_SIZE' : 2048
}

env = poolenv.PoolEnv(CFG)
agent = DQN.DDQN(env,CFG)

best_reward = 0
average_reward = 0
episode_number = []
average_reward_number = []
score_number = []
n_observation_state = gym.spaces.utils.flatdim(env.observation_space)

j=0
for i in tqdm(range(1, CFG['EPISODES'])):
    state = env.reset()
    score = 0
    state = np.reshape(np.concatenate([state['whiteBall'][:,None],state['nonWhiteBall']],axis = 1), [1, n_observation_state])
    
    while True:
        j+=1
        action = agent.choose_action(state)
        state_, reward, won,lost,terminated  = env.step(action)
        state_ = np.reshape(np.concatenate([state_['whiteBall'][:,None],state_['nonWhiteBall']],axis = 1), [1, n_observation_state])
        state = torch.tensor(state).float()
        state_ = torch.tensor(state_).float()

        exp = (state, action, reward, state_, won or lost )
        agent.replay.add(exp)
        agent.learn()

        state = state_
        score += reward

        if j % CFG['SYNC_FREQ'] == 0:
            agent.network2.load_state_dict(agent.network.state_dict())

        if won or lost or terminated:
            if score > best_reward:
                best_reward = score
            average_reward += score 
            episode_number.append(i)
            average_reward_number.append(average_reward/i)
            score_number.append(score)
            
            if i%10==0:
                print("Episode {} Average Reward {} Best Reward {} Last Reward {} Epsilon {}".format(i, average_reward/i, best_reward, score, agent.returning_epsilon()))
            break
  
        

recap_file = {'CFG' : CFG, 'ep_numb' : episode_number, 'score_numb' : score_number,
              'q' : agent.network,'q_targ' : agent.network2}

with open('DDQN_1BALL_V1', 'wb') as file:
    pickle.dump(recap_file, file)
    
plt.plot(episode_number, average_reward_number)
plt.show()