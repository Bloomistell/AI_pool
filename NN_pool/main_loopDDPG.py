# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 14:03:47 2023

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
import DDPG


CFG = {
    'CONTINUOUS_ACTION_SPACE' : True,
    'ALPHA'        : -1,
    'BETA'         : 0,
    'DELTA'        : 0, 
    'NUM_BALLS'    : 2,
    'MAX_TURNS' : 1,
    'EPISODES' : 50000,
    'LR_Q' : 0.0001,
    'LR_MU': 0.001,
    'HIDDEN_LAYER_1_MU_SIZE': 128,
    'HIDDEN_LAYER_2_MU_SIZE' : 256,
    'HIDDEN_LAYER_1_Q_SIZE' : 64,
    'HIDDEN_LAYER_2_Q_SIZE' : 256,
    'HIDDEN_LAYER_3_Q_SIZE' : 64,
    'PURE_EXPLORATION_STEPS' : 10000,
    'GAUSSIAN_NOISE_EXPLORATION_STEPS' : 10000,
    'MEM_SIZE' : 1000000,
    'BATCH_SIZE' : 64,
    'GAMMA' : 0.99,
    'TAU'   : 0.001,
    'GAUSSIAN_NOISE_LEVEL_MAX' : 0.2,
    'GAUSSIAN_NOISE_LEVEL_DECAY' : 0.9993,
    'GAUSSIAN_NOISE_LEVEL_MIN' : 0.000
}
print(CFG['HIDDEN_LAYER_1_MU_SIZE'])
env = poolenv.PoolEnv(CFG)
agent = DDPG.DDPG(env,CFG)
gaussian_noise_level = CFG['GAUSSIAN_NOISE_LEVEL_MAX']
best_reward = 0
average_reward = 0
average_reward_last_50 = 0 
episode_number = []
average_reward_number = []
score_number= []
n_observation_state = gym.spaces.utils.flatdim(env.observation_space)

j=0
for i in tqdm(range(1, CFG['EPISODES'])):
    state = env.reset()
    score = 0
    state = np.reshape(np.concatenate([state['whiteBall'][:,None],state['nonWhiteBall']],axis = 1), [1, n_observation_state])
    
    while True:
        j+=1
        
        if(j <= CFG['PURE_EXPLORATION_STEPS']):
            action = env.action_space.sample()
            action = np.array([action['angle'][0],action['force'][0]])
        elif(j <= CFG['PURE_EXPLORATION_STEPS'] + CFG['GAUSSIAN_NOISE_EXPLORATION_STEPS']):
            action = agent.choose_action(state,gaussian_noise_level)
            gaussian_noise_level *= CFG['GAUSSIAN_NOISE_LEVEL_DECAY']
            gaussian_noise_level = max(CFG['GAUSSIAN_NOISE_LEVEL_MIN'],gaussian_noise_level)
        else:
            action = agent.choose_action(state,0)
            
        state_, reward, won,lost,terminated  = env.step(action)
        state_ = np.reshape(np.concatenate([state_['whiteBall'][:,None],state_['nonWhiteBall']],axis = 1), [1, n_observation_state])
        state = torch.tensor(state).float()
        state_ = torch.tensor(state_).float()

        exp = (state, action, reward, state_, won or lost)
        agent.replay.add(exp)
        agent.learn()

        state = state_
        score += reward


        if won or lost or terminated:
            if score > best_reward:
                best_reward = score
            average_reward += score 
            average_reward_last_50 += score
            
            episode_number.append(i)
            average_reward_number.append(average_reward/i)
            score_number.append(score)
            
            if i%50==0:
                print("Episode {} Average Reward {}  Average reward (Last 50 episodes) {} Best Reward {} Last Reward {} Epsilon {}".format(i, average_reward/i, average_reward_last_50/50, best_reward, score, gaussian_noise_level))
                average_reward_last_50 = 0 
                #test_model(agent,10, observation_space)
            break
  
    
        
recap_file = {'CFG' : CFG, 'ep_numb' : episode_number, 'score_numb' : score_number,
              'q' : agent.q, 'mu' : agent.mu,
              'q_target' : agent.q_target, 'mu_target' : agent.mu_target}

with open('DDPG_1BALL_V1', 'wb') as file:
    pickle.dump(recap_file, file)

mean_50 = np.convolve(average_reward_number, np.ones(50)/50, mode='valid')
var_50 = np.convolve((average_reward_number - mean_50)**2,np.ones(50)/50,mode = 'valid')
#plt.plot(np.convolve(average_reward_number, np.ones(50)/50, mode='valid'))
plt.plot(episode_number[98:],mean_50[49:])
plt.fill_between(episode_number[98:],mean_50 + var_50,mean_50 - var_50)
plt.show()