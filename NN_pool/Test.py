# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 16:00:15 2023

@author: Jolec
"""

import sys

import collisions
import event
import gamestate
import numpy as np
import gymnasium as gym

import poolenv

CFG = {
    'N_BINS_ANGLE' : 180,
    'N_BINS_FORCE' : 16,
    'ALPHA'        : 2,
    'BETA'         : 5,
    'NUM_BALLS' : 2 ,
    'MAX_TURNS' : 10 ,
    'EPISODES' : 150,
    'LR' : 0.0001,
    'MEM_SIZE' : 10000,
    'BATCH_SIZE' : 64,
    'GAMMA' : 0.95,
    'EXPLORATION_MAX' : 1.0,
    'EXPLORATION_DECAY' : 0.999,
    'EXPLORATION_MIN' : 0.001,
    'sync_freq' : 10,
    'HIDDEN_LAYER_1_SIZE' : 512,
    'HIDDEN_LAYER_2_SIZE' : 2048
}

env = poolenv.PoolEnv(CFG)
env.reset()

print('Observation space : ', gym.spaces.utils.flatdim(env.observation_space))
print('Action space : ', env.action_space['angle'].n)
print(env._get_obs())
a = {'angle' : 180,'force' : 30}
observation, reward, terminated = env.step(a)
a = {'angle' : 180,'force' : 40}
observation, reward, terminated = env.step(a)
a = {'angle' : 180,'force' : 50}
observation, reward, terminated = env.step(a)
a = {'angle' : 180,'force' : 60}
observation, reward, terminated = env.step(a)


