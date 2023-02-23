# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 19:02:52 2023

@author: Jolec
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from collections import deque
import copy
from tqdm.notebook import tqdm
import random
import gymnasium as gym 

class Network(torch.nn.Module):
    def __init__(self,env,CFG):
        super().__init__()
        self.CFG = CFG 
        self.input_shape = gym.spaces.utils.flatdim(env.observation_space)
        self.action_space = env.action_space.n
       

        self.fc1 = nn.Linear(self.input_shape, self.CFG['HIDDEN_LAYER_1_SIZE'])
        self.fc2 = nn.Linear( self.CFG['HIDDEN_LAYER_1_SIZE'],  self.CFG['HIDDEN_LAYER_2_SIZE'])
        self.fc3 = nn.Linear( self.CFG['HIDDEN_LAYER_2_SIZE'], self.action_space)

        self.optimizer = optim.Adam(self.parameters(), lr=self.CFG['LR'])
        self.loss = nn.MSELoss()
        #self.to(DEVICE)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
    
        return x
    
class ReplayBuffer:
    def __init__(self,CFG):
        self.memory = deque(maxlen= CFG['MEM_SIZE'])
        self.CFG = CFG
    def add(self, experience):
        self.memory.append(experience)
    
    def sample(self):
        minibatch = random.sample(self.memory, self.CFG['BATCH_SIZE'])

        state1_batch = torch.stack([s1 for (s1,a,r,s2,d) in minibatch])
        action_batch = torch.tensor([a for (s1,a,r,s2,d) in minibatch])
        reward_batch = torch.tensor([r for (s1,a,r,s2,d) in minibatch])
        state2_batch = torch.stack([s2 for (s1,a,r,s2,d) in minibatch])
        done_batch = torch.tensor([d for (s1,a,r,s2,d) in minibatch])

        return (state1_batch, action_batch, reward_batch, state2_batch, done_batch)
    
class DDQN:
    def __init__(self,env,CFG):
        self.CFG = CFG 
        self.env = env
        self.replay = ReplayBuffer(self.CFG)
        self.exploration_rate = self.CFG['EXPLORATION_MAX']
        self.network = Network(self.env,self.CFG)
        self.network2 = copy.deepcopy(self.network) #A
        self.network2.load_state_dict(self.network.state_dict())


    def choose_action(self, observation):
        if random.random() < self.exploration_rate:
            return self.env.action_space.sample()

        state = torch.tensor(observation).float().detach()
        state = state.unsqueeze(0)
        q_values = self.network(state)
        
        return torch.argmax(q_values).item()


    def learn(self):
        if len(self.replay.memory)< self.CFG['BATCH_SIZE']:
            return

        state1_batch, action_batch, reward_batch, state2_batch, done_batch = self.replay.sample()
        q_values = self.network(state1_batch).squeeze()
        

        with torch.no_grad():
            next_q_values = self.network2(state2_batch).squeeze()

        batch_indices = np.arange(self.CFG['BATCH_SIZE'], dtype=np.int64)

        predicted_value_of_now = q_values[batch_indices, action_batch]
        predicted_value_of_future = torch.max(next_q_values, dim=1)[0]

   
        q_target = reward_batch + self.CFG['GAMMA'] * predicted_value_of_future * (1-(done_batch).long())

     
        loss = self.network.loss(q_target, predicted_value_of_now)

        self.network.optimizer.zero_grad()
        loss.backward()
        self.network.optimizer.step()

        self.exploration_rate *= self.CFG['EXPLORATION_DECAY']
        self.exploration_rate = max(self.CFG['EXPLORATION_MIN'], self.exploration_rate)


    def returning_epsilon(self):
        return self.exploration_rate