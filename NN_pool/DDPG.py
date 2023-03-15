# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 16:15:26 2023

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

class criticQNet(torch.nn.Module):
    def __init__(self,env,CFG):
        '''
        input shape : observation_space size + action_space size i.e we have a netwoork that take
        (a,s) as input and output a scalar Q(a,s)
        '''
        super().__init__()
        self.CFG = CFG 
        self.env = env
        
        self.input_shape = gym.spaces.utils.flatdim(env.observation_space) + env.game.array_holes.flatten().shape[0]
        self.action_shape = gym.spaces.utils.flatdim(env.action_space)
        
        self.fc_states_holes = nn.Linear(self.input_shape, self.CFG['HIDDEN_LAYER_1_Q_SIZE'])
        self.fc_action = nn.Linear(self.action_shape,self.CFG['HIDDEN_LAYER_1_Q_SIZE'])
        self.fc2 = nn.Linear(2*self.CFG['HIDDEN_LAYER_1_Q_SIZE'],  self.CFG['HIDDEN_LAYER_2_Q_SIZE'])
        self.fc3 = nn.Linear(self.CFG['HIDDEN_LAYER_2_Q_SIZE'],self.CFG['HIDDEN_LAYER_3_Q_SIZE'])
        self.fc4 = nn.Linear( self.CFG['HIDDEN_LAYER_3_Q_SIZE'], 1)

        
    def forward(self, state,action):
        
        array_holes = torch.unsqueeze(torch.Tensor(self.env._get_holes().flatten()),dim = 0).expand(self.CFG['BATCH_SIZE'],-1)
        
        res_states_holes = torch.cat([state.squeeze(),array_holes],dim = -1)
        res_states_holes =  F.relu(self.fc_states_holes(res_states_holes))
    
        res_action = F.relu(self.fc_action(action))
        
        res = F.relu(self.fc2(torch.cat([res_states_holes,res_action],dim = -1)))
        res = F.relu(self.fc3(res))
        res = self.fc4(res)
    
        return torch.squeeze(res,-1)
    
class actorMuNet(torch.nn.Module):
    def __init__(self,env,CFG):
        
        super().__init__()
        self.CFG = CFG 
        self.env  = env
        
        self.input_shape = gym.spaces.utils.flatdim(env.observation_space) + env.game.array_holes.flatten().shape[0]
        self.action_space = gym.spaces.utils.flatdim(env.action_space)
       
        self.fc1 = nn.Linear(self.input_shape, self.CFG['HIDDEN_LAYER_1_MU_SIZE'])
        self.fc2 = nn.Linear(self.CFG['HIDDEN_LAYER_1_MU_SIZE'],  self.CFG['HIDDEN_LAYER_2_MU_SIZE'])
        self.fc3 = nn.Linear(self.CFG['HIDDEN_LAYER_2_MU_SIZE'], self.action_space)
    
    def forward(self, state):
        
        array_holes = torch.unsqueeze(torch.Tensor(self.env._get_holes().flatten()),dim = 0)
        
        res  =  torch.cat([state.squeeze(),array_holes.expand(state.shape[0],-1).squeeze()],dim = -1)
        res = F.relu(self.fc1(res))
        res = F.relu(self.fc2(res))
        res = F.sigmoid(self.fc3(res))
    
        return res.squeeze()
    
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
    
class DDPG:
    def __init__(self,env,CFG):
        
        self.CFG = CFG 
        self.env = env
        self.replay = ReplayBuffer(self.CFG)
       
        #Setting up critic Q and actor Mu networks'
        self.q = criticQNet(self.env,self.CFG)
        self.mu = actorMuNet(self.env,self.CFG)
        
        #deepcopy of both q and mu networks to initialize target networks'
        self.q_target = copy.deepcopy(self.q) 
        self.mu_target = copy.deepcopy(self.mu)
   
        #parameters in targetet networks only get uptaded via updates from q and mu networks, so we freeze them'
        for param in self.q_target.parameters(): param.requires_grad = False
        for param in self.mu_target.parameters() : param.requires_grad = False 
        
        #Setting up the optimizers for both non-targetet networks'
        self.optimizer_q = optim.Adam(self.q.parameters(), lr= self.CFG['LR_Q'])
        self.optimizer_mu = optim.Adam(self.mu.parameters(), lr= self.CFG['LR_MU'])
        
    def q_loss(self,state1_batch, action_batch, reward_batch, state2_batch, done_batch):
        '''
        Compute loss for the Q network, using transitions stocked in the replay buffer'
        '''
        
        state1_batch, action_batch, reward_batch, state2_batch, done_batch = self.replay.sample()
        
        q_values = self.q(state1_batch.float(),action_batch.float()).squeeze()
        
        with torch.no_grad():
            q_values_target = self.q_target(state2_batch,self.mu_target(state2_batch)).squeeze()
            y = reward_batch + self.CFG['GAMMA'] * q_values_target * (~done_batch).int()
        tmp = nn.MSELoss()
        return tmp(q_values, y)
    
    def mu_loss(self,state):
        '''
        Compute loss for the mu network
        '''
        
        return -(self.q(state,self.mu(state)).mean())
    
    
    def choose_action(self, state, gaussian_noise_level):
        '''
        Choosing wich action to play using the mu network, and gaussian noise to simulate exploration'
        '''
        
        state = torch.tensor(state).float().detach()
        state = state.unsqueeze(0)
        action  = np.clip(self.mu(state).detach().numpy() + 
                np.squeeze(gaussian_noise_level*np.random.randn(gym.spaces.utils.flatdim(self.env.action_space),1))
                ,0,1)
        
        return action


    def learn(self):
        if len(self.replay.memory)< self.CFG['BATCH_SIZE']:
            return

        state1_batch, action_batch, reward_batch, state2_batch, done_batch = self.replay.sample()


        #Gradient descent for q, mu 
        
        self.optimizer_q.zero_grad()
        q_loss = self.q_loss(state1_batch, action_batch, reward_batch, state2_batch, done_batch)
        q_loss.backward()
        self.optimizer_q.step()

        self.optimizer_mu.zero_grad()
        mu_loss = self.mu_loss(state1_batch)
        mu_loss.backward()
        self.optimizer_mu.step()
        
        

        #updates of the target networks 
        
        with torch.no_grad():
            
            for param,param_target in zip(self.q.parameters(),self.q_target.parameters()):
                param_target.data.mul_(1 - self.CFG['TAU'])
                param_target.data.add_(self.CFG['TAU']*param.data)
                
            for param,param_target in zip(self.mu.parameters(),self.mu_target.parameters()):
                param_target.data.mul_(1 - self.CFG['TAU'])
                param_target.data.add_(self.CFG['TAU']*param.data)
            
            
            
            
            
            
            
