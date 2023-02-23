# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 12:51:05 2023

@author: Jolec
"""

import gymnasium as gym 
import pygame
import numpy as np
import config
import gamestate
import collisions 
import cue 
import copy 

class PoolEnv(gym.Env):
    

    def __init__(self, CFG, render_mode=None, size=5):
        self.upperCoordBall_x = config.resolution[0]
        self.upperCoordBall_y  = config.resolution[1]
        self.maxForce = config.cue_max_displacement
        self.n_balls = CFG['NUM_BALLS']
        self.alphaReward = CFG['ALPHA']            #Reward factor if the agent did not touch any ball
        self.betaReward = CFG['BETA']                #Reward factor if the agent potted one or more ball 
        self.deltaReward = CFG['DELTA']             #Reward factor if the agent lost (potted white ball)
        
        if(not CFG['CONTINUOUS_ACTION_SPACE']):
            self.n_bins_angle = CFG['N_BINS_ANGLE']
            self.n_bins_force = CFG['N_BINS_FORCE']
            
        self.CFG = CFG 
        
        self.game = gamestate.GameState(self.n_balls,poolEnv = True,verbose = False)
        
        self._whiteBallLoc = np.zeros((2,1),dtype = float )
        self._nonWhiteBallLoc = np.zeros((2,self.n_balls - 1),dtype = float)
        self.ballNumberCode = np.ones(config.max_ball_num,dtype = int)*100
        k = 0 
        for ball in self.game.balls:
            if(ball.number != 0):
                self.ballNumberCode[ball.number] = k 
                k+=1
                
        #Observation space : we have n_balls, including one white balls,
        #i.e n_balls*2 different continuous coordinates as observation space.
        self.observation_space = gym.spaces.Dict(
            {
                "whiteBall" : gym.spaces.Box(np.zeros((2,1)), np.array([[self.upperCoordBall_x], [self.upperCoordBall_x]]), shape=(2,1), dtype= float),
                "nonWhiteBalls" : gym.spaces.Box(np.zeros((2,self.n_balls -1)), 
                                  np.array([self.upperCoordBall_x*np.ones(self.n_balls -1), self.upperCoordBall_y*np.ones(self.n_balls -1)]), 
                                            shape=(2,self.n_balls -1 ), dtype = float )
            }
        )
        
        if(self.CFG['CONTINUOUS_ACTION_SPACE']):
            #Action space : angle and force of the shot (two continuous space in [0,1]) 
            self.action_space = gym.spaces.Dict(
                {
                    "angle" :gym.spaces.Box(0,1,shape = (1,), dtype = float), 
                    "force" : gym.spaces.Box(0,1,shape = (1,),dtype = float)
                }
            )
        else:
            #Action space : angle and force of the shot, combined into one discrete space of size n_bins_angle*n_bins_force 
            self.action_space = gym.spaces.Discrete(self.n_bins_angle*self.n_bins_force)
              
             
    def _get_obs(self):
        for ball in self.game.balls:   
            if ball.number == 0 :
                self._whiteBallLoc = ball.pos
                self._whiteBallLoc /= config.resolution
            else: 
                self._nonWhiteBallLoc[:,self.ballNumberCode[ball.number]] = ball.pos
                self._nonWhiteBallLoc /= config.resolution[:,None]
        return {"whiteBall": self._whiteBallLoc, "nonWhiteBall" : self._nonWhiteBallLoc}
    
    def get_coord_from_obs(self):
        tmp = self._get_obs()
        return np.array([tmp['whiteBall'],tmp['nonWhiteBall']])
    
    def oneBallRandomStart(self):
        '''
        Randomize all ball positions, at the start of a game.
        '''
        for ball in self.game.balls:
            ball.pos = np.array([np.random.rand()*(self.upperCoordBall_x - 4*config.ball_radius) +  2*config.ball_radius,
                                    np.random.rand()*(self.upperCoordBall_y - 4*config.ball_radius) + 2*config.ball_radius])
             

    def envAction_to_poolAction(self,action):
        
        angle = (action % self.n_bins_angle)*(360/self.n_bins_angle)
        force = 20 + int(action / self.n_bins_angle)*((80/(self.n_bins_force-1)))
        
        return angle, force
    
    def reset(self, randomBallPlacement = True,seed=None, options=None):
        
        super().reset(seed=seed)
        self.game = gamestate.GameState(self.n_balls,poolEnv = True,verbose = False)
        if randomBallPlacement :
            self.oneBallRandomStart()
        observation = self._get_obs()
        
        return observation
    
    def step(self, action):
        
        previousObservation_nw = copy.deepcopy(self._get_obs()['nonWhiteBall'])
        #angle,force = self.envAction_to_poolAction(action)
        #print(previousObservation_nw)
        self.game.next_turn_poolEnv(action[0],action[1])
        
        while not self.game.balls_not_moving():
            collisions.resolve_all_collisions(self.game.balls,self.game.holes, self.game.table_sides)
            self.game.update_balls()
        self.game.check_pool_rules()
        
        observation = self._get_obs()
        
        if(self.game.is_game_over):
            if(not self.game.won) : 
                lost = True
                won = False
                reward = - self.deltaReward
            else:
                lost = False
                won = True 
                reward = self.betaReward
        else: 
            reward =  - self.alphaReward*(not (np.all(np.abs(observation['nonWhiteBall'] -previousObservation_nw)< 0.001)))
            won = False
            lost = False 
            
        terminated = self.game.turn_number >= self.CFG['MAX_TURNS']
        
        
       
        
        return observation, reward, won,lost,terminated 
    

    
    