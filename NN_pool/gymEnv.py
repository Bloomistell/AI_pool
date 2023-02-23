# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 12:51:05 2023

@author: Jolec
"""

import gym
from gym import spaces
import pygame
import numpy as np
import config
import gamestate
import collisions 
import cue 

class PoolEnv(gym.Env):
    

    def __init__(self, num_balls = 2, render_mode=None, size=5):
        self.upperCoordBall_x = config.resolution[0]
        self.upperCoordBall_y  = config.resolution[1]
        self.maxForce = config.cue_max_displacement
        self.n_balls = num_balls
        self.alphaReward = 1               #Reward factor if the agent did not touch any ball
        self.betaReward = 1                #Reward factor if the agent potted one or more ball 
        self.game = gamestate.GameState()
        
        #Observation space : we have n_balls, including one white balls,
        #i.e n_balls*2 different continuous coordinates as observation space.
        self.observation_space = spaces.Dict(
            {
                "whiteBall" : spaces.Box([0,0], [self.upperCoordBall_x, self.upperCoordBall_x], shape=(2,1), dtype= float),
                "nonWhiteBalls" : spaces.Box([0,0], [self.upperCoordBall_x, self.upperCoordBall_y], shape=(2,self.n_balls -1 ), dtype = float )
            }
        )

        #Action space : angle and force of the shot (two continuous space in [0,1]) 
        self.action_space = spaces.Dict(
            {
                "angle" : spaces.box(0,360,shape = (1,), dtype = float), 
                "force" : spaces.Box(0,self.maxForce,shape = (1,),dtype = float)
            }
        )
    def _get_obs(self):
        for ball in self.game.balls:   
            if ball.number == 0 :
                self._whiteBallLoc = ball.pos
            else: 
                self._nonwhiteBallLoc[:,ball.number] = ball.pos 
                
        return {"whiteBall": self._whiteBallLoc, "nonWhiteBall" : self._nonWhiteBallLoc}
    
    def oneBallRandomStart(self):
        '''
        Randomize all ball positions, at the start of a game.
        '''
        for ball in self.game.balls:
            ball.pos = np.array([np.random.rand()*(self.upperCoordBall_x - 2*config.ball_radius) - 2*config.ball_radius],
                                    [np.random.rand()*(self.upperCoordBall_y - 2*config.ball_radius) - 2*config.ball_radius])
        
    def reset(self, randomBallPlacement = True,seed=None, options=None):
        
        super().reset(seed=seed)
        self.game = gamestate.GameState()
        if randomBallPlacement :
            self.oneBallRandomStart()
        observation = self._get_obs(self.game)
        
        return observation
    
    def step(self, action):
        
        previousObservation = self._get_obs()
        
        self.game.next_turn_GymEnv(self,action)
        collisions.resolve_all_collisions(self.game.balls,self.game.holes, self.game.table_sides)
        self.game.update_balls()
        
        if self.game.balls_not_moving():
             self.game.check_pool_rules()
        
        observation = self._get_obs()
        reward =  (- self.alphaReward*(np.all(observation['nonWhiteBall'] == previousObservation['nonWhiteBall'])) + 
                   self.betaReward*(len(self.game.potted)))
        terminated = self.game.is_game_over
         
        return observation, reward, terminated 
    
    
    
    