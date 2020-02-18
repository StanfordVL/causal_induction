from .light_env import LightEnv

import os
import numpy as np
import gym
from gym import error, spaces
from gym.utils import seeding
import copy
DEFAULT_SIZE = 500
import time
from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy, MlpPolicy, MlpLstmPolicy
from stable_baselines.deepq.policies import CnnPolicy as cnnQ
from stable_baselines.deepq.policies import MlpPolicy as mlpQ
from stable_baselines.deepq.policies import MixedPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C, ACER, PPO2, TRPO, DQN

import torch as th
from .models.models import ForwardModel, SupervisedModel, SupervisedModelCNN


class MetaEnv(gym.GoalEnv):
    def __init__(self, horizon=5, num=5, method="one_policy", structure="one_to_one", gc=True, random=0):
        
        self.horizon = horizon
        self.gc = gc
        self.num = num
        self.method = method
        self.structure = structure
        
        if method == "one_policy":
            self.env = LightEnv(horizon, num, None, structure, gc, filename=str(self.gc)+"random"+str(random)+"_"+method)
        elif method == "two_policy":
            self.env = LightEnv(horizon, num, (horizon*(2*num+1)), structure, gc, filename=str(self.gc)+"random"+str(random)+"_"+method)
            self.e = DummyVecEnv([lambda: self.env])
            self.policy = PPO2(MlpLstmPolicy, self.e, verbose=1, vf_coef=0.0001, noptepochs=4, nminibatches=1, n_steps=128)
        elif method == "gt":
            self.env = LightEnv(horizon, num, "gt", structure, gc, filename=str(self.gc)+"gt"+str(random)+"_"+method)  
        elif method == "inference":
            self.env = LightEnv(horizon, num, None, structure, gc, filename=str(self.gc)+"random"+str(random)+"_"+method)
        
        self.metadata = self.env.metadata 
#         if method == "gt":
#             print(len(self.env.traj))
#             self.action_space = spaces.Box(-np.inf,np.inf, shape = (self.num+1+len(self.env.traj),))
#         else:
        self.action_space = self.env.action_space
        
        o = self.reset()
        self.observation_space = spaces.Box(0, 1, shape=o.shape, dtype='float32')
        
        self.e = DummyVecEnv([lambda: self.env])

        self.action_log = []
        
        self.steps = 0
        self.rew = 0
        self.eps = 0
        
    def evaluate(self):
        traj = self.epbuf.flatten()
        self.env.state = np.zeros((self.num))
        self.env.traj = traj
        self.policy.learn(total_timesteps=5)
        rews = []
        for i in range(5):
            obs=self.e.reset()
            done = False
            rew = 0
            last = None
            while not done:
                a, last = self.policy.predict(obs, last)
                obs, reward, done, info = self.e.step(a)
                rew += reward
            rews.append(rew)
        return np.mean(rews)
    
    
    def step(self, action):
        if self.method == "two_policy":
            p = self.env._get_obs()
            p = p.reshape((1, -1))
            a = np.zeros((1, self.num+1))
            a[:,action] = 1

            mem = np.concatenate([p[:,:self.num], a], 1)
            if self.steps == 0:
                self.epbuf = mem
            else: 
                self.epbuf = np.concatenate([self.epbuf, mem], 0)
            
            obs, reward, done, info = self.env.step(action, count=False)
            self.steps += 1
            self.action_log.append(action)
            if self.steps >= self.horizon:
                reward = self.evaluate()
                done = True
                self.eps += 1
                self.rew += reward
            else:
                reward = 0
        elif self.method == "gt":
            self.cs += action[self.num+1:]
            print(action)
#             self.cs = self.cs.clip(0,1)
            act= action[:self.num+1]
            act = np.exp(act) / (np.sum(np.exp(act))+.0001) 
            self.action_log.append(act.argmax())
            obs, reward, done, info = self.env.step(act.argmax(), count=True)
            reward = -((self.env.traj - self.cs)**2).mean()
            self.rew += reward
            print(self.env.traj, self.cs, reward)
#             print(reward)
#             assert(False)
            
        elif self.method == "one_policy":
            if self.steps < self.horizon:
                induction = True
                obs, reward, done, info = self.env.step(action, count=False)
                reward = 0
            elif self.steps == self.horizon:
                self.env.state = np.zeros((self.num))
                induction = False
                obs, reward, done, info = self.env.step(action, count=True)
            else:
                induction = False
                obs, reward, done, info = self.env.step(action, count=True)
            
            if done:
                self.eps += 1
                
            self.rew += reward
            self.steps += 1
            self.action_log.append(action)
        elif self.method == "inference":
            obs, reward, done, info = self.env.step(action, count=True)
            self.eps += 1
            self.rew += reward
            self.steps += 1
            self.action_log.append(action)
        
        if self.method == "inference":
            pass
        elif self.method == "one_policy":
            if induction:
                obs[self.num:] = 0
        else:
            obs = obs[:self.num]
            
#         print(action)
#         print(obs, reward, done, info)
#         if done:
#             assert(False)
        
        if self.method == "gt":
            self.cs = np.zeros(self.env.traj.shape)
            obs = np.concatenate([obs,self.cs])
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset(keep_struct=False)
        if self.method == "inference":
            pass
        elif self.method == "one_policy":
            obs[self.num:] = 0
        else:
            obs = obs[:self.num]
        
        if self.method == "gt":
            self.cs = np.zeros(self.env.traj.shape)
            obs = np.concatenate([obs,self.cs])
    
        self.action_log = []
        self.rew = 0
        self.steps = 0
        return obs