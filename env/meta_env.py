from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py.modder import TextureModder
from .lighting_env import LightingEnv
import mujoco_py
import os
import numpy as np
import gym
from gym import error, spaces
from gym.utils import seeding
import copy
DEFAULT_SIZE = 500



class MetaEnv(gym.GoalEnv):
    def __init__(self, model_path, horizon, n_actions, n_substeps, gc=True, induction=True):
        
        if induction:
            self.env = LightingEnv("arena.xml" ,2 * horizon, n_actions, n_substeps, gc=gc)
        else:
            self.env = LightingEnv("arena.xml" ,horizon, n_actions, n_substeps, gc=gc)
        self.induction = induction

        self.horizon = horizon
        self.gc = gc
        self.metadata = self.env.metadata 
        self.initial_state = self.env.initial_state
        
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.steps = 0

    @property
    def dt(self):
        return self.env.dt
    
    def step(self, action):
        if induction and (steps < self.horizon):
            obs, reward, done, info = self.env.step(np.random.randint(n_actions))
            obs[:,:,3:] = 0
            reward = 0
        else:
            obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def reset(self):
        #self.env = LightingEnv("arena.xml" ,horizon, n_actions, n_substeps, gc=gc)
        self.env.sim.data.light_xpos[:, :2] = np.random.uniform(-3,3, size=(5,2))
        success = self.env.reset()
#         print(self.env.sim.model.light_pos)
#         print(self.env.sim.data.light_xpos)
#         print(self.env.lighting_goal)
#         g = self.env.goal
#         import cv2
#         cv2.imwrite("goal.png", g*255)
#         assert(False)
        self.steps = 0
        return success

    def close(self):
        return self.env.close()

    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        return self.env.render(mode, width, height)
        
    def _get_viewer(self, mode):
        return self.env._get_viewer(mode)
    
    def compute_reward(self, obs, info=None):
        return self.env.compute_reward(obs)