from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py.modder import TextureModder
import mujoco_py
import os
import numpy as np
import gym
from gym import error, spaces
from gym.utils import seeding
import copy
from itertools import permutations
import cv2
DEFAULT_SIZE = 500

class LightEnv(gym.GoalEnv):
    '''Light Switch Environment for Visual Causal Induction'''
    def __init__(self, horizon=5, num=5, cond="gt", structure="one_to_one", gc=True, filename=None, seen=10):
        '''
        Creates Light Switch Environment
        
        Args:
          horizon: Length of episode
          num: Number of switches [5,7,9]
          cond: Whether use default size graph "GT" or custom size (Size).
          structure: Type of causal structure [one_to_one, one_to_many, many_to_one, masterswitch]
          gc: True/False goal conditioned or now
          filename: Path to log episode results
          seen: Number of seen causal structures
        '''
        
        ## Load XML model
        fullpath = os.path.join(os.path.dirname(__file__), 'assets', "arena_v2_"+str(num)+".xml")
        if not os.path.exists(fullpath):
            raise IOError('File {} does not exist'.format(fullpath))
        model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(model)
        
        self.filename = filename
        self.horizon = horizon
        self.cond = cond
        self.gc = gc
        self.structure = structure
        self.num = num
        self.metadata = {'render.modes': ['human', 'rgb_array']}
        
        ## Initialize GT/traj - the underlying causal structure
        if self.cond is None:
            pass
        elif self.cond == "gt":
            if self.structure == "masterswitch":
                a = self.num**2 + self.num
            else:
                a = self.num**2
            self.gt = np.zeros((a))
        else:
            self.traj = np.zeros(self.cond)
        self.aj = np.zeros((self.num, self.num))
        
        ## If goal conditioned, sample goal state
        if self.gc:
            self.goal = self._sample_goal()
        
        ## Set random seed so order of causal structures is preserved
        np.random.seed(1)
        
        
        self.state = np.zeros((self.num)) # Initial State
        self.eps = 0 # Num Episodes
        
        # Generate causal structure
        if (self.structure == "one_to_many") or (self.structure == "many_to_one"):
            if self.num == 9:
                self.all_perms = self.generate_cs_set1(self.num, True)
            else:
                self.all_perms = self.generate_cs_set1(self.num)
        else:
            self.all_perms = self.generate_cs_set2(self.num)
            
        ## Params to randomize strcut and train v test     
        self.keep_struct = True
        self.train = True
        
        ## Shuffled causal structures
        np.random.shuffle(self.all_perms)
        ## Number of all structs
        self.pmsz = self.all_perms.shape[0]
        self.seen = seen
        
        obs = self._get_obs()
        self.action_space = spaces.Discrete(self.num+1)
        self.observation_space = spaces.Box(0, 1, shape=obs.shape, dtype='float32')


    # Env methods
    # ----------------------------

    def step(self, action, count = True, log=False):
        '''Step in env. 
        Args:
          action: which switch to toggle
          count: False if not counted toward episode (for collecting heuristic data)
          log: Log episode results
        '''
        ## If "Do Nothing" Action
        if action == self.num:
            pass
        else:
            if self.structure == "masterswitch":
                ## Only once masterswitch is activated can others be activated
                if (action == self.ms) or (self.state[self.ms] == 1):
                    change = np.zeros(self.num)
                    change[action] = 1
                    self.state = np.abs(self.state - change)
            else:
                change = np.zeros(self.num)
                change[action] = 1
                self.state = np.abs(self.state - change)

        obs = self._get_obs()

        done = False
        info = {'is_success': self._is_success(obs)}
        self.correct.append((info["is_success"]))
        reward = self.compute_reward(obs, info)
        
        if count:
            self.steps += 1
            self.eprew += reward
            if reward == 0:
                done = True
            if (self.steps >= self.horizon):
                done = True
            if done and log:
                with open(self.filename + \
                          "_S" + str(self.seen) + "_"+str(self.structure)+ \
                          "_H"+str(self.horizon)+"_N"+str(self.num)+ \
                          "_T"+str(self.current_cs)+".txt", "a") as f:
                    f.write(str(self.eprew) + "\n")
                with open(self.filename + "_S" + str(self.seen) + \
                          "_"+str(self.structure)+"_H"+str(self.horizon)+ \
                          "_N"+str(self.num)+"_T"+str(self.current_cs)+\
                          "successrate.txt", "a") as f:
                    f.write(str(int(info["is_success"])) + "\n")
            
        return obs, reward, done, info

    def reset(self):
        keep_struct = self.keep_struct
        train = self.train
        if train:
            self.current_cs = "train"
        else:
            self.current_cs = "test"
        
        ## Either reset causal structure or not
        if keep_struct:
            pass
        else:
            ## Select from seen causal structre or unseen causal structure
            if train:
                ind = np.random.randint(0, self.seen)
            else:
                ind = np.random.randint(self.seen, self.pmsz)
            perm = self.all_perms[ind]
            
            ## Set graph according to causal structure
            if self.structure == "one_to_one":
                aj = np.zeros((self.num,self.num))
                for i in range(self.num):
                    aj[i, perm[i]] = 1
                self.aj = aj
                self.gt = self.aj.flatten()
            elif self.structure == "one_to_many":
                aj = np.zeros((self.num,self.num))
                for i in range(self.num):
                    aj[i, perm[i]] = 1
                self.aj = aj.T
                self.gt = self.aj.flatten()
            elif self.structure == "many_to_one":
                aj = np.zeros((self.num,self.num))
                for i in range(self.num):
                    aj[i, perm[i]] = 1
                self.aj = aj
                self.gt = self.aj.flatten()
            elif self.structure == "masterswitch":
                aj = np.zeros((self.num,self.num))
                for i in range(self.num):
                    aj[i, perm[i]] = 1
                self.aj = aj
                self.ms = np.random.randint(self.num)
                m = np.zeros((self.num))
                m[self.ms] = 1
                self.gt = self.aj.flatten()
                self.gt = np.concatenate([self.gt, m])
        self.eprew = 0
        self.steps = 0
        self.correct = []
        self.state = np.zeros((self.num))
        self.eps += 1
        self.goal = self._sample_goal()

        obs = self._get_obs()
        return obs

    def compute_reward(self, obs, info=None):
        ## Distance to goal configuration
        rew =-1 * np.sqrt(((obs[:self.num] - self.goal)**2).sum())
        return rew

    def _get_obs(self, images=False):
        """Returns the observation.
        """
        ## Compute which lights are on based on underlying state and causal graph
        light = np.dot(self.state.T, self.aj)
        light = light % 2
        o = light
        
        ## Set corresponding lights and render image
        if images:
            self.sim.model.light_active[:] = light
            im = self.sim.render(width=32,height=32,camera_name="birdview") / 255.0
            return im

        ## Concatenate goal
        if self.gc:    
            o = np.concatenate([o, self.goal])
        
        ## Concatenate graph
        if self.cond is None:
            pass
        elif self.cond == "gt":
            o = np.concatenate([o, self.gt])
        else:
            o = np.concatenate([o, self.traj])

        return o

    def _is_success(self, obs):
        """Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        return (obs[:self.num] == self.goal).all()

    def _sample_goal(self):
        """Samples a new goal and returns it.
        """ 
        state = np.random.randint(0, 2, size=(self.num))
        light = np.dot(state.T, self.aj)
        light = light % 2
        self.sim.model.light_active[:] = light
        self.goalim = self.sim.render(mode='offscreen', width=32,height=32,camera_name="birdview") / 255.0 
        return light
    
    def generate_cs_set1(self, sz, cut=False):
        '''Generate Causal Structures for Many to One. For every light, 
        every possible combination of switches which could control it.
        
          Args:
            sz: num switches
            cut: Reduce generated structures for efficiency
        '''
        if sz == 1:
            lp = []
            for i in range(self.num):
                lp.append([i])
            return np.array(lp)
        else:
            gs = []
            tm = self.generate_cs_set1(sz-1, cut)
            for t in tm:
                if cut and (np.random.uniform() < 0.4):
                    continue
                for i in range(self.num):
                    gs.append(np.concatenate([np.array([i]), t]))
            return np.array(gs)
        
    def generate_cs_set2(self, sz):
        '''Generate Causal Structures for One to One. All Permutations.
        
          Args:
            sz: num switches
        '''
        t = np.arange(self.num)
        gs = []
        for perm in permutations(t):
            gs.append(np.array(list(perm)))
        return np.array(gs)