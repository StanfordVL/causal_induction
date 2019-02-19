from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py.modder import TextureModder
import mujoco_py
import os
import numpy as np
import gym
from gym import error, spaces
from gym.utils import seeding
import copy
DEFAULT_SIZE = 500



class LightingEnv(gym.GoalEnv):
    def __init__(self, model_path, horizon, n_actions, n_substeps, gc=True):
        if model_path.startswith('/'):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), 'assets', model_path)
        if not os.path.exists(fullpath):
            raise IOError('File {} does not exist'.format(fullpath))

        model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(model, nsubsteps=n_substeps)
        self.viewer = None
        self._viewers = {}
        self.horizon = horizon
        self.gc = gc
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.seed()
        self._env_setup(initial_qpos=None)
        self.initial_state = copy.deepcopy(self.sim.get_state())

        self.goal = self._sample_goal()
        obs = self._get_obs()
        self.action_space = spaces.Discrete(5)#
#         self.action_space = spaces.Box(0., 1., shape=(n_actions,), dtype='float32')
#         self.observation_space = spaces.Dict(dict(
#             desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
#             achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
#             observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
#         ))
#         print(obs.shape)
#         print(self.goal.shape)
#         t = np.concatenate([obs, self.goal], 2)
        self.observation_space = spaces.Box(0, 1, shape=obs.shape, dtype='float32')

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    # Env methods
    # ----------------------------

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        done = False
        info = {
            'is_success': self._is_success(obs, self.goal),
        }
        done = self._is_success(obs, self.goal)
        reward = self.compute_reward(obs, info)
        self.steps += 1
        self.rew += reward
        if done or (self.steps >= self.horizon):
            print(self.rew)
            with open("/Users/surajnair/Documents/stanford/causal_induction/rew.txt", "a") as f:
                f.write(str(self.rew) + "\n")
            self.reset()
        return obs, reward, done, info

    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        self.rew= 0
        self.goal = self._sample_goal().copy()
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        obs = self._get_obs()
        self.steps = 0
        return obs

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def render(self, mode='human', width=DEFAULT_SIZE, height=DEFAULT_SIZE):
        self._render_callback()
        if mode == 'rgb_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self._get_viewer(mode).render()

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == 'rgb_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)
            self._viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer
    
    def compute_reward(self, obs, info=None):
#         print(obs.shape, goal.shape)
#         return -1 * np.sqrt(((obs[:,:,:3] - self.goal)**2).sum())
        if (obs[:,:,:3]==self.goal).all():
            return 0
        else:
            return -1

    # Extension methods
    # ----------------------------

    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.
        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
#         self.sim.set_state(self.initial_state)
#         self.sim.forward()
        id1 = self.sim.model.light_name2id("l1")
        id2 = self.sim.model.light_name2id("l2")
        id3 = self.sim.model.light_name2id("l3")
        id4 = self.sim.model.light_name2id("l4")
        id5 = self.sim.model.light_name2id("l5")
        self.sim.model.light_active[id1] = 0
        self.sim.model.light_active[id2] = 0
        self.sim.model.light_active[id3] = 0
        self.sim.model.light_active[id4] = 0
        self.sim.model.light_active[id5] = 0
        return True

    def _get_obs(self):
        """Returns the observation.
        """
        im = self.sim.render(width=64,height=64,camera_name="bottomview") / 255.0
        if self.gc:
            t = np.concatenate([im, self.goal], 2)
        else:
            t = im
        return t

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
#         act = int(np.argmax(action))
#         print(act)
        self.sim.model.light_active[action] = 1 - self.sim.model.light_active[action]

    def _is_success(self, obs, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        return (obs[:,:,:3]==desired_goal).all()

    def _sample_goal(self):
        """Samples a new goal and returns it.
        """ 
        id1 = self.sim.model.light_name2id("l1")
        id2 = self.sim.model.light_name2id("l2")
        id3 = self.sim.model.light_name2id("l3")
        id4 = self.sim.model.light_name2id("l4")
        id5 = self.sim.model.light_name2id("l5")
        if not self.gc:
            self.sim.model.light_active[id1] = 1
            self.sim.model.light_active[id2] = 1
            self.sim.model.light_active[id3] = 1
            self.sim.model.light_active[id4] = 1
            self.sim.model.light_active[id5] = 1
        else:
            self.sim.model.light_active[id1] = (np.random.uniform() < 0.5)*1
            self.sim.model.light_active[id2] = (np.random.uniform() < 0.5)*1
            self.sim.model.light_active[id3] = (np.random.uniform() < 0.5)*1
            self.sim.model.light_active[id4] = (np.random.uniform() < 0.5)*1
            self.sim.model.light_active[id5] = (np.random.uniform() < 0.5)*1
        return self.sim.render(width=64,height=64,camera_name="bottomview") / 255.0

    def _env_setup(self, initial_qpos):
        """Initial configuration of the environment. Can be used to configure initial state
        and extract information from the simulation.
        """
        pass

    def _viewer_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position,
        for example.
        """
        pass

    def _render_callback(self):
        """A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass
