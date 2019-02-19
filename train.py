from env.lighting_env import LightingEnv
import numpy as np
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C, ACER

l = LightingEnv("arena.xml", 20, 5, 1, gc=False)
import cv2

env = DummyVecEnv([lambda: l])  # The algorithms require a vectorized environment to run

model = ACER(CnnPolicy, env, verbose=1, tensorboard_log="./ppo_lighing_tb/")
model.learn(total_timesteps=200000)

# while True:
#     obs = l.reset()
#     goal = l.goal
#     cv2.imwrite("goal.png", goal)
#     for i in range(100):
#         a = np.random.randint(5)
#         act = np.zeros((5,))
#         act[a] = 1
#         obs, reward, done, info = l.step(act)
#         print(reward)
#         cv2.imwrite("obs"+str(i)+".png", obs[:,:,:3]*255)
#         l.render()
#     assert(False)