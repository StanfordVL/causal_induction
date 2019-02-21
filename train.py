from env.lighting_env import LightingEnv
from env.meta_env import MetaEnv
import numpy as np
from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C, ACER, PPO2, TRPO
import argparse
import cv2

# python3 train.py --metaenv 0 --induction 0 --fixed-goal 0 --lstm 1 --vf-coef .0001 --nminibatches 1 --noptepochs 1 --nenvs 1 --horizon 10

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Causal Meta-RL')
    parser.add_argument('--metaenv', type=int, default=0, help='meta-rl or not')
    parser.add_argument('--induction', type=int, default=0, help='causal induction or not')
    parser.add_argument('--fixed-goal', type=int, default=0, help='fixed goal or no')
    parser.add_argument('--lstm', type=int, default=0, help='lstm policy or no')
    parser.add_argument('--vf-coef', type=float, default=0.0001, help='PPO vf_coef')
    parser.add_argument('--nminibatches', type=int, default=1, help='PPO nminibatches')
    parser.add_argument('--noptepochs', type=int, default=1, help='PPO noptepochs')
    parser.add_argument('--nenvs', type=int, default=1, help='PPO parallel envs')
    parser.add_argument('--horizon', type=int, default=10, help='Env horizon')
    args = parser.parse_args()
    
    if args.metaenv:
        l = MetaEnv("arena.xml", args.horizon, 5, 1, gc= 1 - args.fixed_goal, induction = args.induction)
    else:
        l = LightingEnv("arena.xml", args.horizon, 5, 1, gc= 1 - args.fixed_goal)
    env = DummyVecEnv(args.nenvs * [lambda: l])  # The algorithms require a vectorized environment to run

    logpath = "/cvgl2/u/surajn/workspace/causal_induction/lighting_tb/metaenv"+\
                str(args.metaenv) + "_induction" + str(args.induction) + "_horizon" + str(args.horizon) + "_gc"+str(1-args.fixed_goal) + "_lstm"+str(args.lstm) + \
                "_vf_coef"+str(args.vf_coef) + "_nminibatches" + str(args.nminibatches) + \
                "_noptepochs" + str(args.noptepochs) + "_nenvs" + str(args.nenvs)
    if args.lstm:
        policy = CnnLstmPolicy
    else:
        policy = CnnPolicy
    model = PPO2(policy, env, verbose=1, vf_coef=args.vf_coef, nminibatches=args.nminibatches, noptepochs=args.noptepochs, tensorboard_log=logpath)
    model.learn(total_timesteps=20000000)
    
    # while True:
    #     obs = l.reset()
    #     goal = l.goal
    #     cv2.imwrite("goal.png", goal)
    #     for i in range(100):
    #         a = np.random.randint(5)
    #         obs, reward, done, info = l.step(a)
    #         print(reward)
    #         cv2.imwrite("obs"+str(i)+".png", obs[:,:,:3]*255)
    #         l.render()
    #     assert(False)
