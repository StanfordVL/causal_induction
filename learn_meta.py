# from env.light_env import LightEnv
from env.meta_env import MetaEnv
import numpy as np
from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy, MlpLstmPolicy, MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import A2C, ACER, PPO2, TRPO
import argparse
import torch as th
import cv2

# from stable_baselines.ddpg.policies import CnnPolicy as ddpgcnn
# from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
# from stable_baselines import DDPG

# python3 train.py --metaenv 0 --induction 0 --fixed-goal 0 --lstm 1 --vf-coef .0001 --nminibatches 1 --noptepochs 1 --nenvs 1 --horizon 10

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Causal Meta-RL')
    parser.add_argument('--fixed-goal', type=int, default=0, help='fixed goal or no')
    parser.add_argument('--horizon', type=int, default=10, help='Env horizon')
    parser.add_argument('--num', type=int, default=1, help='K* horiscrzon induction phase')
    parser.add_argument('--structure', type=str, default="one_to_one", help='K* horiscrzon induction phase')
    parser.add_argument('--method', type=str, default="one_policy", help='K* horiscrzon induction phase')
    parser.add_argument('--random', type=int, default=0, help='K* horiscrzon induction phase')
    args = parser.parse_args()

    
    l = MetaEnv(args.horizon, args.num, args.method, args.structure, 1 - args.fixed_goal, random=args.random)
    env = DummyVecEnv(1 * [lambda: l])  # The algorithms require a vectorized environment to run
    if args.method == "one_policy":
        policy = MlpLstmPolicy
        model = PPO2(policy, env, verbose=1, vf_coef=0.0001, noptepochs=4, nminibatches=1,lam=0.99, ent_coef=0.2)
        model.learn(total_timesteps=2*args.horizon*10000000)
    elif args.method == "inference":
        policy = MlpLstmPolicy
        model = PPO2(policy, env, verbose=1, vf_coef=0.0001, noptepochs=4, nminibatches=1,lam=0.99, ent_coef=0.2)
        model.learn(total_timesteps=args.horizon*10000000)
    elif args.method == "gt":
        policy = MlpLstmPolicy
        model = PPO2(policy, env, verbose=1, vf_coef=0.0001, noptepochs=4, nminibatches=1,lam=0.99, ent_coef=0.2)#, tensorboard_log=logpath)
        model.learn(total_timesteps=args.horizon*500000)
    else:
        for i in range(10000):
            obs = l.reset()
            done = False
            while not done:
                a = np.random.randint(args.num + 1)
                obs, reward, done, info = l.step(a)
    #             print(obs[:(2*args.num)], reward, done)
    #             print("_"*50)
        if args.random:
            for i in range(2000000):
                obs = l.reset()
                done = False
                while not done:
                    a = np.random.randint(args.num + 1)
                    obs, reward, done, info = l.step(a)
        else:
            policy = MlpLstmPolicy
            model = PPO2(policy, env, verbose=1, vf_coef=0.0001, noptepochs=4, nminibatches=1,lam=0.99, ent_coef=0.2)#, tensorboard_log=logpath)
            model.learn(total_timesteps=args.horizon*2000000)

# python3 learn.py --horizon 5 --num 5 --fixed-goal 0 --structure one_to_one --method two_policy
