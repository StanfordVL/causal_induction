from env.lighting_env import LightingEnv
from env.meta_env import MetaEnv
from env.induction_env import InductionEnv
from env.twopolicy_env import TwoPolicyEnv
# from env.meta_env import MetaEnv
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
    parser.add_argument('--metaenv', type=int, default=0, help='meta-rl or not')
    parser.add_argument('--induction', type=int, default=0, help='causal induction or not')
    parser.add_argument('--fixed-goal', type=int, default=0, help='fixed goal or no')
    parser.add_argument('--lstm', type=int, default=0, help='lstm policy or no')
    parser.add_argument('--vf-coef', type=float, default=0.0001, help='PPO vf_coef')
    parser.add_argument('--nminibatches', type=int, default=1, help='PPO nminibatches')
    parser.add_argument('--noptepochs', type=int, default=1, help='PPO noptepochs')
    parser.add_argument('--nenvs', type=int, default=1, help='PPO parallel envs')
    parser.add_argument('--horizon', type=int, default=10, help='Env horizon')
    parser.add_argument('--images', type=int, default=1, help='K* horiscrzon induction phase')
    parser.add_argument('--masterswitch', type=int, default=1, help='K* horiscrzon induction phase')
    parser.add_argument('--random', type=int, default=0, help='K* horiscrzon induction phase')
    parser.add_argument('--supervised', type=int, default=1, help='K* horiscrzon induction phase')
    parser.add_argument('--trainsteps', type=int, default=200000, help='K* horiscrzon induction phase')
    args = parser.parse_args()
    
    # SUPERVISED
    # l = InductionEnv("arena.xml", args.horizon, 5, 1, images=args.images)
    l = TwoPolicyEnv("arena.xml", args.horizon, 6, 1, gc= 1 - args.fixed_goal,  images=args.images, masterswitch=args.masterswitch, random=args.random, supervised=args.supervised)


    env = DummyVecEnv(args.nenvs * [lambda: l])  # The algorithms require a vectorized environment to run

#     logpath = "/cvgl2/u/surajn/workspace/causal_induction/lighting_tb/lstm_induction_horizon_experiment2/metaenv"+\
#                 str(args.metaenv) + "_induction" + str(args.induction) + "_horizon" + str(args.horizon) + "_k" + str(args.k) + \
#                 "_gc"+str(1-args.fixed_goal) + "_lstm"+str(args.lstm) + \
#                 "_vf_coef"+str(args.vf_coef) + "_nminibatches" + str(args.nminibatches) + \
#                 "_noptepochs" + str(args.noptepochs) + "_nenvs" + str(args.nenvs)
    
    if args.images:
        if args.lstm:
            policy = CnnLstmPolicy
        else:
            policy = CnnPolicy
    else:
        if args.lstm:
            policy = MlpLstmPolicy
        else:
            policy = MlpPolicy
            
    if args.supervised:
        try:
            a = np.load("data/heur_buf200K_ms"+str(args.masterswitch)+"_h"+str(args.horizon) + ".npy")
            l.buf = th.FloatTensor(a[:200000])
            a2 = np.load("data/heur_gtbuf200K_ms"+str(args.masterswitch)+"_h"+str(args.horizon) + ".npy")
            l.gtbuf = th.FloatTensor(a2[:200000])
        except:
            l.split = "train"
            for i in range(200000):
                obs = l.reset()
                done = False
                while not done:
                    # import time
                    # time.sleep(2)
                    if np.random.uniform() < 0.5:
                        # print("Expert")
                        if l.env.sim.model.light_active[ l.env.master_switch]:
                            while True:
                                a = np.random.randint(6)
                                if a != l.env.master_switch:
                                    break
                        else:
                            a = l.env.master_switch
                    else:
                        # print("Random")
                        a = np.random.randint(6)
                    obs, reward, done, info = l.step(a, train=False)
            buf = l.buf.numpy()
            gtbuf = l.gtbuf.numpy()
            np.save("data/heur_buf200K_ms"+str(args.masterswitch)+"_h"+str(args.horizon), buf)
            np.save("data/heur_gtbuf200K_ms"+str(args.masterswitch)+"_h"+str(args.horizon), gtbuf)
            a = np.load("data/heur_buf200K_ms"+str(args.masterswitch)+"_h"+str(args.horizon) + ".npy")
            l.buf = th.FloatTensor(a[:200000])
            a2 = np.load("data/heur_gtbuf200K_ms"+str(args.masterswitch)+"_h"+str(args.horizon) + ".npy")
            l.gtbuf = th.FloatTensor(a2[:200000])
            
        l.split = "train"
        l.train_supervised(args.trainsteps)

        model = PPO2(policy, env, verbose=1, vf_coef=args.vf_coef, nminibatches=args.nminibatches,
                     noptepochs=args.noptepochs,lam=0.99, ent_coef=0.2)#, tensorboard_log=logpath)
        model.learn(total_timesteps=args.horizon*500000)
        model.save("saved_models/2Policy_h"+str(args.horizon)+"_ms"+str(args.masterswitch) + "_sup"+str(args.supervised))
        l.split = "test"
        rews = []
        for i in range(1000):
            obs = env.reset()
            done = False
            rew = 0
            last = None
            while not done:
                a, last = model.predict(obs, last)
                obs, reward, done, info = env.step(a)
                rew += reward
            rews.append(rew)
        print(np.mean(rews))
    else:
        l.split = "train"
        for i in range(200000):
            obs = l.reset()
            done = False
            while not done:
                a = np.random.randint(6)
                obs, reward, done, info = l.step(a, train=False)
        if not args.random:
            model = PPO2(policy, env, verbose=1, vf_coef=args.vf_coef, nminibatches=args.nminibatches,
                     noptepochs=args.noptepochs,lam=0.99, ent_coef=0.2)#, tensorboard_log=logpath)
            model.learn(total_timesteps=args.horizon*800000)
            model.save("saved_models/2Policy_h"+str(args.horizon)+"_ms"+str(args.masterswitch) + "_sup"+str(args.supervised))
            l.split = "test"
            rews = []
            for i in range(1000):
                obs = env.reset()
                done = False
                rew = 0
                last = None
                while not done:
                    a, last = model.predict(obs, last)
                    obs, reward, done, info = env.step(a)
                    rew += reward
                rews.append(rew)
            print(np.mean(rews))
        else:
            for i in range(800000):
                obs = l.reset()
                done = False
                while not done:
                    a = np.random.randint(6)
                    obs, reward, done, info = l.step(a, train=False)
    
    
# python3 train2policy.py --metaenv 2 --induction 0 --fixed-goal 0 --lstm 1 --vf-coef 0.0001 --nminibatches 1 --noptepochs 4 --nenvs 1 --horizon 20 --images 0 --masterswitch 1 --random 1
    

