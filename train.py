from env.lighting_env import LightingEnv
from env.meta_env import MetaEnv
from env.modelbased_env import ModelBasedEnv
import numpy as np
from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy, MlpPolicy, MlpLstmPolicy
from stable_baselines.deepq.policies import CnnPolicy as cnnQ
from stable_baselines.deepq.policies import MlpPolicy as mlpQ
from stable_baselines.deepq.policies import MixedPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C, ACER, PPO2, TRPO, DQN
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
    parser.add_argument('--k', type=int, default=1, help='K* horiscrzon induction phase')
    parser.add_argument('--images', type=int, default=1, help='K* horiscrzon induction phase')
    parser.add_argument('--masterswitch', type=int, default=1, help='K* horiscrzon induction phase')
    args = parser.parse_args()
    
    if args.metaenv == 1:
        l = MetaEnv("arena.xml", args.horizon, 6, 1, gc= 1 - args.fixed_goal,
                    induction = args.induction, k = args.k, images=args.images,
                    masterswitch=args.masterswitch)
    elif args.metaenv == 0:
        l = LightingEnv("arena.xml", args.horizon, 6, 1, gc= 1 - args.fixed_goal,  images=args.images)
    elif args.metaenv == 2:
        l = ModelBasedEnv("arena.xml", args.horizon, 6, 1, gc= 1 - args.fixed_goal,  images=args.images, masterswitch=args.masterswitch)
        
        
    env = DummyVecEnv(args.nenvs * [lambda: l])  # The algorithms require a vectorized environment to run

    logpath = "/cvgl2/u/surajn/workspace/causal_induction/logs/lookup_tb/metaenv"+\
                str(args.metaenv) + "_induction" + str(args.induction) + "_horizon" + str(args.horizon) + "_k" + str(args.k) + \
                "_gc"+str(1-args.fixed_goal) + "_lstm"+str(args.lstm) + \
                "_vf_coef"+str(args.vf_coef) + "_nminibatches" + str(args.nminibatches) + \
                "_noptepochs" + str(args.noptepochs) + "_nenvs" + str(args.nenvs)

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
    model = PPO2(policy, env, verbose=1, vf_coef=args.vf_coef, nminibatches=args.nminibatches, noptepochs=args.noptepochs)#, tensorboard_log=logpath)
    
    # model = DQN(mlpQ, env, verbose=1, exploration_fraction = 0.1)
    model.learn(total_timesteps=10000000*args.horizon*(1+args.induction))
    model.save("saved_models/PPO2LSTM_lookup_h"+str(args.horizon)+"_ms"+str(args.masterswitch))

#     model.load("saved_models/PPO2LSTM_lookup_h"+str(args.horizon)+"_ms"+str(args.masterswitch))
    
    rews = []
    print("TRAIN")
    for i in range(10000):
        obs=env.reset()
        done = False
        rew = 0
        last = None
        while not done:
            a, last = model.predict(obs, last)
            obs, reward, done, info = env.step(a)
            rew += reward
        rews.append(rew)
    print(np.mean(rews))
        
    l = MetaEnv("arena.xml", args.horizon, 6, 1, gc= 1 - args.fixed_goal,
                    induction = args.induction, k = args.k, images=args.images,
                    masterswitch=args.masterswitch)
    env = DummyVecEnv(args.nenvs * [lambda: l])  # The algorithms require a vectorized environment to run
    
    rews = []
    print("TEST")
    for i in range(10000):
        obs=env.reset()
        done = False
        rew = 0
        last = None
        while not done:
            a, last = model.predict(obs, last)
            obs, reward, done, info = env.step(a)
            rew += reward
        rews.append(rew)
    print(np.mean(rews))
            
    
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
