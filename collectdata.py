from env.light_env import LightEnv
# from env.meta_env import MetaEnv
import numpy as np
from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy, MlpLstmPolicy, MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2 # A2C, ACER, PPO2, TRPO
import argparse
import torch as th
import cv2
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Causal Meta-RL')
    parser.add_argument('--fixed-goal', type=int, default=0, help='fixed goal or no')
    parser.add_argument('--horizon', type=int, default=10, help='Env horizon')
    parser.add_argument('--num', type=int, default=1, help='K* horiscrzon induction phase')
    parser.add_argument('--structure', type=str, default="one_to_one", help='K* horiscrzon induction phase')
    parser.add_argument('--method', type=str, default="traj", help='K* horiscrzon induction phase')
    parser.add_argument('--seen', type=int, default=10, help='K* horiscrzon induction phase')
    parser.add_argument('--images', type=int, default=0, help='K* horiscrzon induction phase')
    args = parser.parse_args()
    
    policy = MlpLstmPolicy
    
    gc = 1 - args.fixed_goal
    buffer = []
    gtbuffer = []

    if args.method == "gt":
        l = LightEnv(args.horizon, 
                 args.num, 
                 "gt",
                 args.structure, 
                 gc, 
                 filename=str(gc)+"_"+args.method,
                seen = args.seen)
    
        env = DummyVecEnv(1 * [lambda: l]) 
        model = PPO2(policy, env, verbose=1, vf_coef=0.0001, noptepochs=4, nminibatches=1)
        for i in range(300000):
            l.keep_struct = False
            obs = env.reset()
            l.keep_struct = True
            before = l.eps
            model.learn(total_timesteps=5)
            after = l.eps
            
            #### TEST ON UNSEEN CS
            l.keep_struct = False
            l.train = False
            for i in range(after - before):
                obs= env.reset()
                done = False
                rew = 0
                last = None
                while not done:
                    a, last = model.predict(obs, last)
                    obs, reward, done, info = env.step(a)
            l.keep_struct = True
            l.train = True
                    
    elif args.method == "traj":
        if args.structure == "masterswitch":
            st = (args.horizon*(2*args.num+1) + (args.horizon-1)*(2*args.num+1))
        else:
            st = (args.horizon*(2*args.num+1))
        l = LightEnv(args.horizon, 
                 args.num, 
                 st,
                 args.structure, 
                 gc, 
                 filename=str(gc)+"_"+args.method,
                     seen = args.seen)
        env = DummyVecEnv(1 * [lambda: l]) 
        model = PPO2(policy, env, verbose=1, vf_coef=0.0001, noptepochs=4, nminibatches=1)
        for q in range(40000):
            l.keep_struct = False
            obs = env.reset()
            l.keep_struct = True
            ##### INDUCTION #####
            ##### OPTIMAL POLICY 1 
            if args.structure == "masterswitch":
                it = None
                for i in range(args.num):
                    p = l._get_obs()
                    if args.images:
                        pi = l._get_obs(images=True)
                    p = p.reshape((1, -1))
                    a = np.zeros((1, args.num+1))
                    a[:,i] = 1

                    if args.images:
                        mem = np.concatenate([np.expand_dims(pi.flatten(), 0), a], 1)
                    else:
                        mem = np.concatenate([p[:,:args.num], a], 1)
                        
                    if i == 0:
                        epbuf = mem
                    else: 
                        epbuf = np.concatenate([epbuf, mem], 0)
                    l.step(i, count = False)
                    p2 = l._get_obs()
                    if (p != p2).any():
                        it = i
                        break
                for i in range(args.num):
                    if i != it:
                        p = l._get_obs()
                        if args.images:
                            pi = l._get_obs(images=True)
                        p = p.reshape((1, -1))
                        a = np.zeros((1, args.num+1))
                        a[:,i] = 1

                        if args.images:
                            mem = np.concatenate([np.expand_dims(pi.flatten(), 0), a], 1)
                        else:
                            mem = np.concatenate([p[:,:args.num], a], 1)
                            
                        epbuf = np.concatenate([epbuf, mem], 0)
                        l.step(i, count = False)
                ln = epbuf.shape[0]
                buf = np.zeros((2 * args.horizon - 1, epbuf.shape[1]))
                buf[:ln] = epbuf
            else:
                for i in range(args.num):
                    p = l._get_obs()
                    if args.images:
                        pi = l._get_obs(images=True)
#                         print(pi.shape)
#                         pi = cv2.cvtColor((255*pi).astype(np.uint8), cv2.COLOR_BGR2RGB)
#                         cv2.imwrite("im"+str(i)+".png", pi)
                    p = p.reshape((1, -1))
                    a = np.zeros((1, args.num+1))
                    a[:,i] = 1

                    if args.images:
                        mem = np.concatenate([np.expand_dims(pi.flatten(), 0), a], 1)
                    else:
                        mem = np.concatenate([p[:,:args.num], a], 1)
                        
                    if i == 0:
                        epbuf = mem
                    else: 
                        epbuf = np.concatenate([epbuf, mem], 0)
                    l.step(i, count = False)
                buf = epbuf

#             print(l.gt.shape)
            buffer.append(buf)
            gtbuffer.append(l.gt)
            if q % 10000 == 0:
                print(q)
#             print(l.gt)
#             print(buf)
#             assert(False)
        buffer = np.stack(buffer, 0)
        gtbuffer = np.stack(gtbuffer, 0)
        print(buffer.shape)
        print(gtbuffer.shape)
        
        np.save("/cvgl2/u/surajn/workspace/causal_induction/data/buf40K_S"+\
                str(args.seen)+"_"+str(args.structure)+"_"+str(args.horizon)+\
                "_I"+str(args.images), buffer)
        np.save("/cvgl2/u/surajn/workspace/causal_induction/data/gtbuf40K_S"+\
                str(args.seen)+"_"+str(args.structure)+"_"+str(args.horizon)+\
                "_I"+str(args.images), gtbuffer)
            
    

# python3 learn.py --horizon 5 --num 5 --fixed-goal 0 --structure one_to_one --method two_policy
