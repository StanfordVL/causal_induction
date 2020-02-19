from env.light_env import LightEnv
import numpy as np
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
import argparse
import torch as th
import cv2
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Causal Meta-RL')
    parser.add_argument('--fixed-goal', type=int, default=0, help='fixed goal or no')
    parser.add_argument('--horizon', type=int, default=5, help='Env horizon')
    parser.add_argument('--num', type=int, default=5, help='Num Switches')
    parser.add_argument('--structure', type=str, default="one_to_one", help='Graph Structure')
    parser.add_argument('--seen', type=int, default=10, help='Number of seen environments')
    parser.add_argument('--images', type=int, default=0, help='Use Images')
    parser.add_argument('--data-dir', type=str, help='Directory to Store Data')
    args = parser.parse_args()
    
    ## Init Buffer
    gc = 1 - args.fixed_goal
    buffer = []
    gtbuffer = []
    num_episodes = 40000

    ## Set Horizon Based On Task
    if args.structure == "masterswitch":
        st = (args.horizon*(2*args.num+1) + (args.horizon-1)*(2*args.num+1))
    else:
        st = (args.horizon*(2*args.num+1))
        
    ## Init Env
    l = LightEnv(args.horizon, 
                 args.num, 
                 st,
                 args.structure, 
                 gc, 
                 filename=str(gc)+"_traj", 
                 seen = args.seen)
    env = DummyVecEnv(1 * [lambda: l]) 
    
    
    for q in range(num_episodes):
        ## Reset Causal Structure
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

        buffer.append(buf)
        gtbuffer.append(l.gt)
        if q % 10000 == 0:
            print(q)
    buffer = np.stack(buffer, 0)
    gtbuffer = np.stack(gtbuffer, 0)
    print(buffer.shape)
    print(gtbuffer.shape)
        
    np.save(args.data_dir+"buf40K_S"+\
                str(args.seen)+"_"+str(args.structure)+"_"+str(args.horizon)+\
                "_I"+str(args.images), buffer)
    np.save(args.data_dir+"gtbuf40K_S"+\
                str(args.seen)+"_"+str(args.structure)+"_"+str(args.horizon)+\
                "_I"+str(args.images), gtbuffer)
            
    

