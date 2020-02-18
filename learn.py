from env.light_env import LightEnv
# from env.meta_env import MetaEnv
import numpy as np
from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy, MlpLstmPolicy, MlpPolicy
from stable_baselines.common.policies import MlpLstmPolicyCustomTConv, MlpLstmPolicyDM, MlpLstmPolicyTraj
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import A2C, ACER, PPO2, TRPO
import argparse
import torch as th
import cv2

def induction(structure, num, horizon, l, images=False):
    ##### INDUCTION #####
    ##### OPTIMAL POLICY 1 
    if structure == "masterswitch":
        it = None
        for i in range(num):
            p = l._get_obs()
            if images:
                 pi = l._get_obs(images=True)
            p = p.reshape((1, -1))
            a = np.zeros((1, num+1))
            a[:,i] = 1

            if images:
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
        for i in range(num):
            if i != it:
                p = l._get_obs()
                if images:
                    pi = l._get_obs(images=True)
                p = p.reshape((1, -1))
                a = np.zeros((1, num+1))
                a[:,i] = 1
                
                if images:
                    mem = np.concatenate([np.expand_dims(pi.flatten(), 0), a], 1)
                else:
                    mem = np.concatenate([p[:,:args.num], a], 1)
                    
                epbuf = np.concatenate([epbuf, mem], 0)
                l.step(i, count = False)
        ln = epbuf.shape[0]
        buf = np.zeros((2 * args.horizon - 1, epbuf.shape[1]))
        buf[:ln] = epbuf
    else:
        for i in range(num):
            p = l._get_obs()
            if images:
                pi = l._get_obs(images=True)
            p = p.reshape((1, -1))
            a = np.zeros((1, num+1))
            a[:,i] = 1

            if images:
                mem = np.concatenate([np.expand_dims(pi.flatten(), 0), a], 1)
            else:
                mem = np.concatenate([p[:,:args.num], a], 1)
                
            if i == 0:
                epbuf = mem
            else: 
                epbuf = np.concatenate([epbuf, mem], 0)
            l.step(i, count = False)
        buf = epbuf
    return buf

def predict(buf, F, structure, num):
    s = th.FloatTensor(buf[:,:-(num+1)]).float().cuda()
    a = th.FloatTensor(buf[:,-(1+num):]).float().cuda()
    predgt = th.clamp(F(s, a), 0, 1)
    return predgt.cpu().detach().numpy().flatten()
    



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
    
    if args.method == "traj_tconv":
        policy = MlpLstmPolicyCustomTConv #MlpLstmPolicy
        policy_kwargs = {"horizon":args.horizon, "num":args.num, "structure":args.structure}
    elif args.method == "trajlstm":
        policy = MlpLstmPolicyDM #MlpLstmPolicy
        policy_kwargs = {"horizon":args.horizon, "num":args.num, "structure":args.structure}
    elif args.method == "traj":
        policy = MlpLstmPolicyTraj #MlpLstmPolicy
        policy_kwargs = {"horizon":args.horizon, "num":args.num, "structure":args.structure}
    else:
        policy = MlpLstmPolicy
        policy_kwargs = None
    
    gc = 1 - args.fixed_goal

    if args.method == "gt":
        l = LightEnv(args.horizon, 
                 args.num, 
                 "gt",
                 args.structure, 
                 gc, 
                 filename="exp2/"+str(gc)+"_"+args.method,
                    seen = args.seen)
    
        env = DummyVecEnv(1 * [lambda: l]) 
        model = PPO2(policy, env, verbose=1, vf_coef=0.0001, noptepochs=4, nminibatches=1, policy_kwargs = policy_kwargs)
        
        for ptt in ["50000", "40000", "30000", "20000", "10000"]:
            try:
                model.load("saved_policies/policy_h"+str(args.horizon)+\
                                   "_"+str(args.structure)+\
                                   "_"+str(args.method) +\
                                   "_I"+str(args.images)+\
                                    "_S"+str(args.seen)+\
                                  "_"+str(ptt))
                break
            except:
                pass
            
        for mep in range(300000):
            l.keep_struct = False
            obs = env.reset()
            l.keep_struct = True
            before = l.eps
            model.learn(total_timesteps=500)
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
            
            if mep % 10000 == 0:
                model.save("saved_policies/policy_h"+str(args.horizon)+\
                           "_"+str(args.structure)+\
                           "_"+str(args.method) +\
                           "_I"+str(args.images)+\
                           "_S"+str(args.seen)+\
                          "_"+str(mep))
                    
    elif (args.method == "traj") or (args.method == "traj_tconv"):
        if args.images:
            imsize = 32*32*3
            stsize = imsize + args.num+1
            if args.structure == "masterswitch":
                st = ((args.horizon*2 - 1) * stsize)
            else:
                st = (args.horizon*stsize)
        else:
            if args.structure == "masterswitch":
                st = (args.horizon*(2*args.num+1) + (args.horizon-1)*(2*args.num+1))
            else:
                st = (args.horizon*(2*args.num+1))
            
        l = LightEnv(args.horizon, 
                 args.num, 
                 st,
                 args.structure, 
                 gc, 
                 filename="exp2/"+ str(gc)+"_"+args.method,
                    seen = args.seen, images=args.images)
        env = DummyVecEnv(1 * [lambda: l]) 
        model = PPO2(policy, env, verbose=1, vf_coef=0.0001, noptepochs=4, nminibatches=1, policy_kwargs=policy_kwargs)
        
        for ptt in ["50000", "40000", "30000", "20000", "10000"]:
            try:
                model.load("saved_policies/policy_h"+str(args.horizon)+\
                                   "_"+str(args.structure)+\
                                   "_"+str(args.method) +\
                                   "_I"+str(args.images)+\
                                    "_S"+str(args.seen)+\
                                  "_"+str(ptt))
                break
            except:
                pass
            
        for mep in range(300000):
            l.keep_struct = False
            obs = env.reset()
            l.keep_struct = True
            
            buf = induction(args.structure,args.num, args.horizon, l, images=args.images)
            traj = buf.flatten()
            l.state = np.zeros((args.num))
            l.traj = traj
            
            before = l.eps
            model.learn(total_timesteps=500)
            after = l.eps
            
            #### TEST ON UNSEEN CS
            l.keep_struct = False
            l.train = False
            for i in range(after - before):
                obs= env.reset()
                buf = induction(args.structure,args.num, args.horizon, l, images=args.images)
                traj = buf.flatten()
                l.state = np.zeros((args.num))
                l.traj = traj
                done = False
                rew = 0
                last = None
                while not done:
                    a, last = model.predict(obs, last)
                    obs, reward, done, info = env.step(a)
            l.keep_struct = True
            l.train = True
            
            if mep % 10000 == 0:
                model.save("saved_policies/policy_h"+str(args.horizon)+\
                           "_"+str(args.structure)+\
                           "_"+str(args.method) +\
                           "_I"+str(args.images)+\
                            "_S"+str(args.seen)+\
                          "_"+str(mep))
    elif args.method == "trajlstm":
        if args.images:
            imsize = 32*32*3
            stsize = imsize + args.num+1
            st = (stsize)
        else:
            st = (2*args.num+1)
            

        l = LightEnv(args.horizon, 
                 args.num, 
                 st,
                 args.structure, 
                 gc, 
                 filename="exp2/"+str(gc)+"_"+args.method,
                    seen = args.seen, dm=True, images=args.images)
        env = DummyVecEnv(1 * [lambda: l])
        model = PPO2(policy, env, verbose=1, vf_coef=0.0001, noptepochs=4, nminibatches=1, policy_kwargs = policy_kwargs)
        for ptt in ["50000", "40000", "30000", "20000", "10000"]:
            try:
                model.load("saved_policies/policy_h"+str(args.horizon)+\
                                   "_"+str(args.structure)+\
                                   "_"+str(args.method) +\
                                   "_I"+str(args.images)+\
                                    "_S"+str(args.seen)+\
                                  "_"+str(ptt))
                break
            except:
                pass
            
        for mep in range(300000):
            l.keep_struct = False
            obs = env.reset()
            l.keep_struct = True
            
            buf = induction(args.structure,args.num, args.horizon, l, images=args.images)
            traj = buf.flatten()
            l.state = np.zeros((args.num))
            l.traj = buf
            
            before = l.eps
            model.learn(total_timesteps=500)
            after = l.eps
            
            #### TEST ON UNSEEN CS
            l.keep_struct = False
            l.train = False
            for i in range(after - before):
                obs= env.reset()
                buf = induction(args.structure,args.num, args.horizon, l, images=args.images)
                traj = buf.flatten()
                l.state = np.zeros((args.num))
                l.traj = buf
                done = False
                rew = 0
                last = None
                while not done:
                    a, last = model.predict(obs, last)
                    obs, reward, done, info = env.step(a)
            l.keep_struct = True
            l.train = True
            
            if mep % 10000 == 0:
                model.save("saved_policies/policy_h"+str(args.horizon)+\
                           "_"+str(args.structure)+\
                           "_"+str(args.method) +\
                           "_I"+str(args.images)+\
                            "_S"+str(args.seen)+\
                          "_"+str(mep))
        
    elif (args.method == "trajF") or (args.method == "trajFi") or (args.method == "trajFia"):
        if args.structure == "masterswitch":
            st = (args.horizon*(2*args.num+1) + (args.horizon-1)*(2*args.num+1))
        else:
            st = (args.horizon*(2*args.num+1))
        tj = "gt"
        l = LightEnv(args.horizon, 
                 args.num, 
                 tj,
                 args.structure, 
                 gc, 
                 filename="exp2/"+str(gc)+"_Redo2_"+args.method,
                    seen = args.seen)
        env = DummyVecEnv(1 * [lambda: l])
        model = PPO2(policy, env, verbose=1, vf_coef=0.0001, noptepochs=4, nminibatches=1)
        
#         for ptt in ["50000", "40000", "30000", "20000", "10000"]:
#             try:
#                 model.load("saved_policies/policy_h"+str(args.horizon)+\
#                                    "_"+str(args.structure)+\
#                                    "_"+str(args.method) +\
#                                    "_I"+str(args.images)+\
#                                     "_S"+str(args.seen)+\
#                                   "_"+str(ptt))
#                 break
#             except:
#                 pass
        
        if args.images:
            addonn = "_I1"
        else:
            addonn = ""
    
        if args.method == "trajF":
            F = th.load("saved_models/cnn_L2_S"+str(args.seen)+"_h"+str(args.horizon)+\
                        "_"+str(args.structure)+addonn).cuda()
        elif args.method == "trajFia":
            F = th.load("saved_models/iter_attn_Redo_L2_S"+str(args.seen)+"_h"+str(args.horizon)+\
                        "_"+str(args.structure)+addonn).cuda()
        else:
            F = th.load("saved_models/iter_L2_S"+str(args.seen)+"_h"+str(args.horizon)+\
                        "_"+str(args.structure)+addonn).cuda()
        F = F.eval()
        trloss = []
        tsloss = []
        for mep in range(300000):
            l.keep_struct = False
            obs = env.reset()
            l.keep_struct = True
            
            buf = induction(args.structure,args.num, args.horizon, l, images=args.images)
            traj = buf.flatten()
            l.state = np.zeros((args.num))
            l.traj = traj
            pred = predict(buf, F,args.structure, args.num)
            trloss.append(((pred-l.gt)**2).sum())
            l.gt = pred

            before = l.eps
            model.learn(total_timesteps=500)
            after = l.eps
            
            #### TEST ON UNSEEN CS
            l.keep_struct = False
            l.train = False
            for i in range(after - before):
                obs= env.reset()
                buf = induction(args.structure,args.num, args.horizon, l, images=args.images)
                traj = buf.flatten()
                l.state = np.zeros((args.num))
                l.traj = traj
                pred = predict(buf, F,args.structure, args.num)
                tsloss.append(((pred-l.gt)**2).sum())
                l.gt = pred
                done = False
                rew = 0
                last = None
                while not done:
                    a, last = model.predict(obs, last)
                    old = obs
                    obs, reward, done, info = env.step(a)
            l.keep_struct = True
            l.train = True
            
            if mep % 10000 == 0:
                model.save("saved_policies/policy_Redo2_h"+str(args.horizon)+\
                           "_"+str(args.structure)+\
                           "_"+str(args.method) +\
                           "_I"+str(args.images)+\
                            "_S"+str(args.seen)+\
                          "_"+str(mep))
        print(np.mean(trloss), np.mean(tsloss))
    
    

# python3 learn.py --horizon 5 --num 5 --fixed-goal 0 --structure one_to_one --method two_policy
