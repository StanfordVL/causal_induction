from env.light_env import LightEnv
# from env.meta_env import MetaEnv
import numpy as np
# from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy, MlpLstmPolicy, MlpPolicy
# from stable_baselines.common.policies import MlpLstmPolicyCustomTConv, MlpLstmPolicyDM, MlpLstmPolicyTraj
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
# from stable_baselines import A2C, ACER, PPO2, TRPO
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
                im = l.sim.render(width=480,height=480,camera_name="birdview")
                cv2.imwrite('o'+str(i)+'.png', cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
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
    predgt = F(s, a)
    print(predgt)
    assert(False)
    return predgt.cpu().detach().numpy().flatten()


def f1score(pred, gt):
    p = 1 * (pred > 0.5)
    
    if np.sum(p) == 0:
        prec = 0
    else:
        prec = np.sum(gt * p) / np.sum(p)
    rec =np.sum(gt*p) / np.sum(gt)
#     print(prec, rec)
    if (prec == 0) and (rec==0):
        return 0
    return 2 * (prec * rec) / (prec+rec)
    



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
    
    
    gc = 1 - args.fixed_goal

    if True:
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
                 filename="exp/"+str(gc)+"_"+args.method,
                    seen = args.seen)
        env = DummyVecEnv(1 * [lambda: l])
#         model = PPO2(policy, env, verbose=1, vf_coef=0.0001, noptepochs=4, nminibatches=1)
        
        if args.images:
            addonn = "_I1"
        else:
            addonn = ""
    
        if args.method == "trajF":
            F = th.load("saved_models/cnn_Redo_L2_S"+str(args.seen)+"_h"+str(args.horizon)+\
                        "_"+str(args.structure)+addonn).cuda()
        elif args.method == "trajFia":
            F = th.load("saved_models/iter_attn_Redo_L2_S"+str(args.seen)+"_h"+str(args.horizon)+\
                        "_"+str(args.structure)+addonn).cuda()
        else:
            F = th.load("saved_models/iter_Redo_L2_S"+str(args.seen)+"_h"+str(args.horizon)+\
                        "_"+str(args.structure)+addonn).cuda()
        F = F.eval()
        trloss = []
        tsloss = []
        for mep in range(100):
            if np.random.uniform() < 0.9:
                continue
            l.keep_struct = False
            obs = env.reset()
            l.keep_struct = True
            
            buf = induction(args.structure,args.num, args.horizon, l, images=args.images)
            pred = predict(buf, F,args.structure, args.num)
            f = f1score(pred, l.gt)
            
            trloss.append(f)

            #### TEST ON UNSEEN CS
            l.keep_struct = False
            l.train = False
            for i in range(1):
                obs= env.reset()
#                 print("________")
                buf = induction(args.structure,args.num, args.horizon, l, images=args.images)
                
#                 print("\n\n\n")
                pred = predict(buf, F,args.structure, args.num)
                

                f = f1score(pred, l.gt)
#                 print(f)
#                 if mep == 1:
#                     assert(False)
#                 print(pred)
#                 print(l.gt)
#                 assert(False)
                
                tsloss.append(f)
#                 assert(False)
                
            l.keep_struct = True
            l.train = True
#         assert(False)
            
        print(np.mean(trloss), np.mean(tsloss))
    
    

# python3 learn.py --horizon 5 --num 5 --fixed-goal 0 --structure one_to_one --method two_policy
