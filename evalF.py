from env.light_env import LightEnv
import numpy as np
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
import argparse
import torch as th
import cv2

def induction(structure, num, horizon, l, images=False):
    '''Heuristic Policy to collect interaction data'''
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
#                 im = l.sim.render(width=480,height=480,camera_name="birdview")
#                 cv2.imwrite('o'+str(i)+'.png', cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
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
    '''Predict Graph'''
    s = th.FloatTensor(buf[:,:-(num+1)]).float().cuda()
    a = th.FloatTensor(buf[:,-(1+num):]).float().cuda()
    predgt = F(s, a)
    return predgt.cpu().detach().numpy().flatten()


def f1score(pred, gt):
    '''Compute F1 Score'''
    p = 1 * (pred > 0.5)
    
    if np.sum(p) == 0:
        prec = 0
    else:
        prec = np.sum(gt * p) / np.sum(p)
    rec =np.sum(gt*p) / np.sum(gt)
    if (prec == 0) and (rec==0):
        return 0
    return 2 * (prec * rec) / (prec+rec)
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Causal Meta-RL')
    parser.add_argument('--fixed-goal', type=int, default=0, help='fixed goal or no')
    parser.add_argument('--horizon', type=int, default=10, help='Env horizon')
    parser.add_argument('--num', type=int, default=1, help='num lights')
    parser.add_argument('--structure', type=str, default="one_to_one", help='causal structure')
    parser.add_argument('--method', type=str, default="traj", help='method')
    parser.add_argument('--seen', type=int, default=10, help='num seen')
    parser.add_argument('--images', type=int, default=0, help='images or no')
    parser.add_argument('--data-dir', type=str, help='data dir')
    args = parser.parse_args()
    
    
    gc = 1 - args.fixed_goal

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
        
    if args.images:
        addonn = "_I1"
    else:
        addonn = ""
    
    if args.method == "trajF":
        F = th.load(args.data_dir+"cnn_Redo_L2_S"+str(args.seen)+"_h"+str(args.horizon)+\
                        "_"+str(args.structure)+addonn).cuda()
    elif args.method == "trajFia":
        F = th.load(args.data_dir+"iter_attn_Redo_L2_S"+str(args.seen)+"_h"+str(args.horizon)+\
                        "_"+str(args.structure)+addonn).cuda()
    else:
        F = th.load(args.data_dir+"iter_Redo_L2_S"+str(args.seen)+"_h"+str(args.horizon)+\
                        "_"+str(args.structure)+addonn).cuda()
    F = F.eval()
    trloss = []
    tsloss = []
    for mep in range(100):
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
            buf = induction(args.structure,args.num, args.horizon, l, images=args.images)
                
            pred = predict(buf, F,args.structure, args.num)
                

            f = f1score(pred, l.gt)
            tsloss.append(f)
                
            l.keep_struct = True
            l.train = True
            
    print(np.mean(trloss), np.mean(tsloss))
    
    
