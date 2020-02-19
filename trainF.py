import os
import numpy as np
import torch as th
import argparse

from F_models import SupervisedModelCNN, IterativeModel, IterativeModelAttention


def train_supervised(F, buf, gtbuf, num, steps = 1, bs=32, images=False):
    buf = th.FloatTensor(buf).float()
    gtbuf = th.FloatTensor(gtbuf).float()
    optimizer = th.optim.Adam(F.parameters(), lr=0.0001)
    for step in range(steps):
        optimizer.zero_grad()
        perm = th.randperm(buf.size(0)-5000)
        testperm = th.randperm(5000) + 35000

        idx = perm[:bs]
        samples = buf[idx]
        gts= gtbuf[idx]
        testidx = testperm[:bs]
        testsamples = buf[testidx]
        testgts= gtbuf[testidx]
        
        if images:
            split = 32*32*3
        else:
            split = num
            
        states = samples[:, :, :split].contiguous().view(bs, -1).cuda()
        actions = samples[:, :, split:].contiguous().view(bs, -1).cuda()
        groundtruth = gts.cuda()
        pred = F(states, actions)

        teststates = testsamples[:, :, :split].contiguous().view(bs, -1).cuda()
        testactions = testsamples[:, :, split:].contiguous().view(bs, -1).cuda()
        testgroundtruth = testgts.cuda()
        testpred = F(teststates, testactions)

        loss = ((pred - groundtruth)**2).sum(1).mean()
        testloss = ((testpred - testgroundtruth)**2).sum(1).mean()

        loss.backward()
        if step % 1000 == 0:
            print((loss / num).cpu().detach().numpy())
            print((testloss / num).cpu().detach().numpy())
            print(pred[0], groundtruth[0])
            print(step)
            print("_"*50)
        optimizer.step()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Causal Meta-RL')
    parser.add_argument('--fixed-goal', type=int, default=0, help='fixed goal or no')
    parser.add_argument('--horizon', type=int, default=10, help='Env horizon')
    parser.add_argument('--num', type=int, default=1, help='Num lights')
    parser.add_argument('--structure', type=str, default="one_to_one", help='Causal Structure')
    parser.add_argument('--type', type=str, default="cnn", help='Model Type')
    parser.add_argument('--seen', type=int, default=10, help='Num seen')
    parser.add_argument('--images', type=int, default=0, help='Images or no')
    parser.add_argument('--data-dir', type=str, help='Data Dir')

    args = parser.parse_args()
    
    
    if args.type == "cnn":
        if args.structure == "masterswitch":
            msv = True
            F = SupervisedModelCNN(2*args.horizon -1,args.num,  ms = msv, images=args.images).cuda()
        else:
            msv = False
            F = SupervisedModelCNN(args.horizon,args.num,  ms = msv, images=args.images).cuda()
    elif args.type == "iter":
        if args.structure == "masterswitch":
            msv = True
            F = IterativeModel(2*args.horizon -1,args.num,  ms = msv, images=args.images).cuda()
        else:
            msv = False
            F = IterativeModel(args.horizon, args.num, ms = msv, images=args.images).cuda()
    elif args.type == "iter_attn":
        if args.structure == "masterswitch":
            msv = True
            F = IterativeModelAttention(2*args.horizon -1,args.num,  ms = msv, images=args.images).cuda()
        else:
            msv = False
            F = IterativeModelAttention(args.horizon, args.num, ms = msv, images=args.images).cuda()
    else:
        raise NotImplementedError 
        
    if args.images:
        addonn = "_I1"
    else:
        addonn = ""
    
    a = np.load(args.data_dir+ "buf40K_S"+str(args.seen)+\
                "_"+str(args.structure)+"_"+str(args.horizon) + addonn + ".npy")
    a2 = np.load(args.data_dir+ "gtbuf40K_S"+str(args.seen)+\
                 "_"+str(args.structure)+"_"+str(args.horizon) + addonn + ".npy")
    
    print(a.shape, a2.shape)
    train_supervised(F, a, a2, args.num, steps=2000, bs=512, images=args.images)
    th.save(F, args.data_dir+\
            str(args.type)+"_Redo_L2_S"+str(args.seen)+"_h"+str(args.horizon)+"_"+str(args.structure) \
            + "_I"+str(args.images))