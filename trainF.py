import os
import numpy as np
import torch as th
import argparse

from F_models import SupervisedModelCNN, IterativeModel, IterativeModelAttention


# def celoss(pred, groundtruth, ms=False):
#     loss_fn = th.nn.NLLLoss()
#     sf = th.nn.Softmax(dim=1)
#     loss6 = None
#     pred1 = sf(pred[:, :5])
#     pred2 = sf(pred[:,5:10])
#     pred3 = sf(pred[:,10:15])
#     pred4 = sf(pred[:,15:20])
#     pred5 = sf(pred[:,20:25])

#     loss1 = loss_fn((pred1+0.00001).log(), groundtruth[:, :5].argmax(1))
#     loss2 = loss_fn((pred2+0.00001).log(), groundtruth[:, 5:10].argmax(1))
#     loss3 = loss_fn((pred3+0.00001).log(), groundtruth[:, 10:15].argmax(1))
#     loss4 = loss_fn((pred4+0.00001).log(), groundtruth[:, 15:20].argmax(1))
#     loss5 = loss_fn((pred5+0.00001).log(), groundtruth[:, 20:25].argmax(1))
#     loss = loss1 + loss2 + loss3 + loss4 + loss5

        
#     if ms:
#         pred6 = sf(pred[:,25:30])
# #             print(pred5[:5])
# #             print(pred6[:5])
# #             print(groundtruth[:5, 20:25].argmax(1))
# #             print(groundtruth[:5, 25:30].argmax(1))
#         loss6 = loss_fn((pred6+0.00001).log(), groundtruth[:, 25:30].argmax(1))
# #             loss += 10*loss6
# #         print(loss1.cpu().detach().numpy(), loss2.cpu().detach().numpy(), loss3.cpu().detach().numpy(), loss4.cpu().detach().numpy(), loss5.cpu().detach().numpy(), loss6.cpu().detach().numpy())
#     return loss, loss6 

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
    parser.add_argument('--num', type=int, default=1, help='K* horiscrzon induction phase')
    parser.add_argument('--structure', type=str, default="one_to_one", help='K* horiscrzon induction phase')
    parser.add_argument('--method', type=str, default="one_policy", help='K* horiscrzon induction phase')
    parser.add_argument('--type', type=str, default="cnn", help='K* horiscrzon induction phase')
    parser.add_argument('--seen', type=int, default=10, help='K* horiscrzon induction phase')
    parser.add_argument('--images', type=int, default=0, help='K* horiscrzon induction phase')
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
    
    a = np.load("/cvgl2/u/surajn/workspace/causal_induction/data/buf40K_S"+str(args.seen)+\
                "_"+str(args.structure)+"_"+str(args.horizon) + addonn + ".npy")
    a2 = np.load("/cvgl2/u/surajn/workspace/causal_induction/data/gtbuf40K_S"+str(args.seen)+\
                 "_"+str(args.structure)+"_"+str(args.horizon) + addonn + ".npy")
    
    print(a.shape, a2.shape)
    train_supervised(F, a, a2, args.num, steps=100000, bs=512, images=args.images)
    th.save(F, "/cvgl2/u/surajn/workspace/causal_induction/saved_models/"+\
            str(args.type)+"_Redo_L2_S"+str(args.seen)+"_h"+str(args.horizon)+"_"+str(args.structure) \
            + "_I"+str(args.images))