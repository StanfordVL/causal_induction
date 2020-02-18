from env.light_env import LightEnv
# from env.meta_env import MetaEnv
import numpy as np
# from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy, MlpLstmPolicy, MlpPolicy
# from stable_baselines.common.policies import MlpLstmPolicyCustomTConv, MlpLstmPolicyDM, MlpLstmPolicyTraj, MlpLstmPolicyMix
# from stable_baselines.common.vec_env import DummyVecEnv
# from stable_baselines import A2C, ACER, PPO2, TRPO
import argparse
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import cv2


        
class BCPolicy(nn.Module):
    """
    Base Class for a SRL network (autoencoder family)
    It implements a getState method to retrieve a state from observations
    """
    def __init__(self, num, structure, attention = False):
        super(BCPolicy, self).__init__()
        self.encoder_conv = nn.Sequential(
            # 224x224xN_CHANNELS -> 112x112x64
            nn.Conv2d(6, 8, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56x56x64
            nn.ReLU(inplace=True),
        )
        self.encoder_conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 27x27x64
            nn.ReLU(inplace=True),
        )
        self.encoder_conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 6x6x64
            nn.ReLU(inplace=True),
        )
        
        self.att = attention
        self.num = num
        if structure == "masterswitch":
            self.ins = self.num + 1
        else:
            self.ins = self.num
        
        self.attlayer = nn.Linear(128, num)
        self.structure = structure
        self.fc1 = nn.Linear(4 * 4 * 32, 128)
            
        if not self.att:
            if structure == "masterswitch":
                self.gfc1 = nn.Linear(num*num + num, 128)
            else:
                self.gfc1 = nn.Linear(num*num, 128)
        else:
            self.gfc1 = nn.Linear(self.num, 128)
            
            
#         self.gfc2 = nn.Linear(128, 128)
#         self.gfc3 = nn.Linear(128, 128)
#         self.gfc4 = nn.Linear(128, 128)
 
        if self.structure == "masterswitch":
            self.fc2 = nn.Linear(256+args.num, 64)
        else:
            self.fc2 = nn.Linear(256, 64)
#         self.fc3 = nn.Linear(256, 128)
#         self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, num)

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        
    def forward(self, x, gr):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        
#         g1 = self.relu(self.gfc2(g1))
#         g1 = self.relu(self.gfc3(g1))
#         g1 = self.relu(self.gfc4(g1))

        x = x.permute(0, 3, 1, 2).contiguous()

        e1 = self.encoder_conv(x)
        e2 = self.encoder_conv2(e1)
        e3 = self.encoder_conv3(e2)
        e3 = e3.view(e3.size(0), -1)
        encoding = self.relu(self.fc1(e3))
        if self.att:
            w = self.softmax(self.attlayer(encoding))
            if self.structure == "masterswitch":
                ms = gr.view((-1, self.ins, self.num))[:, -1, :]
                gr = gr.view((-1, self.ins, self.num))[:, :-1, :]
            else:
                gr = gr.view((-1, self.ins, self.num))
            gr_sel = th.bmm(gr, w.view(w.size(0), -1, 1))
            gr_sel = gr_sel.squeeze(-1)
            g1 = self.relu(self.gfc1(gr_sel))
        else:
            g1 = self.relu(self.gfc1(gr))

        if self.structure == "masterswitch":
            eout = th.cat([g1, encoding, ms], 1)
        else:
            eout = th.cat([g1, encoding], 1)
        a = self.relu(self.fc2(eout))
#         a = self.relu(self.fc3(a))
#         a = self.relu(self.fc4(a))
        a = self.fc5(a)
        return a
    
    
class BCPolicyMemory(nn.Module):
    """
    Base Class for a SRL network (autoencoder family)
    It implements a getState method to retrieve a state from observations
    """
    def __init__(self, num, structure):
        super(BCPolicyMemory, self).__init__()
        self.encoder_conv = nn.Sequential(
            # 224x224xN_CHANNELS -> 112x112x64
            nn.Conv2d(6, 8, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56x56x64
            nn.ReLU(inplace=True),
        )
        self.encoder_conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 27x27x64
            nn.ReLU(inplace=True),
        )
        self.encoder_conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 6x6x64
            nn.ReLU(inplace=True),
        )
        
        self.fc1 = nn.Linear(4 * 4 * 32, 128)
        
        self.aenc = nn.Linear(num+1, 128)

        if structure == "masterswitch":
            self.gfc1 = nn.Linear(num*num + num, 128)
        else:
            self.gfc1 = nn.Linear(num*num, 128)

        self.lstm = nn.LSTMCell(256, 256)
 
        self.fc2 = nn.Linear(256, 64)
        self.fc5 = nn.Linear(64, num)

        self.softmax = nn.Softmax()
        self.relu = nn.ReLU()
        
    def forward(self, x, a, hidden):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        x = x.permute(0, 3, 1, 2).contiguous()
        e1 = self.encoder_conv(x)
        e2 = self.encoder_conv2(e1)
        e3 = self.encoder_conv3(e2)
        e3 = e3.view(e3.size(0), -1)
        encoding = self.relu(self.fc1(e3))
        
        ae = self.relu(self.aenc(a))
        eout = th.cat([ae, encoding], 1)
        if hidden is None:
            hidden = self.lstm(eout)
        else:
            hidden = self.lstm(eout, hidden)

        a = self.relu(self.fc2(hidden[0]))
        a = self.fc5(a)
        return a, hidden


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
    
def train_bc(memory, policy, opt):
    if len(memory['state']) < 50:
        return
    opt.zero_grad()
    choices = np.random.choice(len(memory['state']), 32).astype(np.int32).tolist()
    states = [memory['state'][c] for c in choices]
    graphs = [memory['graph'][c] for c in choices]
    actions = [memory['action'][c] for c in choices]

    states = th.FloatTensor(states).cuda()
    graphs = th.FloatTensor(graphs).cuda()
    actions = th.LongTensor(actions).cuda()

    pred_acts = policy(states, graphs)
    # loss = ((pred_acts - actions)**2).sum(1).mean()
    celoss = nn.CrossEntropyLoss()
    loss = celoss(pred_acts, actions)
    l = loss.cpu().detach().numpy()
    loss.backward()
    opt.step()
    return l

def train_bclstm(trajs, policy, opt):
    if len(trajs) < 10:
        return
    celoss = nn.CrossEntropyLoss()
    opt.zero_grad()
    totalloss = 0
    choices = np.random.choice(len(trajs), 4).astype(np.int32).tolist()
    for t in choices:
        memory = trajs[t]
        hidden = None
        
        buf = memory['graph'][0]
        for w in range(buf.shape[0]):
            states = buf[w, :32*32*3].reshape(1, 32, 32, 3)
            sgg = np.zeros_like(states)
            states = np.concatenate([states, sgg], -1)
            actions = buf[w, 32*32*3:].reshape(1, -1)
            num_acts = actions.shape
            act, hidden = pol(th.FloatTensor(states).cuda(), th.FloatTensor(actions).cuda(), hidden)
        states = np.array(memory['state'])
        actions = np.array(memory['action'])
        preds = []
        for w in range(states.shape[0]):
            a = np.zeros(num_acts)
            
            pred_acts, hidden = pol(th.FloatTensor(states[w:w+1]).cuda(), th.FloatTensor(a).cuda(), hidden)
            preds.append(pred_acts)
        preds = th.cat(preds, 0)
        loss = celoss(preds, th.LongTensor(actions).cuda())
        totalloss += loss

    l = totalloss.cpu().detach().numpy()
    totalloss.backward()
    opt.step()
    return l

def eval_bc(policy, l, train=True, f=None, args=None):
    successes = []
    l.keep_struct = False
    l.train = train
    for mep in range(100):
        obs = l.reset()
        imobs = np.expand_dims(l._get_obs(images=True), 0)
        goalim = np.expand_dims(l.goalim, 0)
        
        if f is None:
            graph = np.expand_dims(l.gt.flatten(), 0)
        else:
            buf = induction(args.structure,args.num, args.horizon, l, images=args.images)
            traj = buf.flatten()

            l.state = np.zeros((args.num))
            l.traj = traj
            pred = predict(buf, f,args.structure, args.num)
            l.gt = pred
            graph = np.expand_dims(pred.flatten(), 0)
            
        # print(goalim.shape)
        # print(l.aj.shape)
        if (mep == 50) and (train):
            print(l.goal)
            print(l.aj)
#             print(l.ms)

        for k in range(args.horizon * 2):
            st = np.concatenate([imobs, goalim], 3)
            act = policy(th.FloatTensor(st).cuda(), th.FloatTensor(graph).cuda())
            action = act[0].argmax()
            
            obs, reward, done, info = l.step(action)
            if (mep == 50) and (train):
                print(action, obs[:5])
            imobs = np.expand_dims(l._get_obs(images=True), 0)
            if done:
                break

        successes.append(l._is_success(obs))
        # assert(False)
    return np.mean(successes)

def eval_bclstm(policy, l, train=True, args=None):
    successes = []
    l.keep_struct = False
    l.train = train
    for mep in range(100):
        hidden = None
        obs = l.reset()
        imobs = np.expand_dims(l._get_obs(images=True), 0)
        goalim = np.expand_dims(l.goalim, 0)
        
        buf = induction(args.structure,args.num, args.horizon, l, images=args.images)
        
        l.state = np.zeros((args.num))
        for w in range(buf.shape[0]):
            states = buf[w, :32*32*3].reshape(1, 32, 32, 3)
            sgg = np.zeros_like(states)
            states = np.concatenate([states, sgg], -1)
            actions = buf[w, 32*32*3:].reshape(1, -1)
            num_acts = actions.shape
            act, hidden = policy(th.FloatTensor(states).cuda(), th.FloatTensor(actions).cuda(), hidden)

        if (mep == 50) and (train):
            print(l.goal)
            print(l.aj)

        for k in range(args.horizon * 2):
            st = np.concatenate([imobs, goalim], 3)
            act, hidden = policy(th.FloatTensor(st).cuda(), th.FloatTensor(np.zeros(num_acts)).cuda(), hidden)
            action = act[0].argmax()
            
            obs, reward, done, info = l.step(action)
            if (mep == 50) and (train):
                print(action, obs[:5])
            imobs = np.expand_dims(l._get_obs(images=True), 0)
            if done:
                break

        successes.append(l._is_success(obs))
    return np.mean(successes)


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
    fname = "bcexp/polattn_"+str(gc)+"_"+args.method

    memsize = 10000
    memory = {'state':[], 'graph':[], 'action':[]}
    if args.method == 'trajlstm':
        pol = BCPolicyMemory(args.num, args.structure).cuda()
    else:
        pol = BCPolicy(args.num, args.structure, True).cuda()
    optimizer = th.optim.Adam(pol.parameters(), lr=0.0001)

    if args.method == "gt":
        l = LightEnv(args.horizon*2, 
                 args.num, 
                 "gt",
                 args.structure, 
                 gc, 
                 filename=fname,
                    seen = args.seen)
    
        successes = []
        l.keep_struct = False
        l.train = True
        for mep in range(100000):
            l.train = True
            obs = l.reset()

            curr = np.zeros((args.num))
            obs = curr
            imobs = l._get_obs(images=True)
            goalim = l.goalim
                    
            goal = l.goal
            for k in range(args.horizon*2):
                g = np.abs(goal - obs[:args.num])
                st = np.concatenate([imobs, goalim], 2)
                sss = 1.0*(np.dot(g, l.aj.T).T > 0.5)

                if args.structure == "masterswitch":
                    sss[l.ms] = 0
                    if sss.max() == 0:
                        break
                        
                action = np.argmax(sss)
                if args.structure == "masterswitch":
                    if obs[:5].max() == 0:
                        action = l.ms
                memory['state'].append(st)
                memory['graph'].append(l.gt.flatten())
                memory['action'].append(action)
                
                if np.random.uniform() < 0.3:
                    action = np.random.randint(args.num)
                else:
                    graph = np.expand_dims(l.gt.flatten(), 0)
                    act = pol(th.FloatTensor(np.expand_dims(st, 0)).cuda(), th.FloatTensor(graph).cuda())
                    action = act[0].argmax()

                obs, reward, done, info = l.step(action)
                imobs = l._get_obs(images=True)
                if done:
                    break
                    
            g = np.abs(goal - obs[:args.num])
            st = np.concatenate([imobs, goalim], 2)
            sss = 1.0*(np.dot(g, l.aj.T).T > 0.5)

            if args.structure == "masterswitch":
                if sss[l.ms]:
                    st = np.concatenate([imobs, goalim], 2)
                    memory['state'].append(st)
                    memory['graph'].append(l.gt.flatten())
                    memory['action'].append(l.ms)
                    obs, reward, done, info = l.step(l.ms)
#             assert(l._is_success(obs))
            memory['state'] = memory['state'][-memsize:]
            memory['graph'] = memory['graph'][-memsize:]
            memory['action'] = memory['action'][-memsize:]
            for _ in range(1):
                loss = train_bc(memory, pol, optimizer)
            if mep % 1000 == 0:
                print("Episode", mep, "Loss:" , loss )
                trainsc = eval_bc(pol, l, True, args=args)
                testsc = eval_bc(pol, l, False, args=args)
                with open(fname + "_S" + str(args.seen) + \
                          "_"+str(args.structure)+"_H"+str(args.horizon)+\
                          "_N"+str(args.num)+"_Ttrainsuccessrate.txt", "a") as f:
                    f.write(str(float(trainsc)) + "\n")
                with open(fname + "_S" + str(args.seen) + \
                          "_"+str(args.structure)+"_H"+str(args.horizon)+\
                          "_N"+str(args.num)+"_Ttestsuccessrate.txt", "a") as f:
                    f.write(str(float(testsc)) + "\n")
                    
                print("Train Success Rate:", trainsc)
                print("Test Success Rate:", testsc)
        
            successes.append(l._is_success(obs))
        print(np.mean(successes))
    elif (args.method == "trajF") or (args.method == "trajFi") or (args.method == "trajFia"):
        if args.structure == "masterswitch":
            st = (args.horizon*(2*args.num+1) + (args.horizon-1)*(2*args.num+1))
        else:
            st = (args.horizon*(2*args.num+1))
        tj = "gt"
        l = LightEnv(args.horizon*2, 
                 args.num, 
                 tj,
                 args.structure, 
                 gc, 
                 filename=fname,
                    seen = args.seen)

        if args.images:
            addonn = "_I1"
        else:
            addonn = ""
    
        if args.method == "trajF":
            FN = th.load("saved_models/cnn_Redo_L2_S"+str(args.seen)+"_h"+str(args.horizon)+\
                        "_"+str(args.structure)+addonn).cuda()
        elif args.method == "trajFia":
            FN = th.load("saved_models/iter_attn_Redo_L2_S"+str(args.seen)+"_h"+str(args.horizon)+\
                        "_"+str(args.structure)+addonn).cuda()
        else:
            FN = th.load("saved_models/iter_Redo_L2_S"+str(args.seen)+"_h"+str(args.horizon)+\
                        "_"+str(args.structure)+addonn).cuda()
        FN = FN.eval()
        successes = []
        l.keep_struct = False
        l.train = False
        for mep in range(100000):
            l.train = True
            obs = l.reset()
            goalim = l.goalim
            imobs = l._get_obs(images=True)
            
            buf = induction(args.structure,args.num, args.horizon, l, images=args.images)
            traj = buf.flatten()
            pred = predict(buf, FN,args.structure, args.num)
            l.state = np.zeros((args.num))

#             if args.structure == "masterswitch":
#                 mp = pred[:(args.num * args.num)].reshape((args.num,args.num))
#                 ms = np.argmax(pred[(args.num * args.num):])
#             else:
#                 mp = pred.reshape((args.num,args.num))
            
            curr = np.zeros((args.num))
            obs = curr
                    
            goal = l.goal
            for k in range(args.horizon*2):
                g = np.abs(goal - obs[:args.num])
                st = np.concatenate([imobs, goalim], 2)
                sss = 1.0*(np.dot(g, l.aj.T).T > 0.5)
                
                if args.structure == "masterswitch":
                    sss[l.ms] = 0
                    if sss.max() == 0:
                        break
                
                action = np.argmax(sss)
                if args.structure == "masterswitch":
                    if obs[:5].max() == 0:
                        action = l.ms
                memory['state'].append(st)
                memory['graph'].append(pred.flatten())
                memory['action'].append(action)
                
                if np.random.uniform() < 0.3:
                    action = np.random.randint(args.num)
                else:
                    pred = predict(buf, FN,args.structure, args.num)
                    graph = np.expand_dims(pred.flatten(), 0)
                    act = pol(th.FloatTensor(np.expand_dims(st, 0)).cuda(), th.FloatTensor(graph).cuda())
                    action = act[0].argmax()


                obs, reward, done, info = l.step(action)
                imobs = l._get_obs(images=True)
                if done:
                    break

            g = np.abs(goal - obs[:args.num])
            st = np.concatenate([imobs, goalim], 2)
            sss = 1.0*(np.dot(g, l.aj.T).T > 0.5)
            if args.structure == "masterswitch":
                if sss[l.ms]:
                    st = np.concatenate([imobs, goalim], 2)
                    memory['state'].append(st)
                    memory['graph'].append(pred.flatten())
                    memory['action'].append(l.ms)
                    obs, reward, done, info = l.step(l.ms)

            memory['state'] = memory['state'][-memsize:]
            memory['graph'] = memory['graph'][-memsize:]
            memory['action'] = memory['action'][-memsize:]
            for _ in range(1):
                loss = train_bc(memory, pol, optimizer)
            if mep % 1000 == 0:
                print("Episode", mep, "Loss:" , loss )
                trainsc = eval_bc(pol, l, True, f=FN, args=args)
                testsc = eval_bc(pol, l, False, f=FN, args=args)
                with open(fname + "_S" + str(args.seen) + \
                          "_"+str(args.structure)+"_H"+str(args.horizon)+\
                          "_N"+str(args.num)+"_Ttrainsuccessrate.txt", "a") as f:
                    f.write(str(float(trainsc)) + "\n")
                with open(fname + "_S" + str(args.seen) + \
                          "_"+str(args.structure)+"_H"+str(args.horizon)+\
                          "_N"+str(args.num)+"_Ttestsuccessrate.txt", "a") as f:
                    f.write(str(float(testsc)) + "\n")
                    
                print("Train Success Rate:", trainsc)
                print("Test Success Rate:", testsc)

            successes.append(l._is_success(obs))
        print(np.mean(successes))
    elif (args.method == "trajlstm"):
        if args.structure == "masterswitch":
            st = (args.horizon*(2*args.num+1) + (args.horizon-1)*(2*args.num+1))
        else:
            st = (args.horizon*(2*args.num+1))
        tj = "gt"
        l = LightEnv(args.horizon*2, 
                 args.num, 
                 tj,
                 args.structure, 
                 gc, 
                 filename=fname,
                    seen = args.seen)

        if args.images:
            addonn = "_I1"
        else:
            addonn = ""
    
        successes = []
        l.keep_struct = False
        l.train = False
        memsize = 100
        trajs = []
        
        for mep in range(100000):
            memory = {'state':[], 'graph':[], 'action':[]}
            hidden = None
            l.train = True
            obs = l.reset()
            goalim = l.goalim
            imobs = l._get_obs(images=True)
            
            
            buf = induction(args.structure,args.num, args.horizon, l, images=args.images)
            memory['graph'].append(buf)
            for w in range(buf.shape[0]):
                states = buf[w, :32*32*3].reshape(1, 32, 32, 3)
                sgg = np.zeros_like(states)
                states = np.concatenate([states, sgg], -1)
                actions = buf[w, 32*32*3:].reshape(1, -1)
                act, hidden = pol(th.FloatTensor(states).cuda(), th.FloatTensor(actions).cuda(), hidden)
            l.state = np.zeros((args.num))

            curr = np.zeros((args.num))
            obs = curr
                    
            goal = l.goal
            for k in range(args.horizon*2):
                g = np.abs(goal - obs[:args.num])
                st = np.concatenate([imobs, goalim], 2)
                sss = 1.0*(np.dot(g, l.aj.T).T > 0.5)
                
                if args.structure == "masterswitch":
                    sss[l.ms] = 0
                    if sss.max() == 0:
                        break
                
                action = np.argmax(sss)
                if args.structure == "masterswitch":
                    if obs[:5].max() == 0:
                        action = l.ms
                memory['state'].append(st)
                memory['action'].append(action)
                
                if np.random.uniform() < 0.3:
                    action = np.random.randint(args.num)
                else:
                    act, s_hidden = pol(th.FloatTensor(states).cuda(), th.FloatTensor(actions).cuda(), hidden)
                    action = act[0].argmax()


                obs, reward, done, info = l.step(action)
                imobs = l._get_obs(images=True)
                if done:
                    break

            g = np.abs(goal - obs[:args.num])
            st = np.concatenate([imobs, goalim], 2)
            sss = 1.0*(np.dot(g, l.aj.T).T > 0.5)
            if args.structure == "masterswitch":
                if sss[l.ms]:
                    st = np.concatenate([imobs, goalim], 2)
                    memory['state'].append(st)
                    memory['action'].append(l.ms)
                    obs, reward, done, info = l.step(l.ms)

#             memory['state'] = memory['state'][-memsize:]
#             memory['graph'] = memory['graph'][-memsize:]
#             memory['action'] = memory['action'][-memsize:]
            if len(memory['state']) != 0:
                trajs.append(memory)
            trajs = trajs[-memsize:]
            for _ in range(1):
                loss = train_bclstm(trajs, pol, optimizer)
            if mep % 1000 == 0:
                print("Episode", mep, "Loss:" , loss )
                trainsc = eval_bclstm(pol, l, True, args=args)
                testsc = eval_bclstm(pol, l, False, args=args)
                with open(fname + "_S" + str(args.seen) + \
                          "_"+str(args.structure)+"_H"+str(args.horizon)+\
                          "_N"+str(args.num)+"_Ttrainsuccessrate.txt", "a") as f:
                    f.write(str(float(trainsc)) + "\n")
                with open(fname + "_S" + str(args.seen) + \
                          "_"+str(args.structure)+"_H"+str(args.horizon)+\
                          "_N"+str(args.num)+"_Ttestsuccessrate.txt", "a") as f:
                    f.write(str(float(testsc)) + "\n")
                    
                print("Train Success Rate:", trainsc)
                print("Test Success Rate:", testsc)

            successes.append(l._is_success(obs))
        print(np.mean(successes))