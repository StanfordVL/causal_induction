from __future__ import print_function, division, absolute_import

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import math
import cv2

        
class ImageEncoder(nn.Module):
    """
    IMage Encoder
    """
    def __init__(self, num):
        super(ImageEncoder, self).__init__()
        self.encoder_conv = nn.Sequential(
            # 224x224xN_CHANNELS -> 112x112x64
            nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1),
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
        
        self.fc = nn.Linear(4 * 4 * 32, num)

        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        e1 = self.encoder_conv(x)
        e2 = self.encoder_conv2(e1)
        e3 = self.encoder_conv3(e2)
        e3 = e3.view(e3.size(0), -1)
        encoding = self.fc(e3)
        return encoding
    

class IterativeModelAttention(nn.Module):
    def __init__(self, horizon,num=5, ms=False, images=False):
        super(IterativeModelAttention, self).__init__()
        
        self.horizon = horizon
        self.ms = ms

        if self.ms:
            self.attnsize = num + 1
            self.outsize = num
        else:
            self.attnsize = num 
            self.outsize = num


        self.images = images
        self.num = num

        self.ie = ImageEncoder(num)
        
        self.fc1 = nn.Linear(2*num+1, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, self.attnsize + num)

        self.fc4 = nn.Linear(self.attnsize*self.outsize, self.attnsize + num)
        
        self.dp = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, s, a):
        if self.images:
            s_im = s.view(-1, 32, 32, 3).permute(0,3,1,2)
            senc = self.ie(s_im)
            sp = senc.view(-1, self.horizon, self.num)
        else:
            sp = s.view(-1, self.horizon, self.num)
        sp[:,:-1] = sp[:,1:] - sp[:,:-1]
        a = a.view(-1, self.horizon, self.num+1)
        e = th.cat([sp, a], 2)
        
        
        p = th.zeros((sp.size(0), self.attnsize, self.outsize)).cuda()
        for i in range(self.horizon):
            inn = e[:,i,:]
            e1 = self.relu(self.dp(self.fc1(inn)))
            e2 = self.relu(self.dp(self.fc2(e1)))
            e3 = self.fc3(e2)

            atn = self.softmax(e3[:, :self.attnsize]).unsqueeze(-1)
            e3 = self.sigmoid(e3[:, self.attnsize:].unsqueeze(1).repeat(1, self.attnsize, 1))
            r = atn * e3
            p = p + r
        
        e3 = self.fc4(p.view(-1, self.attnsize*self.num))
        atn = self.softmax(e3[:, :self.attnsize]).unsqueeze(-1)
        e3 = self.sigmoid(e3[:, self.attnsize:].unsqueeze(1).repeat(1, self.attnsize, 1))
        r = atn * e3
        p = p + r
        p = p.view(-1, self.attnsize*self.num)
        

        return p

class IterativeModel(nn.Module):
    def __init__(self, horizon,num=5, ms=False, images=False):
        super(IterativeModel, self).__init__()
        
        self.images = images
        self.ie = ImageEncoder(num)
        
        self.horizon = horizon
        self.ms = ms
        if self.ms:
            self.outsize = num**2 + num
        else:
            self.outsize = num**2 
            
        self.num = num
        
        self.fc1 = nn.Linear(2*num+1, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, self.outsize)
        
        self.fc4 = nn.Linear(self.outsize, self.outsize)
        
        self.cnn1 = nn.Conv1d(2*num+1, 256, kernel_size=3, padding=1)
        self.cnn2 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.cnn3 = nn.Conv1d(128, 128, kernel_size=3, padding=1)

        self.dp = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, s, a):
        if self.images:
            s_im = s.view(-1, 32, 32, 3).permute(0,3,1,2)
            senc = self.ie(s_im)
            sp = senc.view(-1, self.horizon, self.num)
        else:
            sp = s.view(-1, self.horizon, self.num)
        sp[:,1:] = sp[:,1:] - sp[:,:-1]
        a = a.view(-1, self.horizon, self.num+1)
        e = th.cat([sp, a], 2)
        
        e = e.permute(0,2,1)
        c2 = e
        
        p = th.zeros((sp.size(0), self.outsize)).cuda()
        for i in range(self.horizon):
            e1 = self.relu(self.dp(self.fc1(c2[:,:,i])))
            e2 = self.relu(self.dp(self.fc2(e1)))
            e3 = self.sigmoid(self.fc3(e2))    
            p = p + e3
        p = self.sigmoid(self.fc4(p))

        return p
    
class SupervisedModelCNN(nn.Module):
    def __init__(self, horizon,num=5, ms=False, images=False):
        super(SupervisedModelCNN, self).__init__()
        
        self.images = images
        self.ie = ImageEncoder(num)
        
        self.horizon = horizon
        self.ms = ms
        if self.ms:
            self.outsize = num**2 + num
        else:
            self.outsize = num**2 
            
        self.num = num
        
        self.cnn1 = nn.Conv1d(2*num+1, 256, kernel_size=3, padding=1)
        self.cnn2 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.cnn3 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(self.horizon*128, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, self.outsize)

        self.dp = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, s, a):
        if self.images:
            s_im = s.view(-1, 32, 32, 3).permute(0,3,1,2)
            senc = self.ie(s_im)
            sp = senc.view(-1, self.horizon, self.num)
        else:
            sp = s.view(-1, self.horizon, self.num)
        a = a.view(-1, self.horizon, self.num+1)
        e = th.cat([sp, a], 2)
        
        e = e.permute(0,2,1)
        c1 = self.relu(self.cnn1(e))
        c2 = self.relu(self.cnn2(c1))
        c2 = self.relu(self.cnn3(c2))
        
        c2 = c2.view(-1, self.horizon*128)
        e1 = self.relu(self.dp(self.fc1(c2)))
        e2 = self.relu(self.dp(self.fc2(e1)))
        
        rec = self.sigmoid(self.fc3(e2))
        return rec
    