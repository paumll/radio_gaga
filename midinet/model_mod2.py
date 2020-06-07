import numpy as np 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import ipdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ops import *


MODEL_NAME = 'modelo2_emb'

class sample_generator(nn.Module):
    def __init__(self, pitch_range):
        super(sample_generator, self).__init__()
        self.gf_dim   = 64
        self.y_dim   = 100
        self.n_channel = 256

        self.h1      = nn.ConvTranspose2d(in_channels=244, out_channels=pitch_range, kernel_size=(2,1), stride=(2,2))
        self.h2      = nn.ConvTranspose2d(in_channels=244, out_channels=pitch_range, kernel_size=(2,1), stride=(2,2))
        self.h3      = nn.ConvTranspose2d(in_channels=244, out_channels=pitch_range, kernel_size=(2,1), stride=(2,2))
        self.h4      = nn.ConvTranspose2d(in_channels=244, out_channels=1, kernel_size=(1,pitch_range), stride=(1,2))

        self.h0_prev = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1,pitch_range), stride=(1,2))
        self.h1_prev = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2,1), stride=(2,2))
        self.h2_prev = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2,1), stride=(2,2))
        self.h3_prev = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2,1), stride=(2,2))

        self.linear1 = nn.Linear(200,1024)
        self.linear2 = nn.Linear(1124,self.gf_dim*2*2*1)

        self.embedding = nn.Embedding(25, 100)

    def forward(self, z, prev_x, y ,batch_size,pitch_range):

        h0_prev = lrelu(batch_norm_2d(self.h0_prev(prev_x)),0.2)   #[72, 16, 16, 1]
        h1_prev = lrelu(batch_norm_2d(self.h1_prev(h0_prev)),0.2)  #[72, 16, 8, 1]
        h2_prev = lrelu(batch_norm_2d(self.h2_prev(h1_prev)),0.2)  #[72, 16, 4, 1]
        h3_prev = lrelu(batch_norm_2d(self.h3_prev(h2_prev)),0.2)  #[72, 16, 2, 1])

        y = self.embedding(y)
        y = y.view(batch_size, -1)

        yb = y.view(batch_size,  self.y_dim, 1, 1)  #(72,13,1,1)
        z = torch.cat((z,y),1)         #(72,113)

        h0 = F.relu(batch_norm_1d(self.linear1(z)))    #(72,1024)
        h0 = torch.cat((h0,y),1)   #(72,1037) 1037 -> 1124

        h1 = F.relu(batch_norm_1d(self.linear2(h0)))   #(72, 256)
        h1 = h1.view(batch_size, self.gf_dim * 2, 2, 1)     #(72,128,2,1)
        h1 = conv_cond_concat(h1,yb) #(b,141,2,1) # 141->228
        h1 = conv_prev_concat(h1,h3_prev)  #(72, 157, 2, 1) 157->228+16=244

        h2 = F.relu(batch_norm_2d(self.h1(h1)))  #(72, 128, 4, 1)
        h2 = conv_cond_concat(h2,yb) #([72, 228, 4, 1])
        h2 = conv_prev_concat(h2,h2_prev)  #([72, 244, 4, 1])

        h3 = F.relu(batch_norm_2d(self.h2(h2)))  #([72, 128, 8, 1]) 
        h3 = conv_cond_concat(h3,yb)  #([72, 228, 8, 1])
        h3 = conv_prev_concat(h3,h1_prev) #([72, 244, 8, 1])

        h4 = F.relu(batch_norm_2d(self.h3(h3)))  #([72, 128, 16, 1])
        h4 = conv_cond_concat(h4,yb)  #([72, 228, 16, 1])
        h4 = conv_prev_concat(h4,h0_prev) #([72, 244, 16, 1])

        g_x = torch.sigmoid(self.h4(h4)) #([72, 1, 16, 128])

        return g_x



class generator(nn.Module):
    def __init__(self,pitch_range):
        super(generator, self).__init__()
        self.gf_dim   = 64
        self.y_dim   = 100
        self.n_channel = 256

        self.h1      = nn.ConvTranspose2d(in_channels=244, out_channels=pitch_range, kernel_size=(2,1), stride=(2,2))
        self.h2      = nn.ConvTranspose2d(in_channels=244, out_channels=pitch_range, kernel_size=(2,1), stride=(2,2))
        self.h3      = nn.ConvTranspose2d(in_channels=244, out_channels=pitch_range, kernel_size=(2,1), stride=(2,2))
        self.h4      = nn.ConvTranspose2d(in_channels=244, out_channels=1, kernel_size=(1,pitch_range), stride=(1,2))

        self.h0_prev = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1,pitch_range), stride=(1,2))
        self.h1_prev = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2,1), stride=(2,2))
        self.h2_prev = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2,1), stride=(2,2))
        self.h3_prev = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2,1), stride=(2,2))

        self.linear1 = nn.Linear(200,1024)
        self.linear2 = nn.Linear(1124,self.gf_dim*2*2*1)

        self.embedding = nn.Embedding(25, 100)

    def forward(self, z, prev_x, y ,batch_size,pitch_range):

        h0_prev = lrelu(batch_norm_2d(self.h0_prev(prev_x)),0.2)   #[72, 16, 16, 1]
        h1_prev = lrelu(batch_norm_2d(self.h1_prev(h0_prev)),0.2)  #[72, 16, 8, 1]
        h2_prev = lrelu(batch_norm_2d(self.h2_prev(h1_prev)),0.2)  #[72, 16, 4, 1]
        h3_prev = lrelu(batch_norm_2d(self.h3_prev(h2_prev)),0.2)  #[72, 16, 2, 1])

        y = self.embedding(y)
        y = y.view(batch_size, -1)


        yb = y.view(batch_size,  self.y_dim, 1, 1)  #(72,13,1,1)
        z = torch.cat((z,y),1)         #(72,113)

        h0 = F.relu(batch_norm_1d(self.linear1(z)))    #(72,1024)
        h0 = torch.cat((h0,y),1)   #(72,1037) 1037 -> 1124

        h1 = F.relu(batch_norm_1d(self.linear2(h0)))   #(72, 256)
        h1 = h1.view(batch_size, self.gf_dim * 2, 2, 1)     #(72,128,2,1)
        h1 = conv_cond_concat(h1,yb) #(b,141,2,1) # 141->228
        h1 = conv_prev_concat(h1,h3_prev)  #(72, 157, 2, 1) 157->228+16=244

        h2 = F.relu(batch_norm_2d(self.h1(h1)))  #(72, 128, 4, 1)
        h2 = conv_cond_concat(h2,yb) #([72, 228, 4, 1])
        h2 = conv_prev_concat(h2,h2_prev)  #([72, 244, 4, 1])

        h3 = F.relu(batch_norm_2d(self.h2(h2)))  #([72, 128, 8, 1]) 
        h3 = conv_cond_concat(h3,yb)  #([72, 228, 8, 1])
        h3 = conv_prev_concat(h3,h1_prev) #([72, 244, 8, 1])

        h4 = F.relu(batch_norm_2d(self.h3(h3)))  #([72, 128, 16, 1])
        h4 = conv_cond_concat(h4,yb)  #([72, 228, 16, 1])
        h4 = conv_prev_concat(h4,h0_prev) #([72, 244, 16, 1])

        g_x = torch.sigmoid(self.h4(h4)) #([72, 1, 16, 128])

        return g_x


class discriminator(nn.Module):
    def __init__(self,pitch_range):
        super(discriminator, self).__init__()

        self.df_dim = 64
        self.dfc_dim = 1024
        self.y_dim = 100

        self.h0_prev = nn.Conv2d(in_channels=101, out_channels=101, kernel_size=(4,pitch_range), stride=(1,1)) #con acordes
        # out channels = y_dim +1 
        self.h1_prev = nn.Conv2d(in_channels=101, out_channels=120, kernel_size=(4,1), stride=(1,1)) #sin acordes
        # out channels = df_dim + y_dim
        self.h2_prev = nn.Conv2d(in_channels=220, out_channels=250, kernel_size=(4,1), stride=(1,1)) #con acordes

        self.h3_prev = nn.Conv2d(in_channels=250, out_channels=270, kernel_size=(4,1), stride=(1,1)) #sin acordes
        
        self.linear1 = nn.Linear(1080,self.dfc_dim)
        self.linear2 = nn.Linear(1124,1)

        self.embedding = nn.Embedding(25,100)
        # w' = (w-k+2p)/s + 1
        # w = s(w'-1) + k - 2p
    def forward(self,x,y,batch_size,pitch_range):        
        y = self.embedding(y)
        y = y.view(batch_size, -1)
        yb = y.view(batch_size,self.y_dim, 1, 1)
        x = conv_cond_concat(x, yb)  #x.shape torch.Size([72, 14, 16, 128]) -> 72, 101, 16, 128
        h0 = lrelu(self.h0_prev(x),0.2) # h0 shape: [72, 14, 13, 1] -> 72, 101, 13, 1
        fm = h0
        #h0 = conv_cond_concat(h0, yb) 

        h1 = lrelu(batch_norm_2d(self.h1_prev(h0)),0.2)  #torch.Size([72, 120, 10, 1])
        h1 = conv_cond_concat(h1,yb)  #torch.Size([72, 220, 10, 1])

        h2 = lrelu(batch_norm_2d(self.h2_prev(h1)), 0.2) # [72, 250, 7, 1]

        h3 = lrelu(batch_norm_2d(self.h3_prev(h2)),0.2) # [72, 270, 4, 1]
        h3 = h3.view(batch_size, -1)
        #h3 = torch.cat((h3,y),1)
        
        h4 = lrelu(batch_norm_1d(self.linear1(h3)), 0.2)
        h4 = torch.cat((h4,y),1)
        h5 = self.linear2(h4)

        h5_sigmoid = torch.sigmoid(h5)

        return h5_sigmoid, h5, fm

