import math
from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .module_util import SinusoidalPosEmb, LayerNorm
from einops import rearrange
def group_norm(channels):
    return nn.GroupNorm(32, channels)

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out
    

class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

class CrossAttentionModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.query = nn.Conv2d(channels, channels, 1)
        self.key = nn.Conv2d(channels, channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.scale = math.sqrt(channels)

    def forward(self, q_in, kv_in):
        B, C, H, W = q_in.shape
        q = self.query(q_in).flatten(2).transpose(1, 2)       # B x HW x C
        k = self.key(kv_in).flatten(2)                        # B x C x HW
        v = self.value(kv_in).flatten(2).transpose(1, 2)      # B x HW x C

        attn = torch.softmax((q @ k) / self.scale, dim=-1)    # B x HW x HW
        out = (attn @ v).transpose(1, 2).view(B, C, H, W)
        return out + q_in  


class SimpleGate(nn.Module):
    def forward(self, x):
        if x.shape[1] % 2 != 0:
            raise ValueError(f"[SimpleGate] Channel dimension must be divisible by 2, got {x.shape[1]}")
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
    
class TIM(nn.Module):
    def __init__(self,
                 in_channels=8,
                 model_channels=64,     
                 ):
        super(TIM,self).__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        time_dim = model_channels * 4
        fourier_dim = model_channels
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(fourier_dim),
            nn.Linear(fourier_dim, time_dim*2),
            SimpleGate(),
            nn.Linear(time_dim, time_dim)
        )        

        self.cross_attn_1 = CrossAttentionModule(int(in_channels/2))
        self.cross_attn_2 = CrossAttentionModule(int(in_channels/2))


        self.inBlock = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        
        self.middleBlock1 = nn.Sequential(
            group_norm(model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, model_channels, kernel_size=3, padding=1),
        )

        self.CBMA1=CBAM(channel=model_channels)
        
        self.middleBlock2 = nn.Sequential(
            group_norm(model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, model_channels, kernel_size=3, padding=1),
        )
        self.middleBlock3 = nn.Sequential(
            group_norm(model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, model_channels, kernel_size=3, padding=1),
        )


        self.CBMA2=CBAM(channel=model_channels)
        self.CBMA3=CBAM(channel=model_channels)

        self.outBlock1_v = nn.Sequential(  # for out1
            group_norm(model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, int(in_channels//2), kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.outBlock1_i = nn.Sequential(  # for out1
            group_norm(model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, int(in_channels/2), kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, h1,h2,time):

        if isinstance(time, int) or isinstance(time, float):
            time = torch.tensor([time]).to(h1.device)
        time_emb=self.time_mlp(time)    
        if time_emb.shape[1] % 4 != 0:
            raise ValueError(f"time_emb channel must be divisible by 4, got {time_emb.shape[1]}")
        time_emb = rearrange(time_emb, 'b c -> b c 1 1')
        shift_1, scale_1, shift_2, scale_2 = time_emb.chunk(4, dim=1)

        h1_cross=self.cross_attn_1(h1,h2)
        h2_cross=self.cross_attn_2(h2,h1) 
        h=torch.cat([h1_cross,h2_cross],  dim=1)   
        middle1=self.inBlock(h) 
        middle2=self.middleBlock1(middle1)

        m22=self.CBMA1(middle2)

        middle3=self.middleBlock2(m22)
        middle3 = middle3 * (scale_1 + 1) + shift_1
        middle4=self.middleBlock3(m22)
        middle4 = middle4 * (scale_2 + 1) + shift_2

        m1=self.CBMA2(middle3)
        m2=self.CBMA3(middle4)

        w_v_1=self.outBlock1_v(m1)
        w_i_1 = self.outBlock1_i(m2)

        out1=(w_v_1*h2)+(w_i_1*h1)
        return out1