import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .module_util import SinusoidalPosEmb, LayerNorm

class SimpleGate(nn.Module):
    def forward(self, x):
        if x.shape[1] % 2 != 0:
            raise ValueError(f"[SimpleGate] Channel dimension must be divisible by 2, got {x.shape[1]}")
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    def __init__(self, c, time_emb_dim=None, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        self.mlp = nn.Sequential(
            SimpleGate(), nn.Linear(time_emb_dim // 2, c * 4)
        ) if time_emb_dim else None

        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm(c)
        self.norm2 = LayerNorm(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def time_forward(self, time, mlp):
        time_emb = mlp(time)
        if time_emb.shape[1] % 4 != 0:
            raise ValueError(f"[NAFBlock] time_emb channel must be divisible by 4, got {time_emb.shape[1]}")
        time_emb = rearrange(time_emb, 'b c -> b c 1 1')
        return time_emb.chunk(4, dim=1)

    def forward(self, x):
        inp, time = x
        shift_att, scale_att, shift_ffn, scale_ffn = self.time_forward(time, self.mlp)

        x = inp

        x = self.norm1(x)
        x = x * (scale_att + 1) + shift_att
        x = self.conv1(x).contiguous()
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.norm2(y)
        x = x * (scale_ffn + 1) + shift_ffn
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        x = y + x * self.gamma

        return x, time


class DepthwiseConv1x1(nn.Module):
    """Depthwise conv 3x3 + Pointwise conv 1x1"""
    def __init__(self, dim):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.pw = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        return self.pw(self.dw(x))


class DiM(nn.Module):
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[], heads=4):
        super().__init__()
        self.width = width
        time_dim = width * 4

        fourier_dim = width
        self.vis_time_mlp = nn.Sequential(
            SinusoidalPosEmb(fourier_dim),
            nn.Linear(fourier_dim, time_dim*2),
            SimpleGate(),
            nn.Linear(time_dim, time_dim)
        )

        self.ir_time_mlp = nn.Sequential(
            SinusoidalPosEmb(fourier_dim),
            nn.Linear(fourier_dim, time_dim*2),
            SimpleGate(),
            nn.Linear(time_dim, time_dim)
        )

        # VIS
        self.vis_intro = nn.Conv2d(img_channel * 2, width, 3, padding=1)
        self.vis_ending = nn.Conv2d(width, img_channel, 3, padding=1)
        # IR
        self.ir_intro = nn.Conv2d(img_channel * 2, width, 3, padding=1)
        self.ir_ending = nn.Conv2d(width, img_channel, 3, padding=1)

        self.vis_encoders = nn.ModuleList()
        self.ir_encoders = nn.ModuleList()
        self.vis_decoders = nn.ModuleList()
        self.ir_decoders = nn.ModuleList()
        self.vis_downs = nn.ModuleList()
        self.vis_ups = nn.ModuleList()
        self.ir_downs = nn.ModuleList()
        self.ir_ups = nn.ModuleList()

        chan = width
        for i, num in enumerate(enc_blk_nums):
            vis_encoder = nn.Sequential(*[NAFBlock(chan, time_emb_dim=time_dim) for _ in range(num)])
            ir_encoder = nn.Sequential(*[NAFBlock(chan, time_emb_dim=time_dim) for _ in range(num)])
            self.vis_encoders.append(vis_encoder)
            self.ir_encoders.append(ir_encoder)
            self.vis_downs.append(nn.Conv2d(chan, chan * 2, 2, 2))
            self.ir_downs.append(nn.Conv2d(chan, chan * 2, 2, 2))
            chan *= 2

        self.vis_middle_blks = nn.Sequential(*[NAFBlock(chan, time_emb_dim=time_dim) for _ in range(middle_blk_num)])
        self.ir_middle_blks= nn.Sequential(*[NAFBlock(chan, time_emb_dim=time_dim) for _ in range(middle_blk_num)])

        for i, num in enumerate(dec_blk_nums):
            self.vis_ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1, bias=False),
                nn.PixelShuffle(2)))
            self.ir_ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, 1, bias=False),
                nn.PixelShuffle(2)))
            chan //= 2
            vis_decoder = nn.Sequential(*[NAFBlock(chan, time_emb_dim=time_dim) for _ in range(num)])
            ir_decoder = nn.Sequential(*[NAFBlock(chan, time_emb_dim=time_dim) for _ in range(num)])
            self.vis_decoders.append(vis_decoder)
            self.ir_decoders.append(ir_decoder)

        self.padder_size = 2 ** len(enc_blk_nums)

    def forward(self, vis, vis_cond, ir, ir_cond, time):
        B, C, H, W = vis.shape
        
        if isinstance(time, int) or isinstance(time, float):
            time = torch.tensor([time]).to(vis.device)
        vis_t = self.vis_time_mlp(time)
        ir_t = self.ir_time_mlp(time)

        x_vis = F.pad(torch.cat([vis - vis_cond, vis_cond], dim=1), self._pad(H, W))
        x_ir = F.pad(torch.cat([ir - ir_cond, ir_cond], dim=1), self._pad(H, W))
        
        x_vis = self.vis_intro(x_vis)
        x_ir = self.ir_intro(x_ir)

        vis_feats, ir_feats = [x_vis], [x_ir]

        for i, (vis_enc, ir_enc, vis_down,ir_down) in enumerate(zip(self.vis_encoders, self.ir_encoders, self.vis_downs,self.ir_downs)):
            x_vis, _ = vis_enc([x_vis, vis_t])
            x_ir, _ = ir_enc([x_ir, ir_t])

            vis_feats.append(x_vis)
            ir_feats.append(x_ir)
            x_vis = vis_down(x_vis)
            x_ir = ir_down(x_ir)

        x_vis, _ = self.vis_middle_blks([x_vis, vis_t])
        x_ir, _ = self.ir_middle_blks([x_ir, ir_t])

        for i, (vis_up,ir_up,vis_dec, ir_dec) in enumerate(zip(self.vis_ups,self.ir_ups, self.vis_decoders, self.ir_decoders)):
            x_vis = vis_up(x_vis) + vis_feats.pop()
            x_ir = ir_up(x_ir) + ir_feats.pop()
            
            x_vis, _ = vis_dec([x_vis, vis_t])
            x_ir, _ = ir_dec([x_ir, ir_t])

        out_vis = self.vis_ending(x_vis + vis_feats.pop())[..., :H, :W]
        out_ir = self.ir_ending(x_ir + ir_feats.pop())[..., :H, :W]
        return out_vis, out_ir

    def _pad(self, H, W):
        ph = (self.padder_size - H % self.padder_size) % self.padder_size
        pw = (self.padder_size - W % self.padder_size) % self.padder_size
        return (0, pw, 0, ph)
    
