
import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from .positional_encoding import *

@torch.jit.script
def compl_mul3d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    res = torch.einsum("bixyzt,ioxyzt->boxyzt", a, b)
    return res

class SpectralConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3, modes4):
        super(SpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.modes4 = modes4
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4,
                                    dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4,
                                    dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4,
                                    dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4,
                                    dtype=torch.cfloat))
        self.weights5 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4,
                                    dtype=torch.cfloat))
        self.weights6 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4,
                                    dtype=torch.cfloat))
        self.weights7 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4,
                                    dtype=torch.cfloat))
        self.weights8 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4,
                                    dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[ -4, -3, -2,-1])
        t_dim = self.modes4
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x_ft.shape[2], x_ft.shape[3], x_ft.shape[4], t_dim,
                             device=x.device, dtype=torch.cfloat)
        # if x_ft.shape[4] > self.modes3, truncate; if x_ft.shape[4] < self.modes3, add zero padding
        # +++
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3, :] = compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3,:t_dim], self.weights1)
        # -++
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3, :] = compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3,:t_dim], self.weights2)
        # +-+
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3, :] = compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3,:t_dim], self.weights3)
        # --+
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3, :] = compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3,:t_dim], self.weights4)
        # -+-
        out_ft[:, :, -self.modes1:, :self.modes2, -self.modes3:, :] = compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, -self.modes3:,:t_dim], self.weights5)
        # ++-
        out_ft[:, :, :self.modes1, :self.modes2, -self.modes3:, :] = compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, -self.modes3:, :t_dim], self.weights6)
        # +--
        out_ft[:, :, :self.modes1, -self.modes2:, -self.modes3:, :] = compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, -self.modes3:, :t_dim], self.weights7)
        # ---
        out_ft[:, :, -self.modes1:, -self.modes2:, -self.modes3:, :] = compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, -self.modes3:,:t_dim], self.weights8)
        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(2), x.size(3), x.size(4), x.size(5)), dim=[2, 3, 4, 5])
        return x

# ---------------------------------------VIT------------------------------------------


def pair(t):
    return t if isinstance(t, list) else [t, t, t]

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 17, dim_head = 60, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
        assert dim_head % 3 == 0 and dim_head % 2 == 0, "dim_head must be divisible by 6"
        self.rotary_emb_x = RotaryEmbedding(dim_head // 3)
        self.rotary_emb_y = RotaryEmbedding(dim_head // 3)
        self.rotary_emb_z = RotaryEmbedding(dim_head // 3)

    def forward(self, x,coords):
        qkv = self.to_qkv(x).chunk(3, dim = -1)## 对tensor张量分块 x :1 197 1024   qkv 最后是一个元祖，tuple，长度是3，每个元素形状：1 197 1024
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        coords_x, coords_y, coords_z = coords[..., 0], coords[..., 1], coords[..., 2]
        freqs_x = self.rotary_emb_x(coords_x, device=x.device)
        freqs_y = self.rotary_emb_y(coords_y, device=x.device)
        freqs_z = self.rotary_emb_z(coords_z, device=x.device)

        q = apply_3d_rotary_pos_emb(q, freqs_x, freqs_y, freqs_z)
        k = apply_3d_rotary_pos_emb(k, freqs_x, freqs_y, freqs_z)

        # 计算注意力
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x,coords):
        for attn, ff in self.layers:
            x = attn(x,coords=coords) + x
            x = ff(x) + x
        return x

class ViT3D(nn.Module):
    def __init__(self, *,patch_size,
                 dim, depth, heads, mlp_dim, channels,
                 dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()

        # 三维尺寸处理

        self.pd, self.ph, self.pw = pair(patch_size)

        self.patch_dim = channels * self.pd * self.ph * self.pw

        # 编码器
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (d pd) (h ph) (w pw) -> b (d h w) (pd ph pw c)', # 这里的d h w是指num，和上述的不一致。
                      pd=self.pd, ph=self.ph, pw=self.pw),
            nn.Linear(self.patch_dim, dim),
        )

        # 位置编码
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        # self.dropout = nn.Dropout(emb_dropout)

        # Transformer
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # 解码器
        self.decoder_fc = nn.Linear(dim, self.pd * self.ph * self.pw * channels)

    def forward(self, x):
        x = rearrange(x, 'b nx ny nz c -> b c nx ny nz')
        b, c, d, h, w = x.shape

        # 动态计算分块数量
        num_d = d // self.pd
        num_h = h // self.ph
        num_w = w // self.pw
        # 编码
        x = self.to_patch_embedding(x)

        # 添加位置信息
        grid_d, grid_h, grid_w = torch.meshgrid(
            torch.arange(num_d, device=x.device),
            torch.arange(num_h, device=x.device),
            torch.arange(num_w, device=x.device),
            indexing='ij')
        # x += self.pos_embedding
        # x = self.dropout(x)
        coordinates = torch.stack([grid_d, grid_h, grid_w], dim=-1)  # [num_d, num_h, num_w, 3]
        coordinates = coordinates.reshape(-1, 3)  # [num_d*num_h*num_w, 3]
        coordinates = coordinates.unsqueeze(0).expand(b, -1, -1)  # [b, num_patches, 3]
        # Transformer处理
        x = self.transformer(x,coordinates)

        # 去除cls_token
        x = self.decoder(x,num_d,num_h,num_w)
        x = rearrange(x, 'b c nx ny nz -> b nx ny nz c')
        # 解码
        return x
    def decoder(self, x, num_d, num_h, num_w):
        x = self.decoder_fc(x)
        x = rearrange(x,
                      'b (d h w) (pd ph pw c) -> b c (d pd) (h ph) (w pw)',
                      d=num_d, h=num_h, w=num_w,
                      pd=self.pd, ph=self.ph, pw=self.pw)
        return x
class LatentBlock(nn.Module):
    def __init__(self, patch_dim, dim, depth, heads, mlp_dim, dim_head, dropout=0.):
        super().__init__()
        self.fc_in = nn.Linear(patch_dim, dim)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.fc_out = nn.Linear(dim, patch_dim)

    def forward(self, x, coordinates):
        h = self.fc_in(x)
        h = self.transformer(h, coordinates)
        out = self.fc_out(h)
        return out
