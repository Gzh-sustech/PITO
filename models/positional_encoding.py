import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
import numpy as np

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, min_freq=1/64, scale=1.):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.min_freq = min_freq
        self.scale = scale
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, coordinates, device):
        t = coordinates.to(device).type_as(self.inv_freq)
        t = t * (self.scale / self.min_freq)
        freqs = torch.einsum('... i , j -> ... i j', t, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)


def apply_rotary_pos_emb(t, freqs):
    freqs = freqs.unsqueeze(1)
    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())


def apply_2d_rotary_pos_emb(t, freqs_x, freqs_y):

    d = t.shape[-1]
    t_x, t_y = t[..., :d//2], t[..., d//2:]

    return torch.cat((apply_rotary_pos_emb(t_x, freqs_x),
                      apply_rotary_pos_emb(t_y, freqs_y)), dim=-1)

def apply_3d_rotary_pos_emb(t, freqs_x, freqs_y, freqs_z):

    d = t.shape[-1]
    t_x, t_y, t_z = t[..., :d//3], t[..., d//3:2*d//3], t[..., 2*d//3:]

    return torch.cat((apply_rotary_pos_emb(t_x, freqs_x),
                      apply_rotary_pos_emb(t_y, freqs_y),
                      apply_rotary_pos_emb(t_z, freqs_z)), dim=-1)
