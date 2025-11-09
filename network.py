"""Neural network module for optimal stopping policies.

This module provides PyTorch neural network architectures for learning optimal
stopping policies. The networks take distributional state information (nu) and
optionally discrete state embeddings and time information as inputs, and output
stopping probabilities.

Policy network variants:
- PolicyStateEmbT: Uses state embeddings and time embeddings
- PolicyStateEmb: Uses state embeddings without time information
- PolicyDistT: Uses distribution input with time embeddings
- PolicyDist: Uses distribution input without time information

All networks use ResNet-style fully connected blocks with GroupNorm normalization
and SiLU activation functions. The output is passed through a sigmoid to produce
stopping probabilities in [0, 1].

Utility components:
- timestep_embedding: Creates sinusoidal positional embeddings for time steps
- SiLU: Swish activation function (x * sigmoid(x))
- ResNet_FC: Fully connected ResNet block with residual connections
- zero_module: Utility function to zero out module parameters
"""

import math
import numpy as np

import torch
import torch.nn as nn

from ipdb import set_trace as debug

class PolicyStateEmbT(torch.nn.Module):
    def __init__(self, state_num = 6, nu_dim = 12, out_dim = 1, hidden_dim=64, time_embed_dim=64, num_res_blocks = 3, dtype = torch.float32):
        super(PolicyStateEmbT,self).__init__()

        self.time_embed_dim = time_embed_dim
        hid = hidden_dim
        self.dtype = dtype

        self.embedding = torch.nn.Embedding(num_embeddings=state_num, embedding_dim=hidden_dim, dtype=self.dtype)
        self.t_module = nn.Sequential(
            nn.Linear(self.time_embed_dim, hid),
            SiLU(),
            nn.Linear(hid, hid),
        )
        self.nu_module = nn.Sequential(
            nn.Linear(nu_dim, hid),
            SiLU(),
            nn.Linear(hid, hid),
        )
        self.x_module = ResNet_FC(hidden_dim, hidden_dim, num_res_blocks=num_res_blocks, dtype=self.dtype)
        self.norm1 = nn.GroupNorm(32, hid, dtype=self.dtype)
        self.norm2 = nn.GroupNorm(32, hid, dtype=self.dtype)
        self.out_module = nn.Sequential(
            nn.Linear(hid,hid, dtype=self.dtype),
            SiLU(),
            nn.Linear(hid,out_dim, dtype=self.dtype),
            nn.Sigmoid())

    def forward(self,x, nu, t):
        if len(t.shape)==0:
            t=t[None]
        t_emb = timestep_embedding(t, self.time_embed_dim).to(self.dtype)
        t_out = self.t_module(t_emb)
        state_out = self.embedding(x)
        nu_out = self.nu_module(nu)
        x_out = self.x_module(self.norm1(state_out + nu_out))
        out   = self.out_module(self.norm2(x_out+t_out))
        return out
    
class PolicyStateEmb(torch.nn.Module):
    def __init__(self, state_num = 6, nu_dim = 12, out_dim = 1, hidden_dim=64, time_embed_dim=64, num_res_blocks = 3, dtype = torch.float32):
        super(PolicyStateEmb,self).__init__()

        self.time_embed_dim = time_embed_dim
        hid = hidden_dim
        self.dtype = dtype

        self.embedding = torch.nn.Embedding(num_embeddings=state_num, embedding_dim=hidden_dim, dtype=self.dtype)
        self.nu_module = nn.Sequential(
            nn.Linear(nu_dim, hid),
            SiLU(),
            nn.Linear(hid, hid),
        )
        self.x_module = ResNet_FC(hidden_dim, hidden_dim, num_res_blocks=num_res_blocks, dtype=self.dtype)
        self.norm1 = nn.GroupNorm(32, hid, dtype=self.dtype)
        self.norm2 = nn.GroupNorm(32, hid, dtype=self.dtype)
        self.out_module = nn.Sequential(
            nn.Linear(hid,hid, dtype=self.dtype),
            SiLU(),
            nn.Linear(hid,out_dim, dtype=self.dtype),
            nn.Sigmoid())

    def forward(self,x, nu):
        state_out = self.embedding(x)
        nu_out = self.nu_module(nu)
        x_out = self.x_module(self.norm1(state_out + nu_out))
        out   = self.out_module(self.norm2(x_out))
        return out
    
class PolicyDistT(torch.nn.Module):
    def __init__(self, nu_dim = 12, out_dim = 1, hidden_dim=64, time_embed_dim=64, num_res_blocks = 3, dtype = torch.float32):
        super(PolicyDistT,self).__init__()

        self.time_embed_dim = time_embed_dim
        hid = hidden_dim
        self.dtype = dtype

        self.t_module = nn.Sequential(
            nn.Linear(self.time_embed_dim, hid),
            SiLU(),
            nn.Linear(hid, hid),
        )
        self.nu_module = nn.Sequential(
            nn.Linear(nu_dim, hid),
            SiLU(),
            nn.Linear(hid, hid),
        )
        self.x_module = ResNet_FC(hidden_dim, hidden_dim, num_res_blocks=num_res_blocks, dtype=self.dtype)
        self.norm1 = nn.GroupNorm(32, hid, dtype=self.dtype)
        self.norm2 = nn.GroupNorm(32, hid, dtype=self.dtype)
        self.out_module = nn.Sequential(
            nn.Linear(hid,hid, dtype=self.dtype),
            SiLU(),
            nn.Linear(hid,out_dim, dtype=self.dtype),
            nn.Sigmoid())

    def forward(self, nu, t):
        if len(t.shape)==0:
            t=t[None]
        t_emb = timestep_embedding(t, self.time_embed_dim).to(self.dtype)
        t_out = self.t_module(t_emb)
        nu_out = self.nu_module(nu)
        x_out = self.x_module(self.norm1(nu_out))
        out   = self.out_module(self.norm2(x_out+t_out))
        return out
    
class PolicyDist(torch.nn.Module):
    def __init__(self, nu_dim = 12, out_dim = 1, hidden_dim=64, time_embed_dim=64, num_res_blocks = 3, dtype = torch.float32):
        super(PolicyDist,self).__init__()

        self.time_embed_dim = time_embed_dim
        hid = hidden_dim
        self.dtype = dtype
        self.nu_module = nn.Sequential(
            nn.Linear(nu_dim, hid),
            SiLU(),
            nn.Linear(hid, hid),
        )
        self.x_module = ResNet_FC(hidden_dim, hidden_dim, num_res_blocks=num_res_blocks, dtype=self.dtype)
        self.norm1 = nn.GroupNorm(32, hid, dtype=self.dtype)
        self.norm2 = nn.GroupNorm(32, hid, dtype=self.dtype)
        self.out_module = nn.Sequential(
            nn.Linear(hid,hid, dtype=self.dtype),
            SiLU(),
            nn.Linear(hid,out_dim, dtype=self.dtype),
            nn.Sigmoid())

    def forward(self, nu):
        nu_out = self.nu_module(nu)
        x_out = self.x_module(self.norm1(nu_out))
        out   = self.out_module(self.norm2(x_out))
        return out    




def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class ResNet_FC(nn.Module):
    def __init__(self, data_dim, hidden_dim, num_res_blocks, dtype=torch.float32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dtype = dtype

        self.map=nn.Linear(data_dim, hidden_dim, dtype=self.dtype)
        self.res_blocks = nn.ModuleList(
            [self.build_res_block() for _ in range(num_res_blocks)])
        
        self.norms = nn.ModuleList(
            # [nn.BatchNorm1d(hidden_dim) for _ in range(num_res_blocks)]
            [nn.GroupNorm(32, hidden_dim, dtype=self.dtype) for _ in range(num_res_blocks)]
        )


    def build_linear(self, in_features, out_features):
        linear = nn.Linear(in_features, out_features, dtype=self.dtype)
        return linear

    def build_res_block(self):
        hid = self.hidden_dim
        layers = []
        widths =[hid]*4
        for i in range(len(widths) - 1):
            layers.append(self.build_linear(widths[i], widths[i + 1]))
            layers.append(SiLU())
        return nn.Sequential(*layers)

    def forward(self, x):
        h=self.map(x)
        for res_block, norm in zip(self.res_blocks, self.norms):
            h = (h + res_block(norm(h))) / np.sqrt(2)
        return h
