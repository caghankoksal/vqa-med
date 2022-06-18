
import math
import torch
from torch import nn as nn
import torch.nn.functional as F
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import ModuleList
from torch.nn.modules.normalization import LayerNorm
import numpy as np
import os
from tqdm import tqdm_notebook, trange
import logging
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger()


class Conv1D(nn.Module):
    def __init__(self, nx, nf):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x
        
class FeedForward(nn.Module):
    def __init__(self, dropout, d_model=768, nx=768*4):
        super().__init__()
        self.c_fc    = Conv1D(d_model, nx)
        self.c_proj  = Conv1D(nx, d_model)
        self.act     = F.gelu
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))

class Attention(nn.Module):
    def __init__(self, d_model=768, n_head=12, n_ctx=1024, d_head=64, bias=True, scale=False):
        super().__init__()
        self.n_head  = n_head
        self.d_model = d_model
        self.c_attn  = Conv1D(d_model, d_model*3)
        self.scale   = scale
        self.softmax = nn.Softmax(dim=-1)
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.dropout = nn.Dropout(0.1)
        self.c_proj  = Conv1D(d_model, d_model)

    def get_mask(self, n, device):

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        return mask


        
    def split_heads(self, x):
        "return shape [`batch`, `head`, `sequence`, `features`]"
        new_shape = x.size()[:-1] + (self.n_head, x.size(-1)//self.n_head) 
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3) 
    
    def _attn(self, q, k, v, attn_mask=None):
        scores  = torch.matmul(q, k.transpose(-2, -1))
        if self.scale: 
            scores = scores/math.sqrt(v.size(-1))
        nd, ns  = scores.size(-2), scores.size(-1)
        if attn_mask is not None: 
            scores = scores.masked_fill(attn_mask, -torch.finfo(scores.dtype).max)
            scores = scores - scores.amax(dim=-1, keepdim=True).detach()

        scores  = self.softmax(scores)
        scores  = self.dropout(scores)
        outputs = torch.matmul(scores, v)
        return outputs
    
    def merge_heads(self, x):
        x         = x.permute(0, 2, 1, 3).contiguous()
        new_shape = x.size()[:-2] + (x.size(-2)*x.size(-1),)
        return x.view(*new_shape)
        
    def forward(self, x, mode='train'):
        n, device = x.shape[1], x.device

        x        = self.c_attn(x) #new `x` shape - `[1,3,2304]`


        q, k, v  = x.split(self.d_model, dim=2)
        q, k, v  = self.split_heads(q), self.split_heads(k), self.split_heads(v)
        if mode=='train':
            attn_mask = self.get_mask(n, device)
            out      = self._attn(q, k, v, attn_mask=attn_mask)
        else:
            out      = self._attn(q, k, v)

        out      = self.merge_heads(out)
        out      = self.c_proj(out)
        return out

class TransformerBlockGPT2(nn.Module):
    def __init__(self, d_model=768, n_head=12, dropout=0.1):
        super(TransformerBlockGPT2, self).__init__()
        self.attn        = Attention(d_model=d_model, n_head=n_head, d_head=(d_model//n_head), n_ctx=1024, bias=True, scale=False)
        self.feedforward = FeedForward(dropout=0.1, d_model=d_model, nx=d_model*4)
        self.ln_1        = LayerNorm(d_model)
        self.ln_2        = LayerNorm(d_model)
                
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.feedforward(self.ln_2(x))
        return x