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

from torch.nn.modules.normalization import LayerNorm

#Utils
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, args, causal = False):
        super(MultiHeadedSelfAttention,self).__init__()
        self.proj_q = nn.Linear(args.hidden_size, args.hidden_size)
        self.proj_k = nn.Linear(args.hidden_size, args.hidden_size)
        self.proj_v = nn.Linear(args.hidden_size, args.hidden_size)
        self.drop = nn.Dropout(args.hidden_dropout_prob)
        self.scores = None
        self.n_heads = args.heads
        self.causal = causal
        self.register_buffer("mask", None, persistent=False)


    def get_mask(self, n, device):
        if self.mask is not None and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        return mask


    def forward(self, x, mask):
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (self.split_last(x, (self.n_heads, -1)).transpose(1, 2) for x in [q, k, v])
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))

        if self.causal:
            #Causal mask
            seq_len, device = scores.size(-1), scores.device
            causal_mask = self.get_mask(seq_len, device)
            #print('Scores shape : ',scores.shape)
            scores = scores.masked_fill(causal_mask, -torch.finfo(scores.dtype).max)
            scores = scores - scores.amax(dim=-1, keepdim=True).detach()
            scores = self.drop(F.softmax(scores, dim=-1))

            h = (scores @ v).transpose(1, 2).contiguous()
            h = self.merge_last(h, 2)
            self.scores = scores
            return h
        else:
            if mask is not None:
                #print('MASK is not None ', mask, mask.shape)
                mask = mask[:, None, None, :].float()
                #print('Mask added dimension ', mask, mask.shape)
                #print('Scores shape ',scores.shape)
                scores -= 10000.0 * (1.0 - mask)
            scores = self.drop(F.softmax(scores, dim=-1))
            h = (scores @ v).transpose(1, 2).contiguous()
            h = self.merge_last(h, 2)
            self.scores = scores
            return h
    def split_last(self, x, shape):
        shape = list(shape)
        assert shape.count(-1) <= 1
        if -1 in shape:
            shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
        return x.view(*x.size()[:-1], *shape)
    def merge_last(self, x, n_dims):
        s = x.size()
        assert n_dims > 1 and n_dims < len(s)
        return x.view(*s[:-n_dims], -1)



class PositionWiseFeedForward(nn.Module):
    def __init__(self,args):
        super(PositionWiseFeedForward,self).__init__()
        self.fc1 = nn.Linear(args.hidden_size, args.hidden_size*4)
        self.fc2 = nn.Linear(args.hidden_size*4, args.hidden_size)
    def forward(self, x):
        return self.fc2(gelu(self.fc1(x)))


class TransformerBlockBERT(nn.Module):
    def __init__(self, args, share='all', norm='pre', causal=False):
        super(TransformerBlockBERT, self).__init__()
        
        #self.attn        = Attention(d_model=d_model, n_head=n_head, d_head=(d_model//n_head), n_ctx=1024, bias=True, scale=False)
        self.attention = MultiHeadedSelfAttention(args, causal=causal)

        self.feedforward = PositionWiseFeedForward(args)
        self.norm1        = LayerNorm(args.hidden_size, eps=1e-12)
        self.norm2        = LayerNorm(args.hidden_size, eps=1e-12)
        self.share = share
        self.norm_pos = norm
        self.drop1 = nn.Dropout(args.hidden_dropout_prob)
        self.drop2 = nn.Dropout(args.hidden_dropout_prob)
        self.proj = nn.Linear(args.hidden_size, args.hidden_size)
                
    def forward(self, x, attention_mask=None):
        if self.norm_pos == 'pre':
            hs = self.proj(self.attention(self.norm1(x), attention_mask))
            out = x + self.drop1(hs)

            hs = self.feedforward(self.norm2(out))
            out = out + self.drop2(hs)
        return out

