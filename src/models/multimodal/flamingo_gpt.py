import torch
import torch.nn.functional as F
from einops import rearrange
from torch import einsum, nn
from ..text.gpt2_layers import TransformerBlockGPT2

from .flamingo_pytorch_original import GatedCrossAttentionBlock, PerceiverResampler
# Owner: Lucidrains -> https://github.com/lucidrains/flamingo-pytorch  #I will hack and update necessary parts for my use case
# helper functions

def exists(val):
    return val is not None

# for controlling freezing during training of flamingo

def set_module_requires_grad_(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad

def freeze_all_layers_(module):
    set_module_requires_grad_(module, False)

def unfreeze_all_layers_(module):
    set_module_requires_grad_(module, True)

def freeze_model_and_make_eval_(model):
    model.eval()
    freeze_all_layers_(model)

# normalization
# they use layernorm without bias, something that pytorch does not offer


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# residual


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


# rotary positional embedding
# https://arxiv.org/abs/2104.09864


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len, *, device):
        seq = torch.arange(max_seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = einsum("i , j -> i j", seq, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())


# classic Noam Shazeer paper, except here they use SwiGLU instead of the more popular GEGLU for gating the feedforward
# https://arxiv.org/abs/2002.05202


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x



class FlamingoGPT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        num_tokens,
        depth,
        dim_head=64,
        heads=8,
        ff_mult=4,
        media_token_id=3,
        cross_attn_every=3,
        img_encoder=None,
        perceiver_num_latents=64,
        perceiver_depth=2
    ):
        super().__init__()

        self.token_emb = nn.Embedding(num_tokens, dim)
        self.media_token_id = media_token_id # you need to reserve a special token id for media

        self.img_encoder = img_encoder
        freeze_model_and_make_eval_(self.img_encoder)

        self.perceiver_resampler = PerceiverResampler(
            dim=dim,
            depth=perceiver_depth,
            dim_head=dim_head,
            heads=heads,
            num_latents=perceiver_num_latents
        )

        self.layers = nn.ModuleList([])
        for ind in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(TransformerBlockGPT2(d_model=dim, n_head=depth, dropout=0.1)),
                GatedCrossAttentionBlock(dim=dim, dim_head=dim_head, heads=heads) if not (ind % cross_attn_every) else None
            ]))

        self.to_logits = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, num_tokens, bias=False)
        )

        # they used embedding weight tied projection out to logits, not common, but works
        self.to_logits[-1].weight = self.token_emb.weight
        nn.init.normal_(self.token_emb.weight, std=0.02)
    
    def forward(
        self,
        text,
        images=None,
        image_embeds=None
    ):
        batch, device = text.shape[0], text.device

        flamingo_mode = exists(images) or exists(image_embeds)

        # automatically take care of freezing or unfreezing depending on what is passed in

        if flamingo_mode:
            # in flamingo mode, freeze everything but perceiver and gated cross attention
            freeze_all_layers_(self)
            unfreeze_all_layers_(self.perceiver_resampler)
            [unfreeze_all_layers_(cross_attn) for _, cross_attn in self.layers if exists(cross_attn)]
        else:
            unfreeze_all_layers_(self)

        # derive the media token ids (as a boolean tensor), for calculating the masked cross attention

        if flamingo_mode:
            media_locations = text == self.media_token_id

        text_tokens = self.token_emb(text)

        assert not (exists(images) and exists(image_embeds))

        # encode images into embeddings
        # with the img_encoder passed in at init
        # it can also accept precomputed image embeddings

        if exists(images):
            assert exists(self.img_encoder), 'img_encoder must be passed in for automatic image encoding'
            images = rearrange(images, 'b t ... -> (b t) ...')

            with torch.no_grad():
                image_embeds = self.img_encoder(images)

            image_embeds = rearrange(image_embeds, '(b t) ... -> b t ...', b = batch)

        if exists(image_embeds):
            image_embeds = self.perceiver_resampler(image_embeds)

        # go through layers

        for attn_ff, flamingo_cross_attn in self.layers:
            text_tokens = attn_ff(text_tokens)

            # if image embeds exist and flamingo cross attention set for the layer
            # do the cross attention
            if exists(flamingo_cross_attn) and exists(image_embeds):
                text_tokens = flamingo_cross_attn(
                    text_tokens,
                    image_embeds,
                    media_locations = media_locations
                )

        return self.to_logits(text_tokens)
