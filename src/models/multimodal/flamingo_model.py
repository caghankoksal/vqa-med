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


# parallel attention and feedforward with residual
# discovered by Wang et al + EleutherAI from GPT-J fame


class ParallelTransformerBlock(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8, ff_mult=4):
        super().__init__()
        self.norm = LayerNorm(dim)

        attn_inner_dim = dim_head * heads
        ff_inner_dim = dim * ff_mult
        self.fused_dims = (attn_inner_dim, dim_head, dim_head, (ff_inner_dim * 2))

        self.heads = heads
        self.scale = dim_head**-0.5
        self.rotary_emb = RotaryEmbedding(dim_head)

        self.fused_attn_ff_proj = nn.Linear(dim, sum(self.fused_dims), bias=False)
        self.attn_out = nn.Linear(attn_inner_dim, dim, bias=False)

        self.ff_out = nn.Sequential(SwiGLU(), nn.Linear(ff_inner_dim, dim, bias=False))

        # for caching causal mask and rotary embeddings

        self.register_buffer("mask", None, persistent=False)
        self.register_buffer("pos_emb", None, persistent=False)

    def get_mask(self, n, device):
        if self.mask is not None and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    def get_rotary_embedding(self, n, device):
        if self.pos_emb is not None and self.pos_emb.shape[-2] >= n:
            return self.pos_emb[:n]

        pos_emb = self.rotary_emb(n, device=device)
        self.register_buffer("pos_emb", pos_emb, persistent=False)
        return pos_emb

    def forward(self, x):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, device, h = x.shape[1], x.device, self.heads

        # pre layernorm

        x = self.norm(x)

        # attention queries, keys, values, and feedforward inner

        q, k, v, ff = self.fused_attn_ff_proj(x).split(self.fused_dims, dim=-1)

        # split heads
        # they use multi-query single-key-value attention, yet another Noam Shazeer paper
        # they found no performance loss past a certain scale, and more efficient decoding obviously
        # https://arxiv.org/abs/1911.02150

        q = rearrange(q, "b n (h d) -> b h n d", h=h)

        # rotary embeddings

        positions = self.get_rotary_embedding(n, device)
        q, k = map(lambda t: apply_rotary_pos_emb(positions, t), (q, k))

        # scale

        q = q * self.scale

        # similarity

        sim = einsum("b h i d, b j d -> b h i j", q, k)

        # causal mask

        causal_mask = self.get_mask(n, device)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # aggregate values

        out = einsum("b h i j, b j d -> b h i d", attn, v)

        # merge heads

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.attn_out(out) + self.ff_out(ff)


# Generic Flamingo Model for both GPT2 and Flamingo
class FlamingoModel(nn.Module):
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
        perceiver_depth=2,
        only_attend_immediate_media=True,
        language_model="palm",
        img_encoder_outdim=512,
        pretrained_gpt2_path=None,
        classification_mode = False,
        flamingo_mode = True,
    ):

        super().__init__()
        self.num_tokens = num_tokens
        self.dim = dim
        self.classification_mode = classification_mode
        self.token_emb = nn.Embedding(num_tokens, dim)
        self.media_token_id = (
            media_token_id  # you need to reserve a special token id for media
        )
        self.img_encoder_outdim = img_encoder_outdim
        self.img_encoder = img_encoder
        self.flamingo_mode = flamingo_mode
        freeze_model_and_make_eval_(self.img_encoder)

        self.perceiver_resampler = PerceiverResampler(
            dim=dim,
            depth=perceiver_depth,
            dim_head=dim_head,
            heads=heads,
            num_latents=perceiver_num_latents,
        )

        self.img_encoder_outdim_layer = None
        if self.img_encoder_outdim != self.dim:
            self.img_encoder_outdim = img_encoder_outdim
            self.img_encoder_outdim_layer = nn.Linear(img_encoder_outdim, self.dim)

        self.layers = nn.ModuleList([])
        for ind in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        # According to parameter, palm or gpt2 transformer blocks are used.
                        Residual(
                            ParallelTransformerBlock(
                                dim=dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult
                            )
                            if language_model == "palm"
                            else TransformerBlockGPT2(
                                d_model=dim, n_head=depth, dropout=0.1
                            )
                        ),
                        GatedCrossAttentionBlock(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            only_attend_immediate_media=only_attend_immediate_media,
                        )
                        if not (ind % cross_attn_every)
                        else None,
                    ]
                )
            )

        self.to_logits = nn.Sequential(
            LayerNorm(dim), nn.Linear(dim, num_tokens, bias=False)
        )

        if language_model == "gpt2" and pretrained_gpt2_path is not None:
            self.load_gpt2_weights(pretrained_gpt2_path)

        # they used embedding weight tied projection out to logits, not common, but works
        self.to_logits[-1].weight = self.token_emb.weight
        nn.init.normal_(self.token_emb.weight, std=0.02)

    def load_gpt2_weights(self, path):
        """
        Load weights from a GPT2 model.
        """
        print("GPT 2 Weights are loading...")
        old_keys = []
        new_keys = []
        model_dict = self.layers.state_dict()
        state_dict = torch.load(path)  # pretrained weights
        for key in state_dict.keys():
            if key.startswith("h"):
                cur_key = key.replace("h.", "")
                cur_key = cur_key.replace("mlp", "feedforward")
                index_point = cur_key.index(".")
                cur_key = (
                    cur_key[: index_point + 1] + "0.fn." + cur_key[index_point + 1 :]
                )

                new_keys.append(cur_key)
                old_keys.append(key)

        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.layers.load_state_dict(model_dict)

        # Load Embedding Weights
        self.token_emb.weight.data[: self.num_tokens - 3] = state_dict["wte.weight"]
        print(
            "Loaded GPT2 weights and Embeddings",
            "num_weights loaded : ",
            len(pretrained_dict.keys()),
        )

        self.train_embedding_layer=True
            # automatically take care of freezing or unfreezing depending on what is passed in
        if self.flamingo_mode:
            # in flamingo mode, freeze everything but perceiver and gated cross attention
            freeze_all_layers_(self)
            unfreeze_all_layers_(self.perceiver_resampler)

            for _, cross_attn in self.layers:
                if exists(cross_attn):
                    unfreeze_all_layers_(cross_attn)

        else:
            unfreeze_all_layers_(self)
        # This is downsample layer which is not used in Flamingo to reduce the dimensionality of the image embedding
        # given by the clip
        if self.img_encoder_outdim != self.dim:
            unfreeze_all_layers_(self.img_encoder_outdim_layer)

        if self.train_embedding_layer:
            unfreeze_all_layers_(self.token_emb)



    def forward(self, text, images=None, image_embeds=None, return_attn=False):
        batch, device = text.shape[0], text.device

        # derive the media token ids (as a boolean tensor), for calculating the masked cross attention

        media_locations = text == self.media_token_id
        text_tokens = self.token_emb(text)

        assert not (exists(images) and exists(image_embeds))

        # encode images into embeddings
        # with the img_encoder passed in at init
        # it can also accept precomputed image embeddings

        if exists(images):
            assert exists(
                self.img_encoder
            ), "img_encoder must be passed in for automatic image encoding"
            images = rearrange(images, "b t ... -> (b t) ...")

            with torch.no_grad():
                if return_attn:
                    image_embeds, attns = self.img_encoder(
                        images, return_attn=return_attn
                    )
                else:
                    image_embeds = self.img_encoder(images)
            if self.img_encoder_outdim_layer is not None:
                image_embeds = self.img_encoder_outdim_layer(image_embeds)

            image_embeds = rearrange(image_embeds, "(b t) ... -> b t ...", b=batch)

        if exists(image_embeds):
            image_embeds = self.perceiver_resampler(image_embeds)

        # go through layers

        for attn_ff, flamingo_cross_attn in self.layers:
            text_tokens = attn_ff(text_tokens)

            # if image embeds exist and flamingo cross attention set for the layer
            # do the cross attention
            if exists(flamingo_cross_attn) and exists(image_embeds):
                text_tokens = flamingo_cross_attn(
                    text_tokens, image_embeds, media_locations=media_locations
                )
        if return_attn:
            return self.to_logits(text_tokens), attns
        else:
            if self.classification_mode:
                return self.to_logits(text_tokens), text_tokens
            else:
                return self.to_logits(text_tokens)
