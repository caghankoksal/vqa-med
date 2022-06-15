import clip
import torch
from .clip_model import VisionTransformer
from torch import nn as nn
import pytorch_lightning as pl
from .flamingo_palm_original import FlamingoPaLM

class FlamingoClipPalm(pl.LightningModule):
    def __init__(self, pretrained_clip_path):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, _  = clip.load("ViT-B/32",device=device)
        model.load_state_dict(torch.load(pretrained_clip_path, map_location=device)['state_dict'])
        self.vit_clip  = VisionTransformer(input_resolution=224, patch_size=32, width=768, layers=12, heads=8,output_dim=512)
        self.vit_clip.load_state_dict(model.visual.state_dict())
        
        self.flamingo_palm = FlamingoPaLM(
                                        num_tokens = 31092,          # number of tokens
                                        dim = 512,                  # dimensions
                                        depth = 12,                  # depth
                                        heads = 8,                   # attention heads
                                        dim_head = 64,               # dimension per attention head
                                        img_encoder = self.vit_clip,           # plugin your image encoder (this can be optional if you pass in the image embeddings separately, but probably want to train end to end given the perceiver resampler)
                                        media_token_id = 3,          # the token id representing the [media] or [image]
                                        cross_attn_every = 3,        # how often to cross attend
                                        perceiver_num_latents = 64,  # perceiver number of latents, should be smaller than the sequence length of the image tokens
                                        perceiver_depth = 2          # perceiver resampler depth
                                    )

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        images = x['image']
        input_tokens = x['input_ids']
        flamingo_logits = self.flamingo_palm(input_tokens.squeeze(1), images.unsqueeze(1))
        return flamingo_logits

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        images = batch['image']
        input_tokens = batch['input_ids']
        targets = batch["targets"]
        flamingo_logits = self.flamingo_palm(input_tokens.squeeze(1), images.unsqueeze(1))
        loss = nn.CrossEntropyLoss()(torch.permute(flamingo_logits, (0,2,1)), targets.squeeze(1))
        
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # val defined the training loop.
        # It is independent of forward
        images = batch['image']
        input_tokens = batch['input_ids']
        targets = batch["targets"]
        flamingo_logits = self.flamingo_palm(input_tokens.squeeze(1), images.unsqueeze(1))
        loss = nn.CrossEntropyLoss()(torch.permute(flamingo_logits, (0,2,1)), targets.squeeze(1))
        # Logging to TensorBoard by default
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer