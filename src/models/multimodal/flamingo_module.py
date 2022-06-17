import clip
import torch
from .clip_model import VisionTransformer
from torch import nn as nn
import pytorch_lightning as pl
from .flamingo_palm_original import FlamingoPaLM
from .flamingo_model import FlamingoModel
from transformers import get_linear_schedule_with_warmup,get_constant_schedule_with_warmup
from vit_pytorch.vit import ViT
from vit_pytorch.extractor import Extractor
import torchxrayvision as xrv


class FlamingoModule(pl.LightningModule):
    def __init__(self, pretrained_clip_path, total_steps, num_tokens = 31092, dim=512,
                 depth=12, heads=8, dim_head=64, media_token_id=3190, cross_attn_every=3,
                 perceiver_num_latents = 64, perceiver_depth = 2, image_encoder ="clip", 
                 language_model='gpt2', pretrained_gpt2_path=None ):

        super().__init__()
        print("FlamingoModule will be initialized with parameters : ",
                "Pretrained Clip Path : ", pretrained_clip_path, " \n",
                "Total Steps : ", total_steps, " \n",
                "Num Tokens : ", num_tokens, " \n",
                "Dim : ", dim, " \n",
                "Depth : ", depth, " \n",
                "Heads : ", heads, " \n",
                "Dim Head : ", dim_head, " \n",
                "Media Token Id : ", media_token_id, " \n",
                "Cross Attn Every : ", cross_attn_every, " \n",
                "Perceiver Num Latents : ", perceiver_num_latents, " \n",
                "Perceiver Depth : ", perceiver_depth, " \n",
                "Image Encoder : ", image_encoder, " \n",
                "Language Model : ", language_model, " \n",
                "Pretrained GPT2 Path : ", pretrained_gpt2_path, " \n"
            )
        self.total_steps = total_steps
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if image_encoder == "clip" and pretrained_clip_path != None:
            print("Pretrained clip is being loaded")
            model, _  = clip.load("ViT-B/32",device=device)
            model.load_state_dict(torch.load(pretrained_clip_path, map_location=device)['state_dict'])
            image_encoder  = VisionTransformer(input_resolution=224, patch_size=32, width=768, layers=12, heads=8, output_dim=512)
            image_encoder.load_state_dict(model.visual.state_dict())
            self.img_encoder_outdim = 512

        elif image_encoder == "clip" and pretrained_clip_path == None:
            print("Vit is started from scratch")
            vit = ViT(image_size = 224, patch_size = 32, num_classes = 1000, dim = dim,
                      depth = 6, heads = 16, mlp_dim = 2048, dropout = 0.1, emb_dropout = 0.1
                      )
            print("Vit is initialized")

            image_encoder = Extractor(vit, return_embeddings_only = True)
            self.img_encoder_outdim = dim
        elif image_encoder == "densenet":
            image_encoder = xrv.models.DenseNet(weights="densenet121-res224-mimic_nb")
            self.img_encoder_outdim = None

        

        # It should be better if single Flamingo model is created and used with both GPT2 and Palm
        
        print("Flamingo is being initialized with ", language_model, " as language model")
        self.flamingo_palm = FlamingoModel(
                                        num_tokens = num_tokens,                        # number of tokens
                                        dim = dim,                                      # dimensions
                                        depth = depth,                                  # depth
                                        heads = heads,                                  # attention heads
                                        dim_head = dim_head,                            # dimension per attention head
                                        img_encoder = image_encoder,               # plugin your image encoder (this can be optional if you pass in the image embeddings separately, but probably want to train end to end given the perceiver resampler)
                                        media_token_id = media_token_id,                # the token id representing the [media] or [image]
                                        cross_attn_every = cross_attn_every,            # how often to cross attend
                                        perceiver_num_latents = perceiver_num_latents,  # perceiver number of latents, should be smaller than the sequence length of the image tokens
                                        perceiver_depth = perceiver_depth,              # perceiver resampler depth
                                        language_model=language_model,                  # language model    (gpt2 or palm)
                                        img_encoder_outdim = self.img_encoder_outdim,
                                        pretrained_gpt2_path=pretrained_gpt2_path
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4,weight_decay=0.01)
        """
        
        learning rate is increased linearly from 0 to 10−4 up over the first 5000
        steps then held constant for the duration of training (no improvements were observed from decaying
        the learning rate). Unless specified otherwise we train our models for 500.000 step

        ratio = 500/500.000 = 0.001

        # Number of totals steps : num_epochs * num_batches   

        """

        scheduler = get_constant_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=200,
                )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
