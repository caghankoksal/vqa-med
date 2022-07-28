from audioop import bias
import clip
import torch
from .clip_model import VisionTransformer
from torch import nn as nn
import pytorch_lightning as pl
from .flamingo_palm_original import FlamingoPaLM
from .flamingo_model import FlamingoModel, LayerNorm
from transformers import (
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)
from vit_pytorch.vit import ViT
from vit_pytorch.extractor import Extractor
import torchxrayvision as xrv
from torch import nn 
from torch import functional as F

class FlamingoModule(pl.LightningModule):
    def __init__(
        self,
        pretrained_clip_path,
        warmup_steps=569,
        num_tokens=31092,
        dim=512,
        depth=12,
        num_heads=8,
        dim_head=64,
        media_token_id=3190,
        cross_attn_every=3,
        perceiver_num_latents=64,
        perceiver_depth=2,
        image_encoder="clip",
        language_model="gpt2",
        pretrained_gpt2_path=None,
        classification_mode=True,
        classification_num_classes = 332,
        flamingo_mode = True,
        label_smoothing = 0.1,
        token_label_smoothing = 0.0,
        learning_rate = 1e-4,
        use_image_embeddings = False,
        train_embedding_layer = True,
        classifier_dropout = 0.5,
        use_positional_embedding = True

    ):

        super().__init__()

        self.save_hyperparameters()
        self.warmup_steps = warmup_steps
        device = "cpu"
        self.classification_mode= classification_mode
        self.num_classification_classes = classification_num_classes
        self.flamingo_mode = flamingo_mode
        self.label_smoothing = label_smoothing
        self.token_label_smoothing = token_label_smoothing
        self.learning_rate = learning_rate
        self.use_image_embeddings = use_image_embeddings
        self.classifier_dropout = classifier_dropout
        self.use_positional_embedding = use_positional_embedding

        if image_encoder == "clip" and pretrained_clip_path is not None:
            print("Clip architecture is being loaded")
            model, _ = clip.load("ViT-B/32", device=device)
            print("Clip pretrained weights are being loaded")
            model.load_state_dict(
                torch.load(pretrained_clip_path, map_location=device)["state_dict"]
            )
            image_encoder = VisionTransformer(
                input_resolution=224,
                patch_size=32,
                width=768,
                layers=12,
                heads=8,
                output_dim=512,
            )
            image_encoder.load_state_dict(model.visual.state_dict())
            self.img_encoder_outdim = 512

        elif image_encoder == "clip" and pretrained_clip_path is None:
            print("Vit is started from scratch")
            image_encoder = VisionTransformer(
                input_resolution=224,
                patch_size=32,
                width=768,
                layers=12,
                heads=8,
                output_dim=512,
            )
            self.img_encoder_outdim = 512
            print("Vit is initialized")

            #self.img_encoder_outdim = dim
        elif image_encoder == "densenet":
            image_encoder = xrv.models.DenseNet(weights="densenet121-res224-mimic_nb")
            self.img_encoder_outdim = None

        # It should be better if single Flamingo model is created and used with both GPT2 and Palm

        print(
            "Flamingo is being initialized with ", language_model, " as language model"
        )
        self.flamingo_palm = FlamingoModel(
            num_tokens=num_tokens,  # number of tokens
            dim=dim,  # dimensions
            depth=depth,  # depth
            heads=num_heads,  # attention heads
            dim_head=dim_head,  # dimension per attention head
            img_encoder=image_encoder,  # plugin your image encoder (this can be optional if you pass in the image embeddings separately, but probably want to train end to end given the perceiver resampler)
            media_token_id=media_token_id,  # the token id representing the [media] or [image]
            cross_attn_every=cross_attn_every,  # how often to cross attend
            perceiver_num_latents=perceiver_num_latents,  # perceiver number of latents, should be smaller than the sequence length of the image tokens
            perceiver_depth=perceiver_depth,  # perceiver resampler depth
            language_model=language_model,  # language model    (gpt2 or palm)
            img_encoder_outdim=self.img_encoder_outdim,
            pretrained_gpt2_path=pretrained_gpt2_path,
            classification_mode = self.classification_mode,
            flamingo_mode = self.flamingo_mode,
            train_embedding_layer=train_embedding_layer,
            use_positional_embedding = self.use_positional_embedding
        )

        if self.classification_mode:
            #self.classifier = nn.Linear(dim,self.num_classification_classes )
            #nn.init.normal_(self.classifier.weight, std=0.02)
            if self.use_image_embeddings:
                self.classifier = nn.Sequential(
                    LayerNorm(2*dim),
                    #nn.Linear(dim, 4096),
                    #nn.ReLU(),
                    nn.Dropout(0.1),
                    #nn.Linear(4096, self.num_classification_classes),
                    nn.Linear(2*dim, self.num_classification_classes),
                )
            else:
                self.classifier = nn.Sequential(
                    LayerNorm(dim),
                    #nn.Linear(dim, 4096),
                    #nn.ReLU(),
                    nn.Dropout(self.classifier_dropout),
                    #nn.Linear(4096, self.num_classification_classes),
                    nn.Linear(dim, self.num_classification_classes),
                )

    def forward(self, x, return_attn=False):
        # in lightning, forward defines the prediction/inference actions
        images = x["image"]
        input_tokens = x["input_ids"]
        
        batch_size = images.shape[0]
        
        if return_attn:
            flamingo_logits, attns = self.flamingo_palm(
                input_tokens.squeeze(1), images.unsqueeze(1), return_attn=return_attn
            )
            return flamingo_logits, attns
        else:
            if self.classification_mode and self.use_image_embeddings:
                index_eoq = x["index_eoq"]
                flamingo_logits, token_embeds, image_embeddings = self.flamingo_palm(
                    input_tokens.squeeze(1), images.unsqueeze(1), return_attn=return_attn, 
                    return_image_embeddings = self.use_image_embeddings
                )

                eoq_embeds = token_embeds[torch.arange(batch_size), index_eoq]
                classification_logits = self.classifier(torch.cat([image_embeddings.squeeze(1), eoq_embeds],dim=1))
                classification_logits = torch.softmax(classification_logits, dim=1)

                return flamingo_logits, classification_logits

            elif self.classification_mode:
                index_eoq = x["index_eoq"]
                flamingo_logits, token_embeds = self.flamingo_palm(
                    input_tokens.squeeze(1), images.unsqueeze(1), return_attn=return_attn
                )

                classification_logits = self.classifier(token_embeds[torch.arange(batch_size), index_eoq])
                classification_logits = torch.softmax(classification_logits, dim=1)

                return flamingo_logits, classification_logits
            else:
                flamingo_logits = self.flamingo_palm(
                    input_tokens.squeeze(1), images.unsqueeze(1), return_attn=return_attn
                )
                return flamingo_logits

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        images = batch["image"]
        input_tokens = batch["input_ids"]
        targets = batch["targets"]
        batch_size = images.shape[0]

        if self.classification_mode and self.use_image_embeddings:
            class_labels = batch["label"]
            index_eoq = batch["index_eoq"]
            flamingo_logits, token_embeds, image_embeddings = self.flamingo_palm(
                    input_tokens.squeeze(1), images.unsqueeze(1), 
                    return_image_embeddings = self.use_image_embeddings
                )

            eoq_embeds = token_embeds[torch.arange(batch_size), index_eoq]
            classification_logits = self.classifier(torch.cat([image_embeddings.squeeze(1), eoq_embeds],dim=1))

        elif self.classification_mode:
            class_labels = batch["label"]
            index_eoq = batch["index_eoq"]
        
            flamingo_logits, token_embeds = self.flamingo_palm(
                input_tokens.squeeze(1), images.unsqueeze(1)
            )
            classification_logits = self.classifier(token_embeds[torch.arange(batch_size), index_eoq])
        else:
            flamingo_logits = self.flamingo_palm(
            input_tokens.squeeze(1), images.unsqueeze(1)
        )

        train_loss = nn.CrossEntropyLoss(reduction="none",label_smoothing=self.token_label_smoothing)(
            torch.permute(flamingo_logits, (0, 2, 1)), targets.squeeze(1)
        )
        # Only non pad tokens are considered in the loss
        train_loss = torch.sum(train_loss * batch["token_type_ids"]) / (
            torch.sum(batch["token_type_ids"]) * batch_size
        )

        self.log(
            "train_loss_generation",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        # Classification Loss
        if self.classification_mode:
            train_classification_loss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)(classification_logits, class_labels)

            self.log(
                "train_classification_loss",
                train_classification_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

            self.log(
                "train_total_loss",
                train_loss + train_classification_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
              # Calculate validation accuracy
            train_acc = (torch.argmax(classification_logits, dim=1) == class_labels).float().mean()
            self.log(
                "train_acc",
                train_acc,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True
            )
            # self.loggers[-1].experiment.log_metrics(comet_logs, step=self.global_step)
            self.logger.experiment.add_scalars(
                "losses", {"train_loss": train_loss + train_classification_loss}, self.global_step
            )

            return {"loss": train_loss + train_classification_loss}

        else:
            self.logger.experiment.add_scalars(
                "losses", {"train_loss": train_loss}, self.global_step
            )
            return {"loss": train_loss}




    def validation_step(self, batch, batch_idx):
        # val defined the training loop.
        # It is independent of forward
        images = batch["image"]
        input_tokens = batch["input_ids"]
        targets = batch["targets"]
        batch_size = images.shape[0]

        if self.classification_mode and self.use_image_embeddings:
            class_labels = batch["label"]
            index_eoq = batch["index_eoq"]
            flamingo_logits, token_embeds, image_embeddings = self.flamingo_palm(
                    input_tokens.squeeze(1), images.unsqueeze(1), 
                    return_image_embeddings = self.use_image_embeddings
                )

            eoq_embeds = token_embeds[torch.arange(batch_size), index_eoq]

            classification_logits = self.classifier(torch.cat([image_embeddings.squeeze(1), eoq_embeds],dim=1))

        elif self.classification_mode:
            class_labels = batch["label"]
            index_eoq = batch["index_eoq"]
            flamingo_logits, token_embeds = self.flamingo_palm(
                input_tokens.squeeze(1), images.unsqueeze(1)
            )
            classification_logits = self.classifier(token_embeds[torch.arange(batch_size), index_eoq])
        else:
            flamingo_logits = self.flamingo_palm(
            input_tokens.squeeze(1), images.unsqueeze(1)
        )



        val_loss = nn.CrossEntropyLoss(reduction="none",label_smoothing=self.token_label_smoothing)(
            torch.permute(flamingo_logits, (0, 2, 1)), targets.squeeze(1)
        )
        val_loss = torch.sum(val_loss * batch["token_type_ids"]) / (
            torch.sum(batch["token_type_ids"]) * batch_size
        )
        # Logging to TensorBoard by default
        self.log(
            "val_loss_generation",
            val_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True
        )
        if self.classification_mode:
                    # Classification Loss
            val_classification_loss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)(classification_logits, class_labels)

            self.log(
                "val_classification_loss",
                val_classification_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True
            )

            self.log(
                "val_total_loss",
                val_loss + val_classification_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            # Calculate validation accuracy
            val_acc = (torch.argmax(classification_logits, dim=1) == class_labels).float().mean()
            self.log(
                "val_acc",
                val_acc,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True
            )

            self.logger.experiment.add_scalars(
                "losses", {"validation_loss": val_loss+ val_classification_loss}, self.global_step
            )
            return {"val_loss": val_loss + val_classification_loss}

        else:
            self.logger.experiment.add_scalars(
                "losses", {"validation_loss": val_loss}, self.global_step
            )
            return {"val_loss": val_loss}


        

    def  predict_step(self, batch):
        # @TODO implement predict step
        images = batch["image"]
        input_tokens = batch["input_ids"]


        batch_size = images.shape[0]

        if self.classification_mode:
            class_labels = batch["label"]
            index_eoq = batch["index_eoq"]
            flamingo_logits, token_embeds = self.flamingo_palm(
                input_tokens.squeeze(1), images.unsqueeze(1)
            )
            #classification_logits = self.classifier(token_embeds[torch.arange(batch_size), index_eoq])
            return flamingo_logits #, classification_logits
        else:
            flamingo_logits = self.flamingo_palm(
            input_tokens.squeeze(1), images.unsqueeze(1)
            )
            return flamingo_logits



    def configure_optimizers(self):
        """
        learning rate is increased linearly from 0 to 10−4 up over the first 5000
        steps then held constant for the duration of training (no improvements were observed from decaying
        the learning rate). Unless specified otherwise we train our models for 500.000 step

        ratio = 500/500.000 = 0.001

        We train 200 Epochs with a batch size of 64.
        Full Dataset:
        Train Set len :   182268
        Validation Set len :  22783
        Test Set len :  22784

        182268/64(Batch Size) = 2847 Steps in one epoch

        2847 * 200(num epochs) =  569400 Steps in total

        # According to Flamingos strategy

        569400* 0.001 = 569 Steps for Learning Rate Warm up

        # Number of totals steps : num_epochs * num_batches

        Returns:
            _type_: _description_
        """
        params = list(self.named_parameters())
    
        grouped_parameters = [
        {"params": [p for n, p in params if n.startswith('flamingo_palm.perceiver_resampler')], 'weight_decay': 0},
        {"params": [p for n, p in params if not n.startswith('flamingo_palm.perceiver_resampler') ],'weight_decay': 0.01 },
        ]
        optimizer = torch.optim.AdamW(grouped_parameters, lr=self.learning_rate)

        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
