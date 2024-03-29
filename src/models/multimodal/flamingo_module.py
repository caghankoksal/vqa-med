import clip
import torch
import pytorch_lightning as pl
import wandb
import torchxrayvision as xrv

from .clip_model import VisionTransformer
from torch import nn as nn
from .flamingo_palm_original import FlamingoPaLM
from .flamingo_model import FlamingoModel, LayerNorm
from transformers import (
    get_linear_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)
from vit_pytorch.vit import ViT
from vit_pytorch.extractor import Extractor
from torch import nn 
from torch import functional as F

class FlamingoModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        image_encoder_type=args['model']['image_encoder_type']
        language_model= args['model']['language_encoder']

        self.save_hyperparameters()
        self.label_smoothing = args['model']['label_smoothing']
        self.token_label_smoothing = args['model']['token_label_smoothing']
        
        self.classifier_dropout = args['model']['classifier_dropout']
        #self.img_encoder_outdim = args['model']['img_encoder_out_dim']
        self.pretrained_clip_path = args['model']['pretrained_clip_path']
        self.classification_mode = args['model']['classification_mode']
        self.use_image_embeddings = args['model']['use_image_embeddings']
        self.num_classification_classes = args['model']['num_classification_classes']
        self.warmup_steps = args['train']['warmup_steps']
        self.learning_rate = args['train']['learning_rate']
        self.flamingo_embed_dim = args['model']['flamingo_embed_dim']
        self.use_generation_loss = args['model']['use_generation_loss']
        self.use_scheduler = args['scheduler']
        self.optimizer = args['optimizer']['name']
        self.weight_decay = args['optimizer']['weight_decay']
        self.token_average_classification = args['model']['token_average_classification']
        self.classify_only_image_features = args['model']['classify_only_image_features']
   
        if image_encoder_type == "clip" and self.pretrained_clip_path is not None:
            print("Clip architecture is being loaded")
            model, _ = clip.load("ViT-B/32", device='cpu')
            print("Clip pretrained weights are being loaded")
            model.load_state_dict(
                torch.load(self.pretrained_clip_path, map_location='cpu')["state_dict"]
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

        elif image_encoder_type == "clip" and self.pretrained_clip_path  is None:
            print("Vit is started from scratch")
            image_encoder = VisionTransformer(
                input_resolution=224,
                patch_size=32,
                width=768,
                layers=12,
                heads=8,
                output_dim=512,
            )
            print("Vit is initialized")

            self.img_encoder_outdim = 512


        elif image_encoder_type == "densenet":
            image_encoder = xrv.models.DenseNet(weights="densenet121-res224-mimic_nb")
            self.img_encoder_outdim = None

        elif image_encoder_type == "efficientnet":
            #from efficientnet_pytorch import EfficientNet
            #image_encoder = EfficientNet.from_name('efficientnet-b0')

            import timm
            image_encoder = timm.create_model('tf_efficientnet_b5', pretrained=True)
            #config = resolve_data_config({}, model=self.model)
            #self.transforms = create_transform(**config)
            image_encoder.classifier = nn.Identity()
            self.img_encoder_outdim = 2048
        

        # It should be better if single Flamingo model is created and used with both GPT2 and Palm

        print(
            "Flamingo is being initialized with ", language_model, " as language model"
        )
        self.flamingo_palm = FlamingoModel(
            num_tokens=args['model']['num_tokens'],  # number of tokens
            dim=args['model']['flamingo_embed_dim'],  # dimensions
            depth=args['model']['depth'],  # depth
            heads=args['model']['num_heads'],  # attention heads
            dim_head=args['model']['att_head_dim'],  # dimension per attention head
            img_encoder= image_encoder,  # plugin your image encoder (this can be optional if you pass in the image embeddings separately, but probably want to train end to end given the perceiver resampler)
            media_token_id=args['model']['media_token_id'],  # the token id representing the [media] or [image]
            cross_attn_every=args['model']['cross_att_every'],  # how often to cross attend
            perceiver_num_latents=args['model']['perceiver_num_latents'],  # perceiver number of latents, should be smaller than the sequence length of the image tokens
            perceiver_depth=args['model']['perceicer_depth'],  # perceiver resampler depth
            language_model=args['model']['language_encoder'],  # language model    (gpt2 or palm)
            img_encoder_outdim=self.img_encoder_outdim,
            pretrained_gpt2_path=args['model']['pretrained_language_path'],
            weight_tie_gpt = args['model']['weight_tie_gpt'],
            classification_mode = self.classification_mode,
            flamingo_mode=args['model']['flamingo_mode'],
            train_embedding_layer=args['model']['train_embedding_layer'],
            use_positional_embedding = args['model']['use_positional_embedding'],
            classify_only_image_features = args['model']['classify_only_image_features'],
            freeze_image_encoder = args['model']['freeze_image_encoder']
        )
        print('Flamingo is initalized')

        if self.classification_mode:
            #self.classifier = nn.Linear(dim,self.num_classification_classes )
            #nn.init.normal_(self.classifier.weight, std=0.02)

            if self.classify_only_image_features:
                self.classifier = nn.Sequential(
                    nn.Dropout(args['model']['classifier_dropout']),
                    nn.Linear( self.img_encoder_outdim, self.num_classification_classes),
                    nn.ReLU())
            
            # Concatenate image embeddings to flamingo embeddings
            elif self.use_image_embeddings:
                self.classifier = nn.Sequential(
                    LayerNorm(2*self.flamingo_embed_dim),
                    #nn.Linear(dim, 4096),
                    #nn.ReLU(),
                    nn.Dropout(args['model']['classifier_dropout']),
                    #nn.Linear(4096, self.num_classification_classes),
                    nn.Linear(2*self.flamingo_embed_dim, self.num_classification_classes),
                    #nn.ReLU()
                )
            else:
                self.classifier = nn.Sequential(
                    LayerNorm(self.flamingo_embed_dim),
                    #nn.Linear(dim, 4096),
                    #nn.ReLU(),
                    nn.Dropout(self.classifier_dropout),
                    #nn.Linear(4096, self.num_classification_classes),
                    nn.Linear(self.flamingo_embed_dim, self.num_classification_classes),
                )

            print('Self classifier ',self.classifier)

    def aggregate_tokens_until_eoq(self, token_embeds, eoq_indices, agg_func='mean'):
        """ Instead of only using EOQ token for answer classification
            it aggregates all previous tokens come before <EOQ> token.
        Args:
            token_embeds (_type_): _description_
            eoq_indices (_type_): _description_
            agg_func (str, optional): _description_. Defaults to 'mean'.

        Returns:
            _type_: _description_
        """
        aggregated_tensors = []
        for i, row in enumerate(token_embeds):
            if agg_func == 'mean':
                aggregated_tensors.append(torch.mean(row[:eoq_indices[i]+1,:],dim=0))
        return torch.stack(aggregated_tensors)


    def forward(self, x, return_attn=False, return_embeds=False):
        # in lightning, forward defines the prediction/inference actions
        images = x["image"]
        #print('Images FM shape', images.shape)
        input_tokens = x["input_ids"]
        
        batch_size = images.shape[0]
        
        if return_embeds:
            if self.classification_mode:
                index_eoq = x["index_eoq"]
                flamingo_logits, token_embeds = self.flamingo_palm(
                    input_tokens.squeeze(1), images.unsqueeze(1), return_attn=return_attn, return_embeds=return_embeds
                )

                classification_logits = self.classifier(token_embeds[torch.arange(batch_size), index_eoq])
                classification_logits = torch.softmax(classification_logits, dim=1)

                return token_embeds, classification_logits

            else:
                text_embeds = self.flamingo_palm(
                    input_tokens.squeeze(1), images.unsqueeze(1), return_attn=return_attn, return_embeds=return_embeds
                )
                return text_embeds

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

            elif self.classification_mode and not self.use_image_embeddings :
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
        token_type_ids = batch["token_type_ids"] 
        targets = batch["targets"]
        batch_size = images.shape[0]


        if self.classify_only_image_features:
            class_labels = batch["label"]
            index_eoq = batch["index_eoq"]

            with torch.no_grad():
                image_embeds = self.flamingo_palm.img_encoder(images)


            classification_logits = self.classifier(image_embeds.squeeze(1))
            # Calculate training accuracy
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
            train_classification_loss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)(classification_logits, class_labels)
            return {"loss": train_classification_loss, 'train_acc_step':train_acc}



        elif self.classification_mode and self.use_image_embeddings:
            class_labels = batch["label"]
            index_eoq = batch["index_eoq"]
            flamingo_logits, token_embeds, image_embeddings = self.flamingo_palm(
                    input_tokens.squeeze(1), images.unsqueeze(1), token_type_ids.unsqueeze(1),
                    return_image_embeddings = self.use_image_embeddings
                )

            if self.token_average_classification:
                eoq_embeds=self.aggregate_tokens_until_eoq(token_embeds,index_eoq)
            else:
                eoq_embeds = token_embeds[torch.arange(batch_size), index_eoq]
            classification_logits = self.classifier(torch.cat([image_embeddings.squeeze(1), eoq_embeds],dim=1))


        elif self.classification_mode:
            class_labels = batch["label"]
            index_eoq = batch["index_eoq"]
        
            flamingo_logits, token_embeds = self.flamingo_palm(
                input_tokens.squeeze(1), images.unsqueeze(1), token_type_ids.unsqueeze(1)
            )

            
            classification_logits = self.classifier(token_embeds[torch.arange(batch_size), index_eoq])
        else:
            flamingo_logits = self.flamingo_palm(
            input_tokens.squeeze(1), images.unsqueeze(1), token_type_ids.unsqueeze(1),
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
        wandb.log({"train_loss_generation":train_loss})
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
            wandb.log({"train_classification_loss":train_classification_loss, 'train_total_loss':train_loss + train_classification_loss,
                        "train_acc":train_acc})
            # self.loggers[-1].experiment.log_metrics(comet_logs, step=self.global_step)
            self.logger.experiment.add_scalars(
                "losses", {"train_loss": train_loss + train_classification_loss}, self.global_step
            )

            if self.use_generation_loss == False:
                train_loss=0
            return {"loss": train_loss + train_classification_loss, 'train_acc_step': train_acc }

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

        if self.classify_only_image_features:
            class_labels = batch["label"]
            index_eoq = batch["index_eoq"]
            from einops import rearrange
            #images = rearrange(images, "b t ... -> (b t) ...")

            with torch.no_grad():
                image_embeds = self.flamingo_palm.img_encoder(images)


            #print('self classifier', self.classifier)
            classification_logits = self.classifier(image_embeds.squeeze(1))
            # Calculate validation accuracy
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
            val_classification_loss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)(classification_logits, class_labels)
            return {"val_loss":  val_classification_loss, 'val_acc_step':val_acc}


        elif self.classification_mode and self.use_image_embeddings:
            class_labels = batch["label"]
            index_eoq = batch["index_eoq"]
            flamingo_logits, token_embeds, image_embeddings = self.flamingo_palm(
                    input_tokens.squeeze(1), images.unsqueeze(1), 
                    return_image_embeddings = self.use_image_embeddings
                )

            if self.token_average_classification:
                eoq_embeds = self.aggregate_tokens_until_eoq(token_embeds, index_eoq)
            else:
                eoq_embeds = token_embeds[torch.arange(batch_size), index_eoq]
            #print('EOQ embeds shape', eoq_embeds.shape)
            #print('image_embeddings.squeeze(1), shape ',image_embeddings.squeeze(1).shape)

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

            wandb.log({"val_loss_generation":val_loss, 'val_classification_loss':val_classification_loss,
                        "val_total_loss":val_loss + val_classification_loss, "val_acc":val_acc})

            self.logger.experiment.add_scalars(
                "losses", {"validation_loss": val_loss+ val_classification_loss}, self.global_step
            )
            if self.use_generation_loss == False:
                val_loss = 0
            return {"val_loss": val_loss + val_classification_loss, 'val_acc_step':val_acc}

        # Generation Mode 
        else:
            self.logger.experiment.add_scalars(
                "losses", {"validation_loss": val_loss}, self.global_step
            )
            return {"val_loss": val_loss}


        

    def predict_step(self, batch):
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


    def validation_epoch_end(self, validation_step_outputs):
        val_acc_epoch = torch.mean(torch.stack([metric['val_acc_step'] for metric in validation_step_outputs]))
        val_loss_epoch = torch.mean(torch.stack([metric['val_loss'] for metric in validation_step_outputs]))
        wandb.log({"val_acc_epoch": val_acc_epoch, 'val_loss_epoch':val_loss_epoch, 'epoch':self.current_epoch})
    
    def training_epoch_end(self, train_step_outputs):
        train_acc_epoch = torch.mean(torch.stack([metric['train_acc_step'] for metric in train_step_outputs]))
        train_loss_epoch = torch.mean(torch.stack([metric['loss'] for metric in train_step_outputs]))
        wandb.log({"train_acc_epoch": train_acc_epoch, 'train_loss_epoch':train_loss_epoch, 'epoch':self.current_epoch})


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
        #params = list(self.named_parameters())
    
        # grouped_parameters = [
        # {"params": [p for n, p in params if n.startswith('flamingo_palm.perceiver_resampler')], 'weight_decay': 0},
        # {"params": [p for n, p in params if not n.startswith('flamingo_palm.perceiver_resampler') ],'weight_decay': 0.01 },
        # ]


        if self.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
        )   
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        if self.use_scheduler:
            return [optimizer], [scheduler]
        else:
            return optimizer
