import sys 
import torch
sys.path.append('..')

import os
import pytorch_lightning as pl
from src.datasets.vqa_rad_dataset import VQRadDataModule
from src.models.multimodal.flamingo_module import FlamingoModule

from pytorch_lightning import Trainer, seed_everything
import torchvision.transforms as T

from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers

seed_everything(42, workers=True)

img_mean = (0.48,0.48,0.48)
img_std = (0.265,0.265,0.265)

transforms = {'train':
    T.Compose(
    [
        T.RandomRotation(10),
        T.ToTensor(),
        # T.Normalize(mean=img_mean, std=img_std)
    ]),
    'val':
    T.Compose(
    [
        T.RandomRotation(10),
        T.ToTensor(),
        # T.Normalize(mean=img_mean, std=img_std)
    ]),
    'test':
    T.Compose(
    [
        T.ToTensor(),
        # T.Normalize(mean=img_mean, std=img_std)
    ])
}


# Hyperparameters
NUM_DATA_WORKERS  = 8
ONLY_IMAGES = False
BATCH_SIZE = 64
NUM_EPOCHS = 80
LIMIT_NUM_SAMPLES = None

ACCELERATOR = "gpu"
DEVICES = [6]
# ACCELERATOR = "cpu"
# DEVICES = 1
DATASET_ROOT = '/home/mlmi-matthias/VQA-RAD'
PRETRAINED_CLIP_PATH = '/home/mlmi-matthias/Caghan/pretrained_models/PubMedCLIP_ViT32.pth'
PRETRAINED_GPT2_PATH = "/home/mlmi-matthias/Caghan/pretrained_models/gpt2-pytorch_model.bin"



IMAGE_TYPE = "jpg"
SHUFFLE = True
TOKENIZER  = "gpt2"
LOAD_IN_MEM = True
PREPROCESSED = False

mimic_datamodule = VQRadDataModule(
                                batch_size=BATCH_SIZE, transforms=transforms, root=DATASET_ROOT,
                                limit_num_samples=LIMIT_NUM_SAMPLES, num_workers=NUM_DATA_WORKERS, shuffle=SHUFFLE,
                                tokenizer="gpt2", preprocessed=PREPROCESSED, load_in_memory=LOAD_IN_MEM
)


train_loader = mimic_datamodule.train_dataloader()
val_loader = mimic_datamodule.val_dataloader()

print("Len training dataset : ", len(mimic_datamodule.train_dataset),
    "Batch Size : ", BATCH_SIZE, "NUM_EPOCHS : ",NUM_EPOCHS )
print("Total training steps : ", len(mimic_datamodule.train_dataset)//BATCH_SIZE*NUM_EPOCHS)


# MODEL HPRAMS
VOCAB_SIZE_OF_TOKENIZER = 50258 # mimic_datamodule.train_dataset.tokenizer.vocab_size
LANGUAGE_MODEL = 'gpt2'
NUM_TOKENS = VOCAB_SIZE_OF_TOKENIZER +3 if LANGUAGE_MODEL=="gpt2" else 31092
FLAMINGO_EMBED_DIM = 768
DEPTH = 12
NUM_HEADS = 8
ATT_HEAD_DIM = 64
CROOS_ATT_EVERY=3
MEDIA_TOKEN_ID = mimic_datamodule.train_dataset.tokenizer.\
    all_special_ids[mimic_datamodule.train_dataset.tokenizer.all_special_tokens.index('<image>')]
PERCEIVER_NUM_LATENTS = 64
PERCEIVER_DEPTH = 2
IMAGE_ENCODER = "clip"



print("LANGUAGE_MODEL : ",LANGUAGE_MODEL, "\n"
        "NUM_TOKENS : ",NUM_TOKENS, "\n"
        "FLAMINGO_EMBED_DIM : ",FLAMINGO_EMBED_DIM, "\n"
        "DEPTH : ",DEPTH, "\n"
        "NUM_HEADS : ",NUM_HEADS, "\n"
        "ATT_HEAD_DIM : ",ATT_HEAD_DIM, "\n"
        "CROOS_ATT_EVERY : ",CROOS_ATT_EVERY, "\n"
        "MEDIA_TOKEN_ID : ",MEDIA_TOKEN_ID, "\n"
        "PERCEIVER_NUM_LATENTS : ",PERCEIVER_NUM_LATENTS, "\n"
        "PERCEIVER_DEPTH : ",PERCEIVER_DEPTH, "\n"
        "IMAGE_ENCODER : ",IMAGE_ENCODER, "\n"
        "PRETRAINED_CLIP_PATH : ",PRETRAINED_CLIP_PATH, "\n"
        "PRETRAINED_GPT2_PATH : ",PRETRAINED_GPT2_PATH, "\n")


hyperparams = {
    'pretrained_clip_path': PRETRAINED_CLIP_PATH,
    'warmup_steps': 0,
    'num_tokens': NUM_TOKENS,
    'dim': FLAMINGO_EMBED_DIM,
    'depth': DEPTH,
    'num_heads': NUM_HEADS,
    'dim_head': ATT_HEAD_DIM,
    'cross_attn_every': CROOS_ATT_EVERY,
    'media_token_id': MEDIA_TOKEN_ID,
    'perceiver_num_latents': PERCEIVER_NUM_LATENTS,
    'perceiver_depth': PERCEIVER_DEPTH,
    'image_encoder': IMAGE_ENCODER,
    'language_model': LANGUAGE_MODEL,
    'pretrained_gpt2_path': PRETRAINED_GPT2_PATH,
}


model = FlamingoModule(**hyperparams)

CHECKPOINT_PATH = "/home/mlmi-matthias/Caghan/mlmi-vqa/notebooks/lightning_logs/version_20/checkpoints/epoch=114-val_loss=0.84-other_metric=0.00.ckpt"
START_FROM_CHECKPOINT = True

if START_FROM_CHECKPOINT:
    print("Pretrained Flamingo Model is loaded from checkpoint : ",CHECKPOINT_PATH)
    model.load_state_dict(torch.load(CHECKPOINT_PATH)["state_dict"])


lr_monitor = LearningRateMonitor(logging_interval='step')
from pytorch_lightning.callbacks import ModelCheckpoint
checkpoint_callback = ModelCheckpoint(
            filename='{epoch}-{val_loss:.2f}-{other_metric:.2f}',
                monitor= 'val_loss',
                    save_top_k = 10)

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
early_stopping_callback = EarlyStopping(monitor="val_loss", mode="min",patience=5)


# from pytorch_lightning.strategies import DDPStrategy
trainer = pl.Trainer(max_epochs=NUM_EPOCHS,
                    accelerator=ACCELERATOR, devices=DEVICES,
                    callbacks=[lr_monitor, checkpoint_callback, early_stopping_callback],
                    # strategy=DDPStrategy(find_unused_parameters=False)
                    )

trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
