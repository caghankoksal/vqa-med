# Import comet_ml at the top of your file
import comet_ml
import sys 
sys.path.append('..')

import os
import pytorch_lightning as pl
from src.datasets.mimic_cxr_dataset import MIMICCXRDataModule
from src.models.multimodal.flamingo_module import FlamingoModule

from pytorch_lightning import Trainer, seed_everything
import torchvision.transforms as T

from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers


# sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
seed_everything(42, workers=True)



augmentations = {'train':
    T.Compose(
    [
        T.Resize((224, 224)),
        T.RandomApply([T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)], p=0.8),
        #T.RandomGrayscale(p=0.2),
        T.ToTensor(),
    ]),
    'val':
    T.Compose(
    [
        T.Resize((224, 224)),
        T.RandomApply([T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)], p=0.8),
        #T.RandomGrayscale(p=0.2),
        #T.GaussianBlur(kernel_size=9),
        T.ToTensor(),
    ])
}


NUM_DATA_WORKERS  = 0
ONLY_IMAGES = False
# Hyperparameters
DATA_PATH = '/home/mlmi-matthias/physionet.org/files/mimic-cxr/2.0.0/files/'
BATCH_SIZE = 32
NUM_EPOCHS = 200
LIMIT_NUM_SAMPLES = None

# DATAMODULE
mimic_datamodule = MIMICCXRDataModule(DATA_PATH, transforms=augmentations, only_images=False, batch_size=BATCH_SIZE,
                                limit_num_samples=LIMIT_NUM_SAMPLES, num_data_workers=NUM_DATA_WORKERS, tokenizer="gpt2")
train_loader = mimic_datamodule.train_dataloader()
val_loader = mimic_datamodule.val_dataloader()

print("Len training dataset : ", len(mimic_datamodule.train_dataset), "Batch Size : ", BATCH_SIZE, "NUM_EPOCHS : ",NUM_EPOCHS )
print("Total training steps : ", len(mimic_datamodule.train_dataset)//BATCH_SIZE*NUM_EPOCHS)


# MODEL HPRAMS
VOCAB_SIZE_OF_TOKENIZER = mimic_datamodule.train_dataset.tokenizer.vocab_size
VOCAB_SIZE_OF_TOKENIZER
LANGUAGE_MODEL = 'gpt2'
NUM_TOKENS = VOCAB_SIZE_OF_TOKENIZER+3 if LANGUAGE_MODEL=="gpt2" else 31092
FLAMINGO_EMBED_DIM = 768
DEPTH = 12
NUM_HEADS = 8
ATT_HEAD_DIM = 64
CROOS_ATT_EVERY=3
MEDIA_TOKEN_ID = mimic_datamodule.train_dataset.tokenizer.all_special_ids[mimic_datamodule.train_dataset.tokenizer.all_special_tokens.index('<image>')]
PERCEIVER_NUM_LATENTS = 64
PERCEIVER_DEPTH = 2
IMAGE_ENCODER = "clip"





if os.getcwd().startswith('/home/mlmi-matthias'):
    ACCELERATOR = "gpu"
    DEVICES = [0]
    PRETRAINED_CLIP_PATH = '/home/mlmi-matthias/Caghan/pretrained_models/PubMedCLIP_ViT32.pth'
    PRETRAINED_GPT2_PATH = "/home/mlmi-matthias/Caghan/pretrained_models/gpt2-pytorch_model.bin"
elif os.getcwd().startswith('/Users/caghankoksal'):
    PRETRAINED_CLIP_PATH = '/Users/caghankoksal/Desktop/development/PubMedCLIP_ViT32.pth'
    PRETRAINED_GPT2_PATH = "/Users/caghankoksal/Desktop/development/TransformerPlay/gpt2-pytorch_model.bin"
    ACCELERATOR = "gpu"
    DEVICES = 0
    


# COMET EXPERIMENT LOGGER
COMET_API_KEY = "F2L19mQwKXSoeF1IYEDA2AeHD",
PROJECT_KEY = "flamingo-gpt2"

TOTAL_STEPS=20000



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


model = FlamingoModule(pretrained_clip_path = PRETRAINED_CLIP_PATH,
                      total_steps=TOTAL_STEPS, num_tokens = NUM_TOKENS,
                      dim=FLAMINGO_EMBED_DIM, depth=DEPTH, heads=NUM_HEADS, dim_head=ATT_HEAD_DIM,
                      media_token_id=MEDIA_TOKEN_ID, cross_attn_every=CROOS_ATT_EVERY,
                      perceiver_num_latents = PERCEIVER_NUM_LATENTS, perceiver_depth = PERCEIVER_DEPTH,
                      image_encoder =IMAGE_ENCODER, language_model = LANGUAGE_MODEL,
                      pretrained_gpt2_path=PRETRAINED_GPT2_PATH
                        )



comet_logger = CometLogger(
    api_key= COMET_API_KEY,
    project_name=PROJECT_KEY)

lr_monitor = LearningRateMonitor(logging_interval='step')
tb_logger = pl_loggers.TensorBoardLogger(save_dir="pll_logs/")


trainer = pl.Trainer(max_epochs=NUM_EPOCHS,
                     accelerator=ACCELERATOR, devices=DEVICES,
                     logger=[tb_logger,comet_logger],
                     callbacks=[lr_monitor])

trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)







