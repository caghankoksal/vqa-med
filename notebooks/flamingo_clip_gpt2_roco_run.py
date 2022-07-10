# Import comet_ml at the top of your file
import comet_ml
import sys 
sys.path.append('..')

import os
import pytorch_lightning as pl
from src.datasets.roco_dataset import ROCODataModule
from src.models.multimodal.flamingo_module import FlamingoModule

from pytorch_lightning import Trainer, seed_everything
import torchvision.transforms as T

from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy

if __name__ == '__main__':
    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    seed_everything(42, workers=True)


    
    #MIMIC CXR  Mean and Std of the dataset
    #mean: tensor([0.4719, 0.4719, 0.4719])
    #std:  tensor([0.3029, 0.3029, 0.3029])
    

    augmentations = {'train':
        T.Compose(
        [
            T.Resize((224, 224)),
            T.RandomRotation(10),
            T.ToTensor(),
            T.Normalize(mean=(0.3570, 0.3621, 0.3636), std=(0.2924, 0.2941, 0.2951))
        ]),
        'validation':
        T.Compose(
        [   T.Resize((224, 224)),
            T.RandomRotation(10),
            T.ToTensor(),
            T.Normalize(mean=(0.3570, 0.3621, 0.3636), std=(0.2924, 0.2941, 0.2951))
        ]),
        'test':
        T.Compose(
        [
            T.Resize((224, 224)),
            T.RandomRotation(10),
            T.ToTensor(),
            T.Normalize(mean=(0.3570, 0.3621, 0.3636), std=(0.2924, 0.2941, 0.2951))
        ])
    }


    # Hyperparameters
    NUM_DATA_WORKERS  = 4
    ONLY_IMAGES = False
    BATCH_SIZE = 96
    NUM_EPOCHS = 200
    LIMIT_NUM_SAMPLES = None

    if os.getcwd().startswith('/home/mlmi-matthias'):
        ACCELERATOR = "gpu"
        DEVICES = [4,5,6,7]
        PRETRAINED_CLIP_PATH = '/home/mlmi-matthias/Caghan/pretrained_models/PubMedCLIP_ViT32.pth'
        PRETRAINED_GPT2_PATH = "/home/mlmi-matthias/Caghan/pretrained_models/gpt2-pytorch_model.bin"
        ROOT = '/home/mlmi-matthias/roco-dataset/data'

    elif os.getcwd().startswith('/Users/caghankoksal'):
        PRETRAINED_CLIP_PATH = '/Users/caghankoksal/Desktop/development/PubMedCLIP_ViT32.pth'
        PRETRAINED_GPT2_PATH = "/Users/caghankoksal/Desktop/development/TransformerPlay/gpt2-pytorch_model.bin"
        ACCELERATOR = "cpu"
        DEVICES = 1
        ROOT = '/Users/caghankoksal/Desktop/development/roco-dataset/data'


    IMAGE_TYPE = "jpg"
    TOKENIZER  = "gpt2"
    PREPROCESSED = True



    hyperparameters = {
        "root": ROOT,
        "batch_size": BATCH_SIZE,
        "tokenizer": "gpt2",
        "num_data_workers": NUM_DATA_WORKERS,
        "return_size": False,
        "augmentations": augmentations,
        "limit_num_samples": None,
        "token_max_len": 128
        
    }

    roco_datamodule = ROCODataModule(**hyperparameters)


    train_loader = roco_datamodule.train_dataloader()
    val_loader = roco_datamodule.val_dataloader()

    print("Len training dataset : ", len(roco_datamodule.train_dataset),
        "Batch Size : ", BATCH_SIZE, "NUM_EPOCHS : ",NUM_EPOCHS )
    print("Total training steps : ", len(roco_datamodule.train_dataset)//BATCH_SIZE*NUM_EPOCHS)


    # MODEL HPRAMS
    VOCAB_SIZE_OF_TOKENIZER = 50257 # mimic_datamodule.train_dataset.tokenizer.vocab_size
    LANGUAGE_MODEL = 'gpt2'
    NUM_TOKENS = VOCAB_SIZE_OF_TOKENIZER +3 if LANGUAGE_MODEL=="gpt2" else 31092
    FLAMINGO_EMBED_DIM = 768
    DEPTH = 12
    NUM_HEADS = 8
    ATT_HEAD_DIM = 64
    CROOS_ATT_EVERY=3
    MEDIA_TOKEN_ID = roco_datamodule.train_dataset.tokenizer.\
        all_special_ids[roco_datamodule.train_dataset.tokenizer.all_special_tokens.index('<image>')]
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
        'warmup_steps': 569,
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


    lr_monitor = LearningRateMonitor(logging_interval='step')  
    checkpoint_callback = ModelCheckpoint(
                filename='{epoch}-{val_loss:.2f}-{other_metric:.2f}',
                    monitor= 'val_loss',
                        save_top_k = 10)
    
    trainer = pl.Trainer(max_epochs=NUM_EPOCHS,
                        accelerator=ACCELERATOR, devices=DEVICES,
                        callbacks=[lr_monitor, checkpoint_callback],
                        strategy=DDPStrategy(find_unused_parameters=False))

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
