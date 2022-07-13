# Import comet_ml at the top of your file
from email.policy import strict
import comet_ml
import sys 
sys.path.append('..')

import os
import pytorch_lightning as pl
from src.models.multimodal.flamingo_module import FlamingoModule

from pytorch_lightning import Trainer, seed_everything
import torchvision.transforms as T

from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers

from src.datasets.imageclef_dataset import ImageCLEF2021DataModule
import torch

if __name__ == '__main__':
    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    seed_everything(42, workers=True)

    # Mean and std of the dataset
    # mean: tensor([0.2570, 0.2570, 0.2570])
    # std:  tensor([0.2710, 0.2712, 0.2711])
    augmentations = {
        
        'train': T.Compose([T.Resize((224,224)),
                            T.ToTensor(),
                            T.Normalize(mean=(0.2570, 0.2570, 0.2570), std=(0.2710, 0.2710, 0.2710))
                            ]),
        'val': T.Compose([T.Resize((224,224)),
                            T.ToTensor(),
                            T.Normalize(mean=(0.2570, 0.2570, 0.2570), std=(0.2710, 0.2710, 0.2710))
                            ]),
        'test': T.Compose([T.Resize((224,224)),
                            T.ToTensor(),
                            T.Normalize(mean=(0.2570, 0.2570, 0.2570), std=(0.2710, 0.2710, 0.2710))
                            ])
    }

    
    # Hyperparameters
    NUM_DATA_WORKERS  = 2
    ONLY_IMAGES = False
    BATCH_SIZE = 96
    NUM_EPOCHS = 60
    LIMIT_NUM_SAMPLES = None
    DATASET = "IMAGECLEF"

    if os.getcwd().startswith('/home/mlmi-matthias'):
        ACCELERATOR = "gpu"
        DEVICES = [4,5,6,7]
        PRETRAINED_CLIP_PATH = '/home/mlmi-matthias/Caghan/pretrained_models/PubMedCLIP_ViT32.pth'
        PRETRAINED_GPT2_PATH = "/home/mlmi-matthias/Caghan/pretrained_models/gpt2-pytorch_model.bin"
        MIMIC_CXR_DCM_PATH = '/home/mlmi-matthias/physionet.org/files/mimic-cxr/2.0.0/files/'
        MIMIC_CXR_JPG_PATH = "/home/mlmi-matthias/physionet.org/files/mimic-cxr-jpg/2.0.0/files/"
        SPLIT_PATH = '/home/mlmi-matthias/Caghan/mlmi-vqa/data/external/'
        IMAGECLEF_PATH ='/home/mlmi-matthias/imageclef/'
        CHECKPOINT_PATH = "/home/mlmi-matthias/Caghan/mlmi-vqa/notebooks/lightning_logs/version_20/checkpoints/epoch=114-val_loss=0.84-other_metric=0.00.ckpt"
        ANSWERS_LIST_PATH = '/home/mlmi-matthias/Caghan/mlmi-vqa//data/external/answer_list_imageclef.txt'


    elif os.getcwd().startswith('/Users/caghankoksal'):
        PRETRAINED_CLIP_PATH = '/Users/caghankoksal/Desktop/development/PubMedCLIP_ViT32.pth'
        PRETRAINED_GPT2_PATH = "/Users/caghankoksal/Desktop/development/TransformerPlay/gpt2-pytorch_model.bin"
        ACCELERATOR = "cpu"
        DEVICES = 1
        MIMIC_CXR_DCM_PATH = '/Users/caghankoksal/Desktop/development/Flamingo-playground/physionet.org/files/mimic-cxr/2.0.0/files/'
        MIMIC_CXR_JPG_PATH = '/Users/caghankoksal/Desktop/development/physionet.org/files/mimic-cxr-jpg/2.0.0/files/'
        SPLIT_PATH = '/Users/caghankoksal/Desktop/SS2022/mlmi-vqa/data/external/'
        IMAGECLEF_PATH = "/Users/caghankoksal/Desktop/imageclef/"
        CHECKPOINT_PATH = "/Users/caghankoksal/Desktop/logs_from_cluster/version_20/checkpoints/epoch=117-val_loss=0.84-other_metric=0.00.ckpt"
        ANSWERS_LIST_PATH = '/Users/caghankoksal/Desktop/SS2022/mlmi-vqa/data/external/answer_list_imageclef.txt'


    IMAGE_TYPE = "jpg"
    TOKENIZER  = "gpt2"
    PREPROCESSED = True
    RETURN_IDX_EOC = True

    dataset_hyperparameters = {
        "root": IMAGECLEF_PATH,
        "batch_size": BATCH_SIZE,
        "tokenizer": TOKENIZER,
        "return_size": False,
        "num_data_workers": NUM_DATA_WORKERS,
        "answers_list_path": ANSWERS_LIST_PATH,
        "return_idx_answer_eoc": RETURN_IDX_EOC
    }


    datamodule = ImageCLEF2021DataModule(**dataset_hyperparameters,transforms=augmentations)


    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    print("Len training dataset : ", len(datamodule.train_dataset),
        "Batch Size : ", BATCH_SIZE, "NUM_EPOCHS : ",NUM_EPOCHS )
    print("Total training steps : ", len(datamodule.train_dataset)//BATCH_SIZE*NUM_EPOCHS)


    # MODEL HPRAMS
    VOCAB_SIZE_OF_TOKENIZER = 50257 # mimic_datamodule.train_dataset.tokenizer.vocab_size
    LANGUAGE_MODEL = 'gpt2'
    NUM_TOKENS = VOCAB_SIZE_OF_TOKENIZER +3 if LANGUAGE_MODEL=="gpt2" else 31092
    FLAMINGO_EMBED_DIM = 768
    DEPTH = 12
    NUM_HEADS = 8
    ATT_HEAD_DIM = 64
    CROOS_ATT_EVERY=3
    MEDIA_TOKEN_ID = datamodule.train_dataset.tokenizer.\
        all_special_ids[datamodule.train_dataset.tokenizer.all_special_tokens.index('<image>')]
    PERCEIVER_NUM_LATENTS = 64
    PERCEIVER_DEPTH = 2
    IMAGE_ENCODER = "clip"
    CLASSIFICATION_MODE = True 
    NUM_CLASSES = 332


    hyperparams = {
        'pretrained_clip_path': PRETRAINED_CLIP_PATH,
        'warmup_steps': 10,
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
        'classification_mode': CLASSIFICATION_MODE,
        'classification_num_classes': NUM_CLASSES  # 332 if DATASET=="IMAGECLEF"
    }


    def print_hyperparams(hparams):
        for k,v in hparams.items():
            print(k,v)

    print_hyperparams(hyperparams)


    model = FlamingoModule(**hyperparams)
    
    START_FROM_CHECKPOINT = True

    if START_FROM_CHECKPOINT:
        print("Pretrained Flamingo Model is loaded from checkpoint : ",CHECKPOINT_PATH)
        if os.getcwd().startswith('/home/mlmi-matthias'):
            model.load_state_dict(torch.load(CHECKPOINT_PATH)["state_dict"])
        else:
            model.load_state_dict(torch.load(CHECKPOINT_PATH,map_location=torch.device('cpu'))["state_dict"],strict=False)

    lr_monitor = LearningRateMonitor(logging_interval='step')


    from pytorch_lightning.callbacks import ModelCheckpoint

    checkpoint_callback = ModelCheckpoint(
                filename='{epoch}-{val_loss:.2f}-{other_metric:.2f}',
                    monitor= 'val_loss',
                        save_top_k = 5)

    trainer = pl.Trainer(max_epochs=NUM_EPOCHS,
                        accelerator=ACCELERATOR, devices=DEVICES,
                        callbacks=[lr_monitor, checkpoint_callback])

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
