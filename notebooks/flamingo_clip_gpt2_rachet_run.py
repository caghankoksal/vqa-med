
import sys 
sys.path.append('..')

import os
import pytorch_lightning as pl
from src.models.multimodal.flamingo_module import FlamingoModule
from src.datasets.mimic_cxr_rachet import MIMICCXRRachetDataModule
from src.utils.utils import load_flamingo_weights, print_hyperparams

from pytorch_lightning import Trainer, seed_everything
import torchvision.transforms as T
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
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
                           # T.RandomHorizontalFlip(p=0.5),
                            T.RandomRotation(degrees=10),
                             T.Normalize(mean=(0.4719, 0.4719, 0.4719), std=(0.3029, 0.3029, 0.3029))
                            ]),
        'val': T.Compose([T.Resize((224,224)),
                            T.ToTensor(),
                           # T.RandomHorizontalFlip(p=0.5),
                            T.RandomRotation(degrees=10),
                             T.Normalize(mean=(0.4719, 0.4719, 0.4719), std=(0.3029, 0.3029, 0.3029))
                            ]),
        'test': T.Compose([T.Resize((224,224)),
                            T.ToTensor(),
                           # T.RandomHorizontalFlip(p=0.5),
                            T.RandomRotation(degrees=10),   
                            T.Normalize(mean=(0.4719, 0.4719, 0.4719), std=(0.3029, 0.3029, 0.3029))
                            ])
    }

    

    # Hyperparameters
    NUM_DATA_WORKERS  = 4
    ONLY_IMAGES = False
    BATCH_SIZE = 64
    NUM_EPOCHS = 200
    LIMIT_NUM_SAMPLES = None

    if os.getcwd().startswith('/home/mlmi-matthias'):
        ACCELERATOR = "gpu"
        DEVICES = [5,6]
        PRETRAINED_CLIP_PATH = '/home/mlmi-matthias/Caghan/pretrained_models/PubMedCLIP_ViT32.pth'
        PRETRAINED_GPT2_PATH = "/home/mlmi-matthias/Caghan/pretrained_models/gpt2-pytorch_model.bin"
        MIMIC_CXR_DCM_PATH = '/home/mlmi-matthias/physionet.org/files/mimic-cxr/2.0.0/files/'
        MIMIC_CXR_JPG_PATH = "/home/mlmi-matthias/physionet.org/files/mimic-cxr-jpg/2.0.0/files/"
        SPLIT_PATH = '/home/mlmi-matthias/Caghan/mlmi-vqa/notebooks/'
        MIMIC_ROOT = '/home/mlmi-matthias/physionet.org/'

    elif os.getcwd().startswith('/Users/caghankoksal'):
        PRETRAINED_CLIP_PATH = '/Users/caghankoksal/Desktop/development/PubMedCLIP_ViT32.pth'
        PRETRAINED_GPT2_PATH = "/Users/caghankoksal/Desktop/development/TransformerPlay/gpt2-pytorch_model.bin"
        ACCELERATOR = "cpu"
        DEVICES = 1
        MIMIC_CXR_DCM_PATH = '/Users/caghankoksal/Desktop/development/Flamingo-playground/physionet.org/files/mimic-cxr/2.0.0/files/'
        MIMIC_CXR_JPG_PATH = '/Users/caghankoksal/Desktop/development/physionet.org/files/mimic-cxr-jpg/2.0.0/files/'
        SPLIT_PATH = '/Users/caghankoksal/Desktop/SS2022/mlmi-vqa/data/external/'


    IMAGE_TYPE = "jpg"
    TOKENIZER  = "gpt2"
    PREPROCESSED = True


    

    dataset_hyperparameters = {
        "split_path":SPLIT_PATH,
        "mimic_root": MIMIC_ROOT,
        "batch_size": BATCH_SIZE,
        "tokenizer": TOKENIZER,
        "num_data_workers": NUM_DATA_WORKERS,
        "transforms": augmentations,
        "limit_num_samples": LIMIT_NUM_SAMPLES,
    }



    datamodule = MIMICCXRRachetDataModule(**dataset_hyperparameters)


    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    print("Len training dataset : ", len(datamodule.train_dataset),
        "Batch Size : ", BATCH_SIZE, "NUM_EPOCHS : ",NUM_EPOCHS )
    print("Total training steps : ", len(datamodule.train_dataset)//BATCH_SIZE*NUM_EPOCHS)


    # MODEL HPRAMS
    VOCAB_SIZE_OF_TOKENIZER = 50257 # mimic_datamodule.train_dataset.tokenizer.vocab_size
    LANGUAGE_MODEL = 'gpt2'
    NUM_TOKENS = VOCAB_SIZE_OF_TOKENIZER +4 if LANGUAGE_MODEL=="gpt2" else 31092
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
    CLASSIFICATION_MODE = False
    NUM_CLASSES = 332
    FLAMINGO_MODE = False
    LABEL_SMOOTHING = 0.2
    # Label smoothing for classification task
    TOKEN_LABEL_SMOOTHING = 0.0
    GRADIENT_CLIP_VAL = 1
    LEARNING_RATE = 1e-4
    USE_IMAGE_EMBEDDINGS = True
    TRAIN_EMBEDDING_LAYER = True
    CLASSIFIER_DROPOUT = 0.2


    hyperparams = {
        'pretrained_clip_path': PRETRAINED_CLIP_PATH,
        'warmup_steps': 30,
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
        'classification_num_classes': NUM_CLASSES,  # 332 if DATASET=="IMAGECLEF"
        'flamingo_mode': FLAMINGO_MODE,
        "label_smoothing": LABEL_SMOOTHING,
        "token_label_smoothing": TOKEN_LABEL_SMOOTHING,
        "learning_rate":LEARNING_RATE,
        "use_image_embeddings": USE_IMAGE_EMBEDDINGS,
        "train_embedding_layer": TRAIN_EMBEDDING_LAYER,
        "classifier_dropout": CLASSIFIER_DROPOUT
    }

    print_hyperparams(hyperparams)

    model = FlamingoModule(**hyperparams)
    START_FROM_CHECKPOINT = False

    if START_FROM_CHECKPOINT:
        print("Pretrained Flamingo Model is loaded from checkpoint : ",CHECKPOINT_PATH)
        if os.getcwd().startswith('/home/mlmi-matthias'):
            model.load_state_dict(torch.load(CHECKPOINT_PATH)["state_dict"],strict=False)
        else:
            model.load_state_dict(torch.load(CHECKPOINT_PATH,map_location=torch.device('cpu'))["state_dict"],strict=False)

    lr_monitor = LearningRateMonitor(logging_interval='step')


    if CLASSIFICATION_MODE:
            
        checkpoint_callback = ModelCheckpoint(
                    filename='{epoch}-{val_acc_epoch:.2f}-{val_total_loss_epoch:.2f}-{val_loss_generation_epoch:.2f}-{val_classification_loss_epoch:.2f}',
                    monitor= 'val_acc_epoch',
                    save_top_k = 10,
                    save_last=True,
                    mode="max")

 
        early_stopping_callback = EarlyStopping(monitor="val_acc_epoch", mode="max",patience=10)
    else:
        checkpoint_callback = ModelCheckpoint(
                filename='{epoch}-{val_loss_generation_epoch:.2f}',
                monitor= 'val_loss_generation_epoch',
                save_top_k = 10, 
                save_last=True,
                mode="min")
        early_stopping_callback = EarlyStopping(monitor="val_loss_generation_epoch", mode="min",patience=10)
    #early_stopping_callback = EarlyStopping(monitor="val_acc_epoch", mode="max",patience=10)

    # All our models are trained using the AdamW optimizer with global norm clipping of 1
    trainer = pl.Trainer(max_epochs=NUM_EPOCHS,
                        accelerator=ACCELERATOR, devices=DEVICES,
                        callbacks=[lr_monitor, checkpoint_callback,early_stopping_callback],
                        gradient_clip_val=GRADIENT_CLIP_VAL)

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
