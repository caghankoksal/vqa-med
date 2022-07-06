# Import comet_ml at the top of your file
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
from src.datasets.mimic_cxr_dataset import MIMICCXRDataModule
import torch
from tqdm import tqdm as tqdm


def calculate_statistics(cur_datamodule):
    # placeholders
    psum    = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])

    # loop through images
    for batch in tqdm(cur_datamodule.train_dataloader()):
        psum    += batch["image"].sum(axis = [0, 2, 3])
        psum_sq += (batch["image"] ** 2).sum(axis = [0, 2, 3])

    ###### FINAL CALCULATIONS

    # pixel count
    count = len(cur_datamodule.train_dataset) * 224 * 224

    # mean and std
    total_mean = psum / count
    total_var  = (psum_sq / count) - (total_mean ** 2)
    total_std  = torch.sqrt(total_var)

    # output
    print('mean: '  + str(total_mean))
    print('std:  '  + str(total_std))

    return total_mean, total_std


if __name__ == '__main__':
    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    seed_everything(42, workers=True)

 
    augmentations = {
        
        'train': T.Compose([T.Resize((224,224)),
                            T.ToTensor(),
                            #T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                        ]),
        'val': T.Compose([T.Resize((224,224)),
                            T.ToTensor(),
                            #T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                            ]),
        'test': T.Compose([T.Resize((224,224)),
                            T.ToTensor(),
                            #T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                            ]),
    }

    
    # Hyperparameters
    NUM_DATA_WORKERS  = 2
    ONLY_IMAGES = False
    BATCH_SIZE = 96
    NUM_EPOCHS = 60
    LIMIT_NUM_SAMPLES = None

    if os.getcwd().startswith('/home/mlmi-matthias'):
        ACCELERATOR = "gpu"
        DEVICES = [4,5,6,7]
        PRETRAINED_CLIP_PATH = '/home/mlmi-matthias/Caghan/pretrained_models/PubMedCLIP_ViT32.pth'
        PRETRAINED_GPT2_PATH = "/home/mlmi-matthias/Caghan/pretrained_models/gpt2-pytorch_model.bin"
        MIMIC_CXR_DCM_PATH = '/home/mlmi-matthias/physionet.org/files/mimic-cxr/2.0.0/files/'
        MIMIC_CXR_JPG_PATH = "/home/mlmi-matthias/physionet.org/files/mimic-cxr-jpg/2.0.0/files/"
        SPLIT_PATH = '/home/mlmi-matthias/Caghan/mlmi-vqa/data/external/'
        IMAGECLEF_PATH ='/home/mlmi-matthias/imagepwdclef/'
        CHECKPOINT_PATH = "/home/mlmi-matthias/Caghan/mlmi-vqa/notebooks/lightning_logs/version_20/checkpoints/epoch=114-val_loss=0.84-other_metric=0.00.ckpt"


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


    IMAGE_TYPE = "jpg"
    TOKENIZER  = "gpt2"
    PREPROCESSED = True
    

    dataset_hyperparameters = {
        "root": IMAGECLEF_PATH,
        "batch_size": BATCH_SIZE,
        "tokenizer": TOKENIZER,
        "return_size": False,
        "num_data_workers": NUM_DATA_WORKERS
    }


    #imageclef_datamodule = ImageCLEF2021DataModule(**dataset_hyperparameters,transforms=augmentations)

    """
    print("Len training dataset  ImageCLEF2021DataModule: ", len(imageclef_datamodule.train_dataset),
        "Batch Size : ", BATCH_SIZE, "NUM_EPOCHS : ",NUM_EPOCHS )
    print("Total training steps ImageCLEF2021DataModule: ", len(imageclef_datamodule.train_dataset)//BATCH_SIZE*NUM_EPOCHS)
    """

    #######################################################################################################################
    # ImageCLEF Statistics
    #imageclef_total_mean, imageclef_total_std = calculate_statistics(imageclef_datamodule)

    #######################################################################################################################


    #######################################################################################################################
    # MIMIC DATAMODULE
    mimic_datamodule = MIMICCXRDataModule(MIMIC_CXR_DCM_PATH, MIMIC_CXR_JPG_PATH, 
                                        transforms=augmentations, only_images=False, batch_size=BATCH_SIZE,
                                        limit_num_samples=LIMIT_NUM_SAMPLES, num_data_workers=NUM_DATA_WORKERS,
                                        tokenizer="gpt2",image_type="jpg", split_path=SPLIT_PATH, preprocessed=PREPROCESSED
    )

    print("Len training dataset  MIMICCXRDataModule: ", len(mimic_datamodule.train_dataset),
        "Batch Size : ", BATCH_SIZE, "NUM_EPOCHS : ",NUM_EPOCHS )
    print("Total training steps MIMICCXRDataModule: ", len(mimic_datamodule.train_dataset)//BATCH_SIZE*NUM_EPOCHS)

    mimic_total_mean, mimic_total_std = calculate_statistics(mimic_datamodule)