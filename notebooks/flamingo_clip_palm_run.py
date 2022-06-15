import sys 
sys.path.append('..')

import os
from src.models.multimodal.flamingo_palm_clip import FlamingoClipPalm
import pytorch_lightning as pl

from src.datasets.mimic_cxr_dataset import MIMICCXRDataModule
from pytorch_lightning import Trainer, seed_everything
import torchvision.transforms as T
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

data_path = '/home/mlmi-matthias/physionet.org/files/mimic-cxr/2.0.0/files/'
BATCH_SIZE = 32
NUM_EPOCHS = 200
datamodule = MIMICCXRDataModule(data_path, transforms=augmentations, only_images=False, batch_size=BATCH_SIZE,
                                limit_num_samples=None)
train_loader = datamodule.train_dataloader()
val_loader = datamodule.val_dataloader()

pretrained_clip_path = '/home/mlmi-matthias/Caghan/pretrained_models/PubMedCLIP_ViT32.pth'

print("Len training dataset : ", len(datamodule.train_dataset), "Batch Size : ", BATCH_SIZE, "NUM_EPOCHS : ",NUM_EPOCHS )
print("Total training steps : ", len(datamodule.train_dataset)//BATCH_SIZE*NUM_EPOCHS)
model = FlamingoClipPalm(pretrained_clip_path = pretrained_clip_path)
trainer = pl.Trainer(max_epochs=NUM_EPOCHS,
                     accelerator="gpu", devices=1)
trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader, datamodule=datamodule)


