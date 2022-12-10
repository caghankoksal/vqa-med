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

from src.utils.utils import load_config

if __name__ == '__main__':
    seed_everything(42, workers=True)

    img_mean = (0.48,0.48,0.48)
    img_std = (0.265,0.265,0.265)

    augmentations = {'train':
        T.Compose(
        [   
            T.Resize((224,224)),
            T.RandomRotation(10),
            T.ToTensor(),
            T.Normalize(mean=img_mean, std=img_std)
        ]),
        'val':
        T.Compose(
        [
            T.Resize((224,224)),
            T.RandomRotation(10),
            T.ToTensor(),
            T.Normalize(mean=img_mean, std=img_std)
        ]),
        'test':
        T.Compose(
        [
            T.Resize((224,224)),
            T.ToTensor(),
            T.Normalize(mean=img_mean, std=img_std)
        ])
    }


    args = load_config('/u/home/koksal/mlmi-vqa/configs','config.yaml')

    mimic_datamodule = VQRadDataModule(args, augmentations= augmentations)

    train_loader = mimic_datamodule.train_dataloader()
    val_loader = mimic_datamodule.val_dataloader()



    model = FlamingoModule(args) 

    if args['pretrained']:
        print("Pretrained Flamingo Model is loaded from checkpoint : ",args['pretrained'])
        model.load_state_dict(torch.load(args['pretrained'])["state_dict"])


    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping

    
    lr_monitor = LearningRateMonitor(logging_interval='step')

    CLASSIFICATION_MODE = True
    if CLASSIFICATION_MODE:
        checkpoint_callback = ModelCheckpoint(
                    filename='{epoch}-{val_acc_epoch:.2f}-{val_total_loss_epoch:.2f}-{val_loss_generation_epoch:.2f}-{val_classification_loss_epoch:.2f}',
                    monitor= 'val_acc_epoch',
                    save_top_k = 3,
                    save_last=True,
                    mode="max")

        #early_stopping_callback = EarlyStopping(monitor="val_acc_epoch", mode="max",patience=10)
        early_stopping_callback = EarlyStopping(monitor="val_total_loss_epoch", mode="min",patience=5)
    else:
        checkpoint_callback = ModelCheckpoint(
                filename='{epoch}-{val_loss_generation_epoch:.2f}',
                monitor= 'val_loss_generation_epoch',
                save_top_k = 3, 
                save_last=True,
                mode="min")
        early_stopping_callback = EarlyStopping(monitor="val_loss_generation_epoch", mode="min",patience=10)

    #early_stopping_callback = EarlyStopping(monitor="val_acc_epoch", mode="max",patience=10)

    # All our models are trained using the AdamW optimizer with global norm clipping of 1
    print(args['train']['devices'])
    trainer = pl.Trainer(max_epochs=args['train']['num_epochs'],
                        accelerator=args['train']['accelerator'], devices=args['train']['devices'],
                        callbacks=[lr_monitor, checkpoint_callback,early_stopping_callback],
                        gradient_clip_val=1)

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)