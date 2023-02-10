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
import wandb

from src.datasets.imageclef_dataset import ImageCLEF2021DataModule


if __name__ == '__main__':
    seed_everything(42, workers=True)

    args = load_config('/u/home/koksal/mlmi-vqa/configs','config.yaml')
     


    augmentations = {'train':
        T.Compose(
        [   
            T.Resize((args['train']['augmentation']['resize_size'])),
            
            #T.RandomResizedCrop(224,scale=args['train']['augmentation']['random_resize_scale'],
            #                    ratio=args['train']['augmentation']['random_resize_ratio']),
            T.RandomRotation((args['train']['augmentation']['random_rotation'])),
            T.ColorJitter(brightness=args['train']['augmentation']['color_jitter']['brightness'],
                          contrast=args['train']['augmentation']['color_jitter']['contrast'],
                          saturation=args['train']['augmentation']['color_jitter']['saturation'],
                          hue=args['train']['augmentation']['color_jitter']['hue']),
            T.ToTensor(),
            T.Normalize(mean=args['dataset']['mean'], std=args['dataset']['std']),
        ]),
        'val':
        T.Compose(
        [
            T.Resize((args['test']['augmentation']['resize_size'])),
            T.RandomRotation((args['test']['augmentation']['random_rotation'])),
            T.ToTensor(),
            T.Normalize(mean=args['dataset']['mean'], std=args['dataset']['std'])
        ]),
        'test':
        T.Compose(
        [
            T.Resize((args['test']['augmentation']['resize_size'])),
            T.ToTensor(),
            T.Normalize(mean=args['dataset']['mean'], std=args['dataset']['std'])
        ])
    }

    wandb.init(project="flamingo-research", config=args)

    
    if args.dataset == 'vqarad':
        datamodule = VQRadDataModule(args, augmentations= augmentations)
    elif args.dataset == 'imageclef':
        
        datamodule = ImageCLEF2021DataModule(args, augmentations = augmentations)

    # Data Loaders
    train_loader = mimic_datamodule.train_dataloader()
    val_loader = mimic_datamodule.val_dataloader()

    model = FlamingoModule(args) 
    wandb.watch(model, log_freq=100)
    
    if args['pretrained']:
        print("Pretrained Flamingo Model is loaded from checkpoint : ",args['pretrained'])
        model.load_state_dict(torch.load(args['pretrained'])["state_dict"])


    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping

    
    lr_monitor = LearningRateMonitor(logging_interval='step')


    if args['model']['classification_mode']:
        checkpoint_callback = ModelCheckpoint(
                    filename='{epoch}-{val_acc_epoch:.2f}-{val_total_loss_epoch:.2f}-{val_loss_generation_epoch:.2f}-{val_classification_loss_epoch:.2f}',
                    monitor= 'val_acc_epoch',
                    save_top_k = 3,
                    save_last=True,
                    mode="max")

        #early_stopping_callback = EarlyStopping(monitor="val_acc_epoch", mode="max",patience=10)
        #early_stopping_callback = EarlyStopping(monitor="val_total_loss_epoch", mode="min",patience=5)
    else:
        checkpoint_callback = ModelCheckpoint(
                filename='{epoch}-{val_loss_generation_epoch:.2f}',
                monitor= 'val_loss_generation_epoch',
                save_top_k = 3, 
                save_last=True,
                mode="min")
        #early_stopping_callback = EarlyStopping(monitor="val_loss_generation_epoch", mode="min",patience=10)

    #early_stopping_callback = EarlyStopping(monitor="val_acc_epoch", mode="max",patience=10)

    # All our models are trained using the AdamW optimizer with global norm clipping of 1
    print(args['train']['devices'])
    trainer = pl.Trainer(max_epochs=args['train']['num_epochs'],
                        accelerator=args['train']['accelerator'], devices=args['train']['devices'],
                        callbacks=[lr_monitor, checkpoint_callback, ], #early_stopping_callback],
                        gradient_clip_val=args['optimizer']['gradient_clip'])

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)