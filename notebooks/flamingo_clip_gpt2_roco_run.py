# Import comet_ml at the top of your file
import comet_ml
import os
#os.environ['CUDA_VISIBLE_DEVICES'] =6
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

from src.utils.utils import load_config
if __name__ == '__main__':
    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    seed_everything(42, workers=True)




    config = load_config('configs', 'config.yaml')

    
    #MIMIC CXR  Mean and Std of the dataset
    #mean: tensor([0.4719, 0.4719, 0.4719])
    #std:  tensor([0.3029, 0.3029, 0.3029])
    

    augmentations = {'train':
        T.Compose(
        [
            T.Resize((config['train']['data_augmentation']['resize_size'])),
            T.RandomRotation(config['train']['data_augmentation']['random_rotation']),
            T.ToTensor(),
            #T.Normalize(mean=(0.3570, 0.3621, 0.3636), std=(0.2924, 0.2941, 0.2951))
        ]),
        'validation':
        T.Compose(
        [   T.Resize((224, 224)),
            T.RandomRotation(10),
            T.ToTensor(),
            #T.Normalize(mean=(0.3570, 0.3621, 0.3636), std=(0.2924, 0.2941, 0.2951))
        ]),
        'test':
        T.Compose(
        [
            T.Resize((224, 224)),
            T.RandomRotation(10),
            T.ToTensor(),
            #T.Normalize(mean=(0.3570, 0.3621, 0.3636), std=(0.2924, 0.2941, 0.2951))
        ])
    }


    # Hyperparameters
    NUM_DATA_WORKERS  = config['dataset']['num_workers']
    ONLY_IMAGES = config['dataset']['only_images']
    BATCH_SIZE = config['train']['batch_size']
    NUM_EPOCHS = config['train']['epochs']
    LIMIT_NUM_SAMPLES = config['dataset']['limit_num_samples']




    IMAGE_TYPE = "jpg"
    TOKENIZER  = "gpt2"
    PREPROCESSED = True



    dataset_hyperparameters = {
        "root": ROOT,
        "batch_size": BATCH_SIZE,
        "tokenizer": "gpt2",
        "num_data_workers": NUM_DATA_WORKERS,
        "return_size": False,
        "augmentations": augmentations,
        "limit_num_samples": None,
        "token_max_len": 128
        
    }

    roco_datamodule = ROCODataModule(**dataset_hyperparameters)


    train_loader = roco_datamodule.train_dataloader()
    val_loader = roco_datamodule.val_dataloader()

    print("Len training dataset : ", len(roco_datamodule.train_dataset),
        "Batch Size : ", BATCH_SIZE, "NUM_EPOCHS : ",NUM_EPOCHS )
    print("Total training steps : ", len(roco_datamodule.train_dataset)//BATCH_SIZE*NUM_EPOCHS)


    # MODEL HPRAMS
    VOCAB_SIZE_OF_TOKENIZER = 50257 # mimic_datamodule.train_dataset.tokenizer.vocab_size
    LANGUAGE_MODEL = 'gpt2'
    NUM_TOKENS = VOCAB_SIZE_OF_TOKENIZER +4 if LANGUAGE_MODEL=="gpt2" else 31092
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
    CLASSIFICATION_MODE = False 
    NUM_CLASSES = 332
    FLAMINGO_MODE = False
    LABEL_SMOOTHING = 0.0
    GRADIENT_CLIP_VAL = 0




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
        'classification_mode': CLASSIFICATION_MODE,
        'classification_num_classes': NUM_CLASSES,  # 332 if DATASET=="IMAGECLEF"
        'flamingo_mode': FLAMINGO_MODE,
        "label_smoothing": LABEL_SMOOTHING
    }


    model = FlamingoModule(**hyperparams)


    lr_monitor = LearningRateMonitor(logging_interval='step')  
    checkpoint_callback = ModelCheckpoint(
                filename='{epoch}-{val_loss:.2f}-{other_metric:.2f}',
                    monitor= 'val_total_loss_epoch',
                        save_top_k = 5)
    
    trainer = pl.Trainer(max_epochs=NUM_EPOCHS,
                        accelerator=ACCELERATOR, devices=DEVICES,
                        callbacks=[lr_monitor, checkpoint_callback],
                        )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
