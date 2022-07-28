import json
import glob
import os
import numpy as np
import torch
import torchvision.transforms as T
import pytorch_lightning as pl 

from torch.utils.data import DataLoader, Dataset
from fastai.medical.imaging import get_dicom_files
from sklearn import model_selection
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,PreTrainedTokenizerFast, GPT2Tokenizer
from PIL import Image
from tqdm import tqdm as tqdm
import pandas as pd


class MIMICCXR(Dataset):
    """ MIMIC CXR Dataset"""
    def __init__(self, mimic_root, valid_indices, only_first_image=True, only_images=False,
                tokenizer='scibert', tokenizer_add_special_tokens=True, token_max_len=256,
                return_pil=False, transforms=None, image_type='dcm', preprocessed=True):
        
        self.mimic_root = mimic_root
        self.valid_indices = valid_indices
        self.only_first_image = only_first_image
        self.only_images = only_images
        self.token_max_len = token_max_len
        self.return_pil = return_pil
        self.transforms = transforms
        self.tokenizer_type ='gpt2'
        self.image_type  = image_type
        self.preprocessed = preprocessed

        #if self.image_type == 'jpg':
           # self.jpeg = TurboJPEG()
        #else:
        self.jpeg = None
        
        if tokenizer == "sciFive":
            self.tokenizer = AutoTokenizer.from_pretrained("razent/SciFive-large-Pubmed_PMC-MedNLI")
        elif tokenizer == "scibert":
            self.tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        elif tokenizer == "gpt2":
            
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        elif tokenizer == "scratch":
            # TODO : Implement a scratch tokenizer
            pass
        elif tokenizer is None:
            self.tokenizer = None

        if tokenizer_add_special_tokens:
            special_tokens_dict = {'additional_special_tokens': ['<image>', '<EOC>']}
            # Set the beginning of sentence token to <BOS>
            #self.tokenizer.bos_token = '<BOS>'
            num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)


    def __len__(self):
        return len(self.valid_indices)
  
    def __getitem__(self, idx):
        sample = self.valid_indices.iloc[idx]
        
        if self.preprocessed:
            img_path = sample['img_path'].replace("mimic-cxr-jpg","mimic-cxr-jpg-resized")
            report = sample['report']
        else:
            folder_path = sample['folder_path']
            report = sample['report']
        images = []

        if self.image_type == 'jpg':
            
            img = Image.open(os.path.join(self.mimic_root,img_path)).convert('RGB')
            if self.transforms is not None:
                images = self.transforms(img)
            else:
                if not self.return_pil:
                    images = T.transforms.ToTensor()(img)
            
            
                        
        # Put image at the beginning of the explanation
        report = self.tokenizer.bos_token + ' ' + '<image> ' + 'Output: ' + report + ' <EOC>'

        encoding = self.tokenizer.encode_plus(report, padding='max_length', truncation=True, max_length=self.token_max_len, return_tensors="pt")

        if self.tokenizer_type == 'gpt2':

            input_ids = encoding['input_ids']
            token_type_ids = encoding['attention_mask']
        else:
            input_ids = encoding['input_ids']
            token_type_ids = encoding['token_type_ids']

        eoc_token_id = self.tokenizer.all_special_ids[self.tokenizer.all_special_tokens.index('<EOC>')]
        pad_token_id = self.tokenizer.pad_token_id
        targets = torch.cat( ( input_ids[:,1:], torch.tensor([pad_token_id]).unsqueeze(1) ), dim=1)

        return {"image":images, "text": report, "input_ids": input_ids, "token_type_ids": token_type_ids, "targets" : targets}
        

                


class MIMICCXRRachetDataModule(pl.LightningDataModule):
    def __init__(self, mimic_root:str, batch_size: int = 32, transforms=None,
                 return_pil=False, limit_num_samples=None, num_data_workers=4,
                 tokenizer='scibert', image_type="jpg", split_path=None, preprocessed = True):
        super().__init__()
        self.mimic_root = mimic_root
        self.batch_size = batch_size
        self.transforms = transforms
        self.limit_num_samples = limit_num_samples
        self.return_pil = return_pil
        self.num_data_workers = num_data_workers
        self.tokenizer = tokenizer
        self.image_type = image_type
        self.split_path = split_path
        self.preprocessed = preprocessed

        self.setup()

    def setup(self):
        # Reads the Library and creates the paths to the images

    
        #all_datapoints_path = self.split_path + "mimic_cxr_all.json"
        train_split_path = self.split_path + 'MIMIC_AP_PA_train.csv'
        val_split_path = self.split_path + 'MIMIC_AP_PA_validate.csv'
        test_split_path = self.split_path + 'MIMIC_AP_PA_test.csv'

        self.train_split = pd.read_csv(train_split_path)
        self.val_split = pd.read_csv(val_split_path)
        self.test_split = pd.read_csv(test_split_path)
            

           
        # Limit all predefined paths to the number of limited samples
        if self.limit_num_samples is not None:
            self.train_split = self.train_split[:self.limit_num_samples]
            self.val_split = self.val_split[:self.limit_num_samples]
            self.test_split = self.test_split[:self.limit_num_samples]

        self.train_dataset = MIMICCXR(self.mimic_root, self.train_split, transforms=self.transforms["train"],
                                     tokenizer=self.tokenizer, image_type=self.image_type,
                                     preprocessed = self.preprocessed)
        self.validation_dataset = MIMICCXR(self.mimic_root,self.val_split, transforms=self.transforms["val"],
                                           tokenizer=self.tokenizer, image_type=self.image_type,
                                           preprocessed = self.preprocessed)
        self.test_dataset = MIMICCXR(self.mimic_root,self.test_split, tokenizer=self.tokenizer,
                                    image_type=self.image_type, preprocessed = self.preprocessed
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          num_workers=self.num_data_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size,
                          num_workers=self.num_data_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          num_workers=self.num_data_workers, pin_memory=True)

