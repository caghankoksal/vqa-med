import torch
import matplotlib.pyplot as plt
import json
import os 
import pickle as pkl
import pytorch_lightning as pl 
import numpy as np

from glob import glob
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from pathlib import Path
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,PreTrainedTokenizerFast, GPT2Tokenizer
from sklearn import model_selection
from turbojpeg import TurboJPEG

## NOTE: The VQA RAD dataset has 107 unique chest xray images and ~700 QAs

class VQARadDataset(Dataset):
    def __init__(self, root, mode='train', samples=None, transform=None,
                tokenizer='scibert', question_tokenize=None, answer_tokenize=None, tokenizer_add_special_tokens=True, 
                token_max_length=256, return_pil=False, preprocessed=True, load_in_memory=False):
        self.root = root
        self.transform = transform
        self.samples = samples
        self.load_in_memory = load_in_memory
        self.jpeg = TurboJPEG()
        self.mode = mode


        if load_in_memory:
            # load all images in the folder
            self.data_images = []
            print(f'Loading all images into memory...')

            files = glob(root + '/*.jpg')
            print(f'Found {len(files)} chest xrays in folder')

            for file in files:
                in_file = open(file, 'rb')
                img = self.jpeg.decode(in_file.read())
                np.moveaxis(img,-1,0)                       # make it (3,224,224)
                in_file.close()
                self.data_images.append({f'{os.path.basename(file)}': img})


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
        return len(self.samples)

    def __getitem__(self, idx):
        cur_sample = self.samples[idx]

        image_name = cur_sample["image_name"]
        answer = cur_sample["answer"]
        answer_type = cur_sample["answer_type"]
        question = cur_sample["question"]
        question_type = cur_sample["question_type"]


        if self.load_in_memory:
            img = self.data_images[image_name]
        else:
            in_file = open(cur_sample['image_path'], 'rb')
            img = self.jpeg.decode(in_file.read())
            np.moveaxis(img,-1,0)                       # make it (3,224,224)
            in_file.close()

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        # Put image at the beginning of the question
        text = self.tokenizer.bos_token + ' ' + '<image> ' + 'Question: ' + question + ' Answer: ' + answer + ' <EOC>'

        encoding = self.tokenizer.encode_plus(text, padding='max_length', truncation=True, max_length=self.token_max_len, return_tensors="pt")

        if self.tokenizer_type == 'gpt2':
            input_ids = encoding['input_ids']
            token_type_ids = encoding['attention_mask']
        else:
            input_ids = encoding['input_ids']
            token_type_ids = encoding['token_type_ids']

        eoc_token_id = self.tokenizer.all_special_ids[self.tokenizer.all_special_tokens.index('<EOC>')]
        pad_token_id = self.tokenizer.pad_token_id
        targets = torch.cat( ( input_ids[:,1:], torch.tensor([pad_token_id]).unsqueeze(1) ), dim=1)

        if self.mode == "test":
            return {"image":img, "question": question, "qa_pair": text, "input_ids": input_ids, "token_type_ids": token_type_ids, "targets" : targets}
        else:
            return {"image":img, "question": question, "answer": answer, "qa_pair": text, "input_ids": input_ids, "token_type_ids": token_type_ids, "targets" : targets}


class VQRadDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32, transforms=None, root='/home/mlmi-matthias/Data/VQA_RAD_preprocessed',
                limit_num_samples=None, num_workers=8, shuffle=True,
                tokenizer='scibert', preprocessed=True, load_in_memory=False):
        
        super().__init__()

        self.batch_size = batch_size
        self.transforms = transforms
        self.root = root
        self.limit_num_samples = limit_num_samples
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.tokenizer = tokenizer
        self.preprocessed = preprocessed
        self.load_in_memory = load_in_memory

        # read preprocessed QAs and images
        with open(root + 'vqa_rad_chest_paths.pkl', 'rb') as f:
            self.sample_dicts = pkl.load(f)
        
        # only use 90% for train-val, 10% is always test
        # from which train is 80% and val is 20%
        train_test_split = int(0.9 * len(self.sample_dicts))
        self.train_split, self.val_split = model_selection.train_test_split(self.sample_dicts[:train_test_split], test_size=0.2, shuffle=self.shuffle)
        self.test_split = self.sample_dicts[train_test_split:]

        # Limit all predefined paths to the number of limited samples
        if self.limit_num_samples is not None:
            self.train_split = self.train_split[:self.limit_num_samples]
            self.val_split = self.val_split[:self.limit_num_samples]
            self.test_split = self.test_split[:self.limit_num_samples]

        self.train_dataset = VQARadDataset(self.root, 'train', self.train_split, transform=self.transforms["train"],
                                    tokenizer=self.tokenizer, preprocessed = self.preprocessed, load_in_memory = self.load_in_memory)

        self.validation_dataset = VQARadDataset(self.root, 'val', self.val_split, transform=self.transforms["val"],
                                    tokenizer=self.tokenizer, preprocessed = self.preprocessed, load_in_memory = self.load_in_memory)

        self.test_dataset = VQARadDataset(self.root, 'test', self.test_split, transform=self.transforms["val"], 
                                    tokenizer=self.tokenizer, preprocessed = self.preprocessed, load_in_memory = self.load_in_memory)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=True)