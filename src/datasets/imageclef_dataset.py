
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset
import pytorch_lightning as pl
from PIL import Image
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,PreTrainedTokenizerFast, GPT2Tokenizer
from PIL import Image
from tqdm import tqdm as tqdm
import torchvision.transforms as T


def make_dataset(root):
    imgs = []

    with open(root + '/label.txt', 'r') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        name, question, answer = line.strip().split('|')       
        imgs.append([name, question, answer])

    return imgs


def make_testset(root, mode='test2021'):
    imgs = []
    with open(root + '/' + mode + '/Task1-VQA-2021-TestSet-Questions.txt', 'r') as f:
        lines = f.readlines()

    for line in lines:
        name, question = line.strip().split('|')
        imgs.append([name, question])
    return imgs


class MedLTDataset(Dataset):
    def __init__(self,root = 'data/', path='test2021', mode='train', transform=None,
                 return_size=False,tokenizer='scibert', tokenizer_add_special_tokens=True,
                 token_max_len=128, limit_num_samples=None, tokenizer_type='gpt2'):
        self.mode = mode
        self.transform = transform
        self.return_size = return_size
        self.path = path
        self.root = root
        self.tokenizer_type = tokenizer_type
        self.limit_num_samples = limit_num_samples

        if mode == 'train':
            imgs = make_dataset(root+mode)
        elif mode == 'val':
            imgs = make_dataset(root+mode)
        elif mode == 'test':
            imgs = make_testset(root)

        self.imgs = imgs
        self.transform = transform
        self.return_size = return_size
        self.tokenizer = tokenizer
        self.tokenizer_add_special_tokens = tokenizer_add_special_tokens
        self.token_max_len = token_max_len


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


        if self.limit_num_samples:
            imgs = imgs[:self.limit_num_samples]


    def __getitem__(self, item):
        if  self.mode != 'test':
            image_name, question, answer = self.imgs[item]
            image_path = self.root + self.mode + '/images/' + image_name + '.jpg'
        else:
            image_name, question = self.imgs[item]
            image_path = self.root + self.path + '/images/' + image_name + '.jpg'
        
        assert os.path.exists(image_path), ('{} does not exist'.format(image_path))

        image = Image.open(image_path).convert('RGB')

        w, h = image.size
        size = (h, w)
        if not self.mode == 'test':
            sample = {'image': image, 'question': question, 'answer': answer}
        else:
            sample = {'image': image, 'question': question}

        if self.transform:
            sample["image"] = self.transform(sample["image"])

        if self.return_size:
            sample['size'] = torch.tensor(size)

        
        # Put image at the beginning of the explanation
        question_answer_pair = self.tokenizer.bos_token + ' ' + '<image> ' + 'question: ' + sample["question"] +\
                                ' answer: '+sample["answer"] + ' <EOC>'
        encoding = self.tokenizer.encode_plus(question_answer_pair, padding='max_length', truncation=True,
                                              max_length=self.token_max_len, return_tensors="pt")

        if self.tokenizer_type == 'gpt2':
            input_ids = encoding['input_ids']
            token_type_ids = encoding['attention_mask']
        else:
            input_ids = encoding['input_ids']
            token_type_ids = encoding['token_type_ids']

        #eoc_token_id = self.tokenizer.all_special_ids[self.tokenizer.all_special_tokens.index('<EOC>')]
        pad_token_id = self.tokenizer.pad_token_id
        targets = torch.cat( ( input_ids[:,1:], torch.tensor([pad_token_id]).unsqueeze(1) ), dim=1)


        sample["input_ids"] = input_ids
        sample["token_type_ids"] = token_type_ids
        sample["targets"] = targets
        sample['ID'] = image_name
        sample["qa_pair"] = question_answer_pair
        return sample

    def __len__(self):
        return len(self.imgs)


class ImageCLEF2021DataModule(pl.LightningDataModule):
    def __init__(self, root='data', batch_size: int = 32, transforms=None, tokenizer:str ='gpt2',
                 return_size = False, num_data_workers=0, limit_num_samples=None):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.transforms = transforms
        self.tokenizer = tokenizer
        self.return_size = return_size
        self.num_data_workers = num_data_workers
        self.limit_num_samples = limit_num_samples

        self.setup()

    def setup(self):
        self.train_dataset = MedLTDataset(root=self.root, mode='train',
                                          transform=self.transforms['train'],
                                          tokenizer=self.tokenizer,
                                          return_size=self.return_size,
                                          limit_num_samples=self.limit_num_samples)
        self.val_dataset = MedLTDataset(root=self.root,mode='val',
                                        transform=self.transforms['val'],
                                        tokenizer=self.tokenizer, return_size=self.return_size,
                                        limit_num_samples=self.limit_num_samples)
        self.test_dataset = MedLTDataset(root=self.root, mode='test',
                                         transform=self.transforms['test'],
                                         tokenizer=self.tokenizer, return_size=self.return_size,
                                        limit_num_samples=self.limit_num_samples)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          num_workers=self.num_data_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_data_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          num_workers=self.num_data_workers, pin_memory=True)
