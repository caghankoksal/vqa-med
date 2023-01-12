import torch
import matplotlib.pyplot as plt
import json
import os 
import pickle as pkl
import pytorch_lightning as pl 
import numpy as np

import torchvision.transforms as T

from glob import glob
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from pathlib import Path
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,PreTrainedTokenizerFast, GPT2Tokenizer
from sklearn import model_selection
#from turbojpeg import TurboJPEG

## NOTE: The VQA RAD dataset has 107 unique chest xray images and ~700 QAs

class VQARadDataset(Dataset):
    def __init__(self, root, answers2label_path, label2answer_path, mode='train', samples=None, 
                transform=None, tokenizer='scibert', tokenizer_add_special_tokens=True, 
                token_max_length=256, load_in_memory=False, return_idx_answer_eoc = True):
        self.root = root
        self.transform = transform
        self.samples = samples
        self.load_in_memory = load_in_memory
        #self.jpeg = TurboJPEG()
        self.mode = mode
        #self.tokenizer_type ='gpt2'
        self.token_max_len = token_max_length
        self.return_idx_answer_eoc = return_idx_answer_eoc

        # # Create answers dictionary
        # answer_list = []
        # for sample in samples:
        #     answer_list.append(str(sample['answer']).strip().lower())
      
        # # Create a dictionary of answers
        # self.answer_to_label = {}
        # self.label_to_answer = {}
        # for i,ans in enumerate(set(answer_list)):
        #     self.answer_to_label[ans] = i
        #     self.label_to_answer[i] = ans
        
        #/u/home/koksal/mlmi-vqa/data/answer2label_vqarad.json
        self.answer_to_label =  json.load(open(answers2label_path))
        self.label_to_answer = json.load(open(label2answer_path))

        print('Num unique answers vqa-rad ', len(set(self.answer_to_label.keys())))


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
        elif tokenizer =='bert':
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            print('BERT Tokenizer is initalized')
        elif tokenizer is None:
            self.tokenizer = None

        if tokenizer_add_special_tokens:
            special_tokens_dict = {'additional_special_tokens': ['<image>', '<EOC>', '<EOQ>']}
            # Set the beginning of sentence token to <BOS>
            #self.tokenizer.bos_token = '<BOS>'
            num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        cur_item = self.samples[idx]
        
        image_name = cur_item["image_name"]
        answer = cur_item["answer"]
        answer_type = cur_item["answer_type"]
        question = cur_item["question"]
        question_type = cur_item["question_type"]


        cur_sample = {}
        cur_sample['answer'] = str(answer).strip().lower()
        cur_sample['image_name'] = image_name
        cur_sample['answer_type'] = answer_type
        cur_sample['question'] = question
        cur_sample['question_type'] = question_type
        

        if self.load_in_memory:
            img = self.data_images[image_name]
        else:
            img = Image.open(os.path.join(self.root,'Images',cur_sample['image_name']))

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        
        
        # Put image at the beginning of the question
        if self.mode == "test":
            text = self.tokenizer.bos_token + ' ' + '<image> ' + 'Question: '  + question + '<EOQ>' + ' Answer: '  + ' <EOC>'
        else:
            label = self.answer_to_label.get(str(answer).strip().lower())
            cur_sample['label'] = label
            #text = self.tokenizer.bos_token + ' ' + '<image> ' + 'Question: ' + '<EOQ>' + question + ' Answer: ' + str(answer) + ' <EOC>'
              # Put image at the beginning of the explanation

            if self.tokenizer.name_or_path == 'gpt2':
                text = self.tokenizer.bos_token + ' ' + '<image> ' + 'question: ' + cur_sample["question"] +\
                                ' <EOQ>' +' answer: '+cur_sample["answer"] + ' <EOC>'
            elif self.tokenizer.name_or_path == 'bert-base-uncased':
                text = '<image> ' + 'question: ' + cur_sample["question"] +\
                                ' <EOQ>' +' answer: '+cur_sample["answer"] + ' <EOC>'

        #print('text : ',text)
        input_encoding = self.tokenizer.encode_plus(text, padding='max_length', truncation=True, max_length=self.token_max_len, return_tensors="pt")

        if  self.tokenizer.name_or_path  == 'gpt2':
            input_ids = input_encoding['input_ids']
            token_type_ids = input_encoding['attention_mask']
        elif self.tokenizer.name_or_path == 'bert-base-uncased':
            input_ids = input_encoding['input_ids']
            token_type_ids = input_encoding['attention_mask']
        else:
            input_ids = input_encoding['input_ids']
            token_type_ids = input_encoding['token_type_ids']

        eoc_token_id = self.tokenizer.all_special_ids[self.tokenizer.all_special_tokens.index('<EOC>')]
        pad_token_id = self.tokenizer.pad_token_id

        if self.return_idx_answer_eoc:
            # 3280 is the index of the : token since we use Answer: 
            #index_of_answer = (input_ids==3280).nonzero()[0][-1].item()
            # Index of  ':' token which comes after answer: so we use its embedding for classifcation
            ans_token_id = self.tokenizer.convert_tokens_to_ids(':')
            index_of_answer = (input_ids==ans_token_id).nonzero()[0][-1].item()
            # End of Chunk
            eoc_token_id = self.tokenizer.convert_tokens_to_ids('<EOC>')
            index_of_eoc = (input_ids==eoc_token_id).nonzero()[0][-1].item()
            # # End of Question
            eoq_token_id = self.tokenizer.convert_tokens_to_ids('<EOQ>')
            index_of_eoq = (input_ids==eoq_token_id).nonzero()[0][-1].item()
            
            cur_sample["index_answer"] = index_of_answer
            cur_sample["index_eoc"] = index_of_eoc
            cur_sample["index_eoq"] = index_of_eoq



        # TODO create target without answer -> see what happens

        targets = torch.cat( ( input_ids[:,1:], torch.tensor([pad_token_id]).unsqueeze(1) ), dim=1)
        cur_sample["input_ids"] = input_ids
        cur_sample["token_type_ids"] = token_type_ids
        cur_sample["targets"] = targets
        cur_sample['ID'] = image_name
        cur_sample["qa_pair"] = text
        cur_sample['image'] = img


        return cur_sample 


class VQRadDataModule(pl.LightningDataModule):
    def __init__(self, args, augmentations):
        
        super().__init__()

        
        self.augmentations = augmentations
        self.batch_size = args["train"]['batch_size']
        self.root = args['dataset']['vqa_rad_path']
        self.answers2label_path = args['dataset']['answers2label_path']
        self.label2answer_path = args['dataset']['label2answer_path']
        self.limit_num_samples = args['dataset']['limit_num_samples']
        self.num_workers = args['dataset']['num_workers']
        self.shuffle = args['dataset']['shuffle']
        self.tokenizer = args['dataset']['tokenizer']
        self.load_in_memory = args['dataset']['load_in_memory']
        print('Load memory', self.load_in_memory)


        with open(os.path.join(self.root,'VQA-RAD_public.json'), 'r') as f:
            self.sample_dicts = json.load(f)

        self.train_split= []
        self.val_split = []
        for sample in self.sample_dicts:
            if sample['phrase_type'].startswith('test'):
                self.val_split.append(sample)
            else:
                self.train_split.append(sample)

        # Official VQA RAD split does not have val split so currently just create workaround
        #self.test_split = self.val_split.copy()

        print(f'There are {len(self.sample_dicts)} QA pairs in VQA-RAD dataset')
        print(f'Training set has {len(self.train_split)} Test set has {len(self.val_split)} questions')

        # only use 90% for train-val, 10% is always test
        # from which train is 80% and val is 20%
        #train_test_split = int(0.9 * len(self.sample_dicts))
        #self.train_split, self.val_split = model_selection.train_test_split(self.sample_dicts[:train_test_split], test_size=0.2, shuffle=self.shuffle)
        #self.test_split = self.sample_dicts[train_test_split:]

        # Limit all predefined paths to the number of limited samples
        if self.limit_num_samples is not None:
            self.train_split = self.train_split[:self.limit_num_samples]
            self.val_split = self.val_split[:self.limit_num_samples]
            self.test_split = self.val_split[:self.limit_num_samples]

        self.train_dataset = VQARadDataset(self.root,  self.answers2label_path, self.label2answer_path, 
                                          mode='train', samples=self.train_split, transform=self.augmentations["train"],
                                          tokenizer=self.tokenizer, load_in_memory = self.load_in_memory)

        self.validation_dataset = VQARadDataset(self.root, self.answers2label_path, self.label2answer_path, 
                                                mode='val', samples=self.val_split, transform=self.augmentations["val"],
                                                tokenizer=self.tokenizer, load_in_memory = self.load_in_memory)

        self.test_dataset = VQARadDataset(self.root, self.answers2label_path, self.label2answer_path,
                                         mode='test', samples=self.val_split, transform=self.augmentations["val"], 
                                         tokenizer=self.tokenizer, load_in_memory = self.load_in_memory)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, pin_memory=True)