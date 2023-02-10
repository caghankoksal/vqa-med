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
import _pickle as cPickle
import src.datasets.utils as utils
## NOTE: The VQA RAD dataset has 107 unique chest xray images and ~700 QAs

COUNTING_ONLY = False
def is_howmany(q, a, label2ans):
    if 'how many' in q.lower() or \
       ('number of' in q.lower() and 'number of the' not in q.lower()) or \
       'amount of' in q.lower() or \
       'count of' in q.lower():
        if a is None or answer_filter(a, label2ans):
            return True
        else:
            return False
    else:
        return False

def answer_filter(answers, label2ans, max_num=10):
    for ans in answers['labels']:
        if label2ans[ans].isdigit() and max_num >= int(label2ans[ans]):
            return True
    return False


def _create_entry(img, data, answer):
    if None!=answer:
        answer.pop('image_name')
        answer.pop('qid')
    entry = {
        'qid' : data['qid'],
        'image_name'    : data['image_name'],
        'image'       : img,
        'question'    : data['question'],
        'answer'      : answer,
        'answer_text' : data['answer'],
        'answer_type' : data['answer_type'],
        'question_type': data['question_type'],
        'phrase_type' : data['phrase_type']}
    return entry


def _load_dataset(dataroot, name, img_id2val, label2ans):
    """Load entries
    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val', 'test'
    """
    data_path = os.path.join(dataroot, name + 'set.json')
    samples = json.load(open(data_path))
    samples = sorted(samples, key=lambda x: x['qid'])
    print(f'Samples length  {name} :  {len(samples)}')

    answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
    answers = cPickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['qid'])
    print(f'answers length  {name} :  {len(answers)}')

    utils.assert_eq(len(samples), len(answers))
    entries = []
    for sample, answer in zip(samples, answers):
        utils.assert_eq(sample['qid'], answer['qid'])
        utils.assert_eq(sample['image_name'], answer['image_name'])
        img_id = sample['image_name']
        #if not COUNTING_ONLY or is_howmany(sample['question'], answer, label2ans):
        if not COUNTING_ONLY or is_howmany(sample['question'], answer, label2ans):
            entries.append(_create_entry(img_id2val[img_id], sample, answer))

    return entries

    
class VQARadDataset(Dataset):
    def __init__(self, root, ans2label, label2ans, mode='train', samples=None, 
                transform=None, tokenizer='scibert', tokenizer_add_special_tokens=True, 
                token_max_length=64, load_in_memory=False, return_idx_answer_eoc = True):
        self.root = root
        self.transform = transform
        self.samples = samples
        self.load_in_memory = load_in_memory
        self.mode = mode
        self.token_max_len = token_max_length
        self.return_idx_answer_eoc = return_idx_answer_eoc

        self.ans2label = ans2label
        self.label2ans = label2ans

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
        

        print('Num unique answers vqa-rad ', len(set(self.label2ans)))

        if tokenizer == "sciFive":
            self.tokenizer = AutoTokenizer.from_pretrained("razent/SciFive-large-Pubmed_PMC-MedNLI")
        elif tokenizer == "scibert":
            self.tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        elif tokenizer == "bert-clinical":
            self.tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        elif tokenizer == 'pubmedbert':
            self.tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext')
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

    def preprocess_question(self, sentence):
        sentence = sentence.lower()
        if "? -yes/no" in sentence:
            sentence = sentence.replace("? -yes/no", "")
        if "? -open" in sentence:
            sentence = sentence.replace("? -open", "")
        if "? - open" in sentence:
            sentence = sentence.replace("? - open", "")
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s').replace('...', '').replace('x ray', 'x-ray').replace('.', '')

        return sentence

    def __getitem__(self, idx):
        
        # Cur item is from medvqa repo processing 
        cur_item = self.samples[idx]
        image_name = cur_item["image_name"]
        answer = cur_item["answer"]
        question = cur_item["question"]        

        cur_sample = {}
        cur_sample['image_name'] = image_name
        cur_sample['question'] = question

        #Normalize question according to med vqa processing
        cur_sample['question'] = self.preprocess_question(cur_sample['question'])

        img = Image.open(os.path.join(self.root,'Images',cur_sample['image_name']))
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        
        # Put image at the beginning of the question
        if self.mode == "test":
            text = self.tokenizer.bos_token + ' ' + '<image> ' + 'Question: '  + question + '<EOQ>' + ' Answer: '  + ' <EOC>'
        else:
            # Here how the input is proccessed for training 
            cur_sample['label'] = answer['labels'][0]

            if self.tokenizer.name_or_path == 'gpt2':
                text = self.tokenizer.bos_token + ' ' + '<image> ' + 'question: ' + cur_sample["question"] +\
                                ' <EOQ>' +' answer: '+ self.label2ans[cur_sample['label']]  + ' <EOC>'
            # we are currently using bert 
            elif self.tokenizer.name_or_path == 'bert-base-uncased':
                text = '<image> ' + 'question: ' + cur_sample["question"] + ' <EOQ>' #+' answer: '+ self.label2ans[cur_sample['label']] + ' <EOC>'

            elif self.tokenizer.name_or_path == 'emilyalsentzer/Bio_ClinicalBERT':
                text = '<image> ' + 'question: ' + cur_sample["question"] + ' <EOQ>' #+' answer: '+ self.label2ans[cur_sample['label']] + ' <EOC>'

        input_encoding = self.tokenizer.encode_plus(text, padding='max_length', truncation=True, max_length=self.token_max_len, return_tensors="pt")

        if  self.tokenizer.name_or_path  == 'gpt2':
            input_ids = input_encoding['input_ids']
            token_type_ids = input_encoding['attention_mask']
        elif self.tokenizer.name_or_path == 'bert-base-uncased':
            input_ids = input_encoding['input_ids']
            token_type_ids = input_encoding['attention_mask']
        
        elif self.tokenizer.name_or_path == 'emilyalsentzer/Bio_ClinicalBERT':
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
            #ans_token_id = self.tokenizer.convert_tokens_to_ids(':')
            #index_of_answer = (input_ids==ans_token_id).nonzero()[0][-1].item()

             # # End of Question
            eoq_token_id = self.tokenizer.convert_tokens_to_ids('<EOQ>')
            index_of_eoq = (input_ids==eoq_token_id).nonzero()[0][-1].item()

            # End of Chunk
            # eoc_token_id = self.tokenizer.convert_tokens_to_ids('<EOC>')
            # index_of_eoc = (input_ids==eoc_token_id).nonzero()[0][-1].item()
           
            
            #cur_sample["index_answer"] = index_of_answer
            #cur_sample["index_eoc"] = index_of_eoc
            cur_sample["index_eoq"] = index_of_eoq

        
        #cur_sample.pop('answer')
        cur_sample.pop('image_name')


        # TODO create target without answer -> see what happens
        # For Autoregressive decoding 
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

        #self.train_split= []
        #self.val_split = []
        #for sample in self.sample_dicts:
        #    if sample['phrase_type'].startswith('test'):
        #        self.val_split.append(sample)
        #    else:
        #        self.train_split.append(sample)

        #self.ans2label =  json.load(open(self.answers2label_path))
        #self.label2ans = json.load(open(self.label2answer_path))
        #print(f'Test set has {len(self.val_split)} questions before removing')
        #self.val_split = [sample for sample in  self.val_split if  sample['answer'].lower().strip() in self.answer_to_label.keys() ]
        #print(f'Test set has {len(self.val_split)} questions after removing')


        self.ans2label = cPickle.load(open(self.answers2label_path, 'rb'))
        self.label2ans = cPickle.load(open(self.label2answer_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        dataroot = '/u/home/koksal/mlmi-vqa/data'
        # TODO: load img_id2idx
        self.img_id2idx = json.load(open(os.path.join(dataroot, 'imgid2idx.json')))

        self.train_split = _load_dataset(dataroot, 'train', self.img_id2idx, self.label2ans)
        self.val_split = _load_dataset(dataroot, 'test', self.img_id2idx, self.label2ans)

        print('Training split length ',len(self.train_split))
        print('Test split length ',len(self.val_split))

        # Eliminate Test set answers which are not in the test
        self.val_split = [each for each in self.val_split if len(each['answer']['labels'])!= 0]
        print('Test split length after removal ',len(self.val_split))


        # Official VQA RAD split does not have val split so currently just create workaround
        #self.test_split = self.val_split.copy()

        #print(f'There are {len(self.sample_dicts)} QA pairs in VQA-RAD dataset')
        #print(f'Training set has {len(self.train_split)} Test set has {len(self.val_split)} questions')

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

        self.train_dataset = VQARadDataset(self.root,  self.ans2label, self.label2ans, 
                                          mode='train', samples=self.train_split, transform=self.augmentations["train"],
                                          tokenizer=self.tokenizer, load_in_memory = self.load_in_memory)

        self.validation_dataset = VQARadDataset(self.root, self.ans2label, self.label2ans, 
                                                mode='val', samples=self.val_split, transform=self.augmentations["val"],
                                                tokenizer=self.tokenizer, load_in_memory = self.load_in_memory)

        self.test_dataset = VQARadDataset(self.root, self.ans2label, self.label2ans,
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