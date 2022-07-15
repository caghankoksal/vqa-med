
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


from turbojpeg import TurboJPEG
import json

class RAD_CLEF_Mixed_DataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 32, transforms=None, tokenizer:str ='gpt2',
                 return_size = False, num_data_workers=0, limit_num_samples=None):
        super().__init__()
        self.batch_size = batch_size
        if transforms is None:
            transforms = {"train": None, "val": None, "test": None}
        self.transforms = transforms
        self.return_size = return_size
        self.num_data_workers = num_data_workers
        self.limit_num_samples = limit_num_samples
        if tokenizer != "gpt2":
            raise Exception("only gpt2 is supported here")

        self.setup()

    def setup(self):
        special_tokens_dict = {'additional_special_tokens': ['<image>', '<EOC>']}
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokenizer.add_special_tokens(special_tokens_dict)

        CLEF_DATASET_ROOT = '/home/mlmi-matthias/imageclef/'
        self.train_dataset_CLEF = MedLTDataset(  
            root=CLEF_DATASET_ROOT,
            tokenizer=self.tokenizer,
            mode='train',
            transform=self.transforms['train'],
            limit_num_samples=self.limit_num_samples
        )

        print("CLEF data length", len(self.train_dataset_CLEF))
        
        self.val_dataset_CLEF = MedLTDataset(
            root=CLEF_DATASET_ROOT,
            tokenizer=self.tokenizer,
            mode='val',
            transform=self.transforms['val'],
            limit_num_samples=self.limit_num_samples
        )

        self.test_dataset_CLEF = MedLTDataset(
            root=CLEF_DATASET_ROOT,
            tokenizer=self.tokenizer,
            mode='test',
            transform=self.transforms['test'],
            limit_num_samples=self.limit_num_samples
        )


        RAD_DATASET_ROOT = '/home/mlmi-matthias/VQA-RAD/'
        self.train_dataset_RAD = VQARadDataset(
            root=RAD_DATASET_ROOT,
            tokenizer=self.tokenizer,
            mode='train',
            samples=None,
            transform=self.transforms["train"],
            preprocessed = None,
            load_in_memory = False
        )

        print("RAD data length", len(self.train_dataset_RAD))

        self.val_dataset_RAD = VQARadDataset(
            root=RAD_DATASET_ROOT,
            tokenizer=self.tokenizer,
            mode='val',
            samples=None,
            transform=self.transforms["val"],
            preprocessed = None,
            load_in_memory = False
        )

        self.test_dataset_RAD = VQARadDataset(
            root=RAD_DATASET_ROOT,
            tokenizer=self.tokenizer,
            mode='test',
            samples=None,
            transform=self.transforms["test"],
            preprocessed = None,
            load_in_memory = False
        )

        self.train_dataset = RAD_CLEF_Mixed_Dataset(self.train_dataset_RAD, self.train_dataset_CLEF, self.limit_num_samples)
        self.val_dataset = RAD_CLEF_Mixed_Dataset(self.val_dataset_RAD, self.val_dataset_CLEF, self.limit_num_samples)
        self.test_dataset = RAD_CLEF_Mixed_Dataset(self.test_dataset_RAD, self.test_dataset_CLEF, self.limit_num_samples)

        # print(self.train_dataset[0]["image"].size())
        # print(self.train_dataset[1]["image"].size())
        # print(self.train_dataset[2]["image"].size())
        # print(self.train_dataset[3]["image"].size())
        # print("############################")
        # print(self.train_dataset[0]["question"])
        # print(self.train_dataset[1]["question"])
        # print("############################")
        # print(self.train_dataset[0]["answer"])
        # print(self.train_dataset[1]["answer"])
        # print("############################")
        # print(self.train_dataset[0]["qa_pair"])
        # print(self.train_dataset[1]["qa_pair"])
        # print("############################")
        # print(self.train_dataset[0]["input_ids"].size())
        # print(self.train_dataset[1]["input_ids"].size())
        # print("############################")
        # print(self.train_dataset[0]["token_type_ids"].size())
        # print(self.train_dataset[1]["token_type_ids"].size())
        # print("############################")
        # print(self.train_dataset[0]["targets"].size())
        # print(self.train_dataset[1]["targets"].size())
        # print("############################")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          num_workers=self.num_data_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_data_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          num_workers=self.num_data_workers, pin_memory=True)



class RAD_CLEF_Mixed_Dataset(Dataset):
    def __init__(self, RAD_Dataset, CLEF_Dataset, limit_num_samples):
        self.RAD_Dataset = RAD_Dataset
        self.CLEF_Dataset = CLEF_Dataset
        self.limit_num_samples = limit_num_samples
        self.RAD_chosen = set()
        self.CLEF_chosen = set()

    def __getitem__(self, item):
        if item >= self.__len__():
            raise IndexError("list bigger then len")
        if item < 0:
            raise IndexError("list index smaller then zero")
        
        if item % 2 == 0:
            return self.RAD_Dataset.__getitem__(item // 2 % len(self.RAD_Dataset))
        else:
            return self.CLEF_Dataset.__getitem__(item // 2 % len(self.CLEF_Dataset))


    def __len__(self):
        # since we do balancing of the two datasets, each dataset is drawn from the same amount of times
        balancedNumber = max(self.RAD_Dataset.__len__(), self.CLEF_Dataset.__len__()) * 2
        if self.limit_num_samples:
            return min(balancedNumber, self.limit_num_samples)
        else:
            return balancedNumber







class VQARadDataset(Dataset):
    def __init__(self, root, tokenizer, mode='train', samples=None, transform=None,
                question_tokenize=None, answer_tokenize=None, tokenizer_add_special_tokens=True, 
                token_max_length=256, return_pil=False, preprocessed=True, load_in_memory=False):
        self.root = root
        self.transform = transform
        self.load_in_memory = load_in_memory
        self.jpeg = TurboJPEG()
        self.mode = mode
        self.tokenizer_type ='gpt2'
        self.token_max_len = token_max_length
        self.tokenizer = tokenizer

        self.allannotations = json.load(open(os.path.join(root, "VQA-RAD_public.json")))

        trainCut, valCut = round(len(self.allannotations) * 0.7), round(len(self.allannotations) * 0.9)
        if mode == "train":
            self.annotations = self.allannotations[: trainCut]
        elif mode == "val":
            self.annotations = self.allannotations[trainCut : valCut]
        elif mode == "test":
            self.annotations = self.allannotations[trainCut :]
        else:
            raise Exception("either train, val or test")



    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        cur_sample = self.annotations[idx]

        image_name = cur_sample["image_name"]
        answer = cur_sample["answer"]
        answer_type = cur_sample["answer_type"]
        question = cur_sample["question"]
        question_type = cur_sample["question_type"]


        if self.load_in_memory:
            img = self.data_images[image_name]
        else:
            in_file = open(os.path.join(self.root, "images/", cur_sample['image_name']), 'rb')
            img = self.jpeg.decode(in_file.read())
            img = Image.fromarray(img)
            in_file.close()

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = T.ToTensor()(img)

        # Put image at the beginning of the question
        if self.mode == "test":
            text = self.tokenizer.bos_token + ' ' + '<image> ' + 'Question: ' + str(question) + ' Answer: '
        else:
            text = self.tokenizer.bos_token + ' ' + '<image> ' + 'Question: ' + str(question) + ' Answer: ' + str(answer) + ' <EOC>'

        input_encoding = self.tokenizer.encode_plus(text, padding='max_length', truncation=True, max_length=self.token_max_len, return_tensors="pt")

        if self.tokenizer_type == 'gpt2':
            input_ids = input_encoding['input_ids']
            token_type_ids = input_encoding['attention_mask']
        else:
            input_ids = input_encoding['input_ids']
            token_type_ids = input_encoding['token_type_ids']

        eoc_token_id = self.tokenizer.all_special_ids[self.tokenizer.all_special_tokens.index('<EOC>')]
        pad_token_id = self.tokenizer.pad_token_id

        # TODO create target without answer -> see what happens

        targets = torch.cat( ( input_ids[:,1:], torch.tensor([pad_token_id]).unsqueeze(1) ), dim=1)

        # if self.mode == "test":
        #     return {"image":img, "question": question, "answer": answer, "qa_pair": text, "input_ids": input_ids, "token_type_ids": token_type_ids, "targets" : targets}
        # else:
        return {"image":img, "question": question, "answer": answer, "qa_pair": text, "input_ids": input_ids, "token_type_ids": token_type_ids, "targets" : targets}








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
    def __init__(self, tokenizer, root = 'data/', path='test2021', mode='train', transform=None,
                 return_size=False, token_max_len=256,
                 limit_num_samples=None):
        self.mode = mode
        self.return_size = return_size
        self.path = path
        self.root = root
        self.limit_num_samples = limit_num_samples
        self.token_max_len = token_max_len
        self.tokenizer_type ='gpt2'

        if mode == 'train':
            imgs = make_dataset(root+mode)
        elif mode == 'val':
            imgs = make_dataset(root+mode)
        elif mode == 'test':
            imgs = make_testset(root)

        self.imgs = imgs
        self.transform = transform
        self.tokenizer = tokenizer


        if self.limit_num_samples:
            self.imgs = self.imgs[:self.limit_num_samples]


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
        else:
            sample["image"] = T.ToTensor()(sample["image"])


        if self.return_size:
            sample['size'] = torch.tensor(size)

        
        # Put image at the beginning of the explanation
        question_answer_pair = self.tokenizer.bos_token + ' ' + '<image> ' + 'question: ' + str(sample["question"]) +\
                                ' answer: ' + str(sample["answer"]) + ' <EOC>'
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


        sample["qa_pair"] = question_answer_pair
        sample["input_ids"] = input_ids
        sample["token_type_ids"] = token_type_ids
        sample["targets"] = targets
        # sample['ID'] = image_name # the other dataset also does not give an ID field
        return sample

    def __len__(self):
        return len(self.imgs)
