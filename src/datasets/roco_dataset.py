import json
import os 
import unicodedata
from torch.utils.data import DataLoader,Dataset
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import transforms
from typing import Callable, Optional
import torch
import pytorch_lightning as pl
from PIL import Image

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,PreTrainedTokenizerFast, GPT2Tokenizer
from tqdm import tqdm as tqdm
import torchvision.transforms as T

class RocoDataset(Dataset):
    """
    Dtaset for loading image-text data for tasks like CLIP training, Image Captioning.
    Args:
        root: (string): The root path where the dataset is stored
        mode: (string): The mode of the dataset
        token_max_len: (int): The maximum length of the token
        input_size: (int): The size of the input image
        transform: (callable): A function/transform that takes in an PIL image and returns a transformed version
        tokenizer: (string): The tokenizer to use
        limit_num_samples: (int): The maximum number of samples to load
        tokenizer_type: (string): The type of tokenizer to use
        tokenizer_add_special_tokens: (bool): Whether to add special tokens to the tokenizer


        Mean and Std: 
        mean: tensor([0.3570, 0.3621, 0.3636])
        std:  tensor([0.2924, 0.2941, 0.2951])
    """


    def __init__(
        self,
        mode: str,
        root="",
        token_max_len=128,
        input_size = 224,
        transform: Optional[Callable] = None,
        tokenizer: str = "scibert",
        limit_num_samples: Optional[int] = None,
        tokenizer_type: str = "gpt2",
        tokenizer_add_special_tokens=True,
    ):
        super().__init__()
        self.input_size = input_size
        self.mode = mode
        self.transform = transform
        self.limit_num_samples = limit_num_samples
        self.tokenizer_type = tokenizer_type
        self.tokenizer_add_special_tokens = tokenizer_add_special_tokens


        if self.mode == "train":
            file_path = os.path.join(root, mode, mode + '.json')
        elif self.mode == "eval" or self.mode == "validation":
            file_path = os.path.join(root, mode, 'validation.json')

        elif self.mode == "test":
            file_path = os.path.join(root, mode, mode + '.json')
        else:
            raise ValueError(f"{mode} dataset is not supported!")

        with open(file_path, 'r') as f:
            self.data = json.load(f)


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
            self.data = self.data[:self.limit_num_samples]


    def _load_image(self, idx: int):
        path = self.data[idx]['image_path']
        try:
            image = Image.open(path).convert("RGB")
            return image
        except FileNotFoundError:
            print(f"No image at {path} exists!")


    def _load_target(self, idx):
        return self.data[idx]["caption"]

    def __getitem__(self, index: int):
        image = self._load_image(index)
        caption = self._load_target(index)
        image = self.transform(image)

        # Put image at the beginning of the explanation
        report = self.tokenizer.bos_token + ' ' + '<image> ' + 'Output: ' + caption + ' <EOC>'
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

        return {"image":image, "text": report, "input_ids": input_ids, "token_type_ids": token_type_ids, "targets" : targets}

    def __len__(self) -> int:
        return len(self.data)



class ROCODataModule(pl.LightningDataModule):
    def __init__(self, root='data', batch_size: int = 32, augmentations=None, tokenizer:str ='gpt2',
                 return_size = False, num_data_workers=0, limit_num_samples=None, token_max_len=256):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.transforms = augmentations
        self.tokenizer = tokenizer
        self.return_size = return_size
        self.num_data_workers = num_data_workers
        self.limit_num_samples = limit_num_samples
        self.token_max_len = token_max_len

        self.setup()

    def setup(self):
        self.train_dataset = RocoDataset(root=self.root, mode='train',
                                          transform=self.transforms['train'],
                                          tokenizer=self.tokenizer,
                                          limit_num_samples=self.limit_num_samples,
                                          tokenizer_type=self.tokenizer,
                                          tokenizer_add_special_tokens=True,
                                          token_max_len=self.token_max_len)


        self.val_dataset = RocoDataset(root=self.root,mode='validation',
                                        transform=self.transforms['validation'],
                                        tokenizer=self.tokenizer,
                                        limit_num_samples=self.limit_num_samples,
                                        tokenizer_type=self.tokenizer,
                                        tokenizer_add_special_tokens=True,
                                        token_max_len=self.token_max_len)
        self.test_dataset = RocoDataset(root=self.root, mode='test',
                                         transform=self.transforms['test'],
                                         tokenizer=self.tokenizer,
                                         limit_num_samples=self.limit_num_samples,
                                         tokenizer_type=self.tokenizer,
                                         tokenizer_add_special_tokens=True,
                                         token_max_len=self.token_max_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          num_workers=self.num_data_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_data_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          num_workers=self.num_data_workers, pin_memory=True)
