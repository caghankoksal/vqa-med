import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import json
import os 
from PIL import Image
from torchvision import transforms


class VQ_Rad_Dataset(Dataset):
    def __init__(self, root, split="train",
                 transform=None, question_tokenize=None, answer_tokenize=None):
        self.root = root
        self.transform = transform

        if split == "train":
            self.annotations = json.load(open(os.path.join(root, "trainset.json")))
        elif split == "val":
            self.annotations = json.load(open(os.path.join(root, "testset.json")))

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        cur_ann = self.annotations[idx]
        #qid = cur_ann["qid"]
        image_name = cur_ann["image_name"]
        #image_organ = cur_ann["image_organ"]
        answer = cur_ann["answer"]
        #answer_type = cur_ann["answer_type"]
        #question_type = cur_ann["question_type"]
        question = cur_ann["question"]
        #phrase_type = cur_ann["phrase_type"]
        # Img path of the given question
        cur_image_path = os.path.join(self.root, "images", image_name)
        img = Image.open(cur_image_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img, question, answer
