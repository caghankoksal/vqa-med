import torch
from torch.utils.data import DataLoader, Dataset
from fastai.medical.imaging import get_dicom_files
from sklearn import model_selection
import torchvision.transforms as T
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,PreTrainedTokenizerFast
import numpy as np
from PIL import Image
import pytorch_lightning as pl 
from tqdm import tqdm as tqdm
from PIL import Image


class MIMICCXR(Dataset):
    """ MIMIC CXR Dataset"""
    def __init__(self, valid_indices, all_samples,only_first_image = True, only_images=False, tokenizer='scibert',
                tokenizer_add_special_tokens=True, token_max_len=256, return_pil=False, transforms=None):
        self.valid_indices = valid_indices
        self.all_samples = all_samples
        self.only_first_image = only_first_image
        self.only_images = only_images
        self.token_max_len = token_max_len
        self.return_pil = return_pil
        self.transforms = transforms
        
        if tokenizer == "sciFive":
            self.tokenizer = AutoTokenizer.from_pretrained("razent/SciFive-large-Pubmed_PMC-MedNLI")
        elif tokenizer == "scibert":
            self.tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        elif tokenizer == "scratch":
            # TODO : Implement a scratch tokenizer
            pass
        elif tokenizer is None:
            self.tokenizer = None

        if tokenizer_add_special_tokens:
            special_tokens_dict = {'additional_special_tokens': ['<image>', '<EOC>']}
            # Set the beginning of sentence token to <BOS>
            self.tokenizer.bos_token = '<BOS>'
            num_added_toks = self.tokenizer.add_special_tokens(special_tokens_dict)


    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        sample = self.all_samples[self.valid_indices[idx]]
        folder_path = sample['folder_path']
        txt_path = sample['txt_path']
        #images_path = [os.path.join(folder_path, image) for image in os.listdir(folder_path) if image.endswith('.dcm')]
        dcom_images = get_dicom_files(folder_path)

        
        # Take the first image as the image to be predicted
        if self.only_first_image==True:
            dcom_images = dcom_images[:1]

        images = []
        for img in dcom_images:
            dcm_read = img.dcmread()
            # Scale pixel values according to Rescale Slope and Rescale Intercept
            dcm_read = dcm_read.scaled_px
            # Convert pixel_array (img) to -> gray image (img_2d_scaled)
            ## Step 2. Rescaling grey scale between 0-255
            img_2d_scaled = (np.maximum(dcm_read,0) / dcm_read.max()) * 255.0

            ## Step 3. Convert to uint
            img_2d_scaled = np.uint8(img_2d_scaled)

            # Add 3rd dimension to gray image
            
            if len(img_2d_scaled.shape) == 2:
                img_2d_scaled_process = img_2d_scaled[:, :, np.newaxis]
                img_2d_scaled_process = np.concatenate([img_2d_scaled_process, img_2d_scaled_process, img_2d_scaled_process], axis=2)
            
            img_2d_scaled_process = Image.fromarray(img_2d_scaled_process)

            if self.transforms != None:
                img_2d_scaled_process = self.transforms(img_2d_scaled_process)
            else:
                if not self.return_pil:
                    img_2d_scaled_process = T.transforms.ToTensor()(img_2d_scaled_process)
                    
            images.append(img_2d_scaled_process)
        
        if self.only_first_image==True:
            images = images[0]


        # Text Processing
        if self.only_images == False:
            with open(txt_path, 'r') as f:
                lines = f.readlines()

            lines = [line.strip().rstrip() for line in lines ]
            lines = [line for line in lines if line!='']
            # Remove hidden patient information.
            lines = [line.replace('___', '')for line in lines]
            #lines = lines[1:]
            report = " ".join(lines)

            # Put image at the beginning of the explanation
            report = '<BOS> ' + '<image> ' + 'Output: ' + report + ' <EOC>'

            encoding = self.tokenizer.encode_plus(report, padding='max_length', truncation=True, max_length=self.token_max_len, return_tensors="pt")
            input_ids = encoding['input_ids']
            token_type_ids = encoding['token_type_ids']
            
            #At the beginnin At the end guess EOC token
            eoc_token_id = self.tokenizer.all_special_ids[self.tokenizer.all_special_tokens.index('<EOC>')]
            targets = torch.cat( ( input_ids[:,1:], torch.tensor([eoc_token_id]).unsqueeze(1) ), dim=1)

            return {"image":images, "text": report, "input_ids": input_ids, "token_type_ids": token_type_ids, "targets" : targets}
        else:
            return images

                


class MIMICCXRDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32, transforms=None, only_first_image=True,
                 only_images=False, return_pil=False, limit_num_samples=None, num_data_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transforms = transforms
        self.limit_num_samples = limit_num_samples
        self.only_first_image = only_first_image
        self.only_images = only_images
        self.return_pil = return_pil
        self.num_data_workers = num_data_workers

        self.setup()

    def setup(self):
        self.data_points = self.return_valid_samples(self.data_dir)
        if self.limit_num_samples != None:
            self.data_points = self.data_points[:self.limit_num_samples]
            
        self.train_split, val = model_selection.train_test_split(torch.arange(len(self.data_points)), test_size=0.2)
        self.val_split, self.test_split = model_selection.train_test_split(val, test_size=0.5)
        self.train_dataset = MIMICCXR(self.train_split, self.data_points, transforms=self.transforms["train"],)
        self.validation_dataset = MIMICCXR(self.val_split, self.data_points, transforms=self.transforms["val"])
        self.test_dataset = MIMICCXR(self.test_split, self.data_points)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_data_workers)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.batch_size, num_workers=self.num_data_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,num_workers=self.num_data_workers)


    def return_valid_samples(self, mimic_cxr_path):
        valid_samples = []
        for folder in tqdm([folder for folder in os.listdir(mimic_cxr_path) if not folder.startswith(".")]):
            for patient in [folder for folder in os.listdir(os.path.join(mimic_cxr_path, folder)) if not folder.endswith('.html') and not folder.startswith(".")]:
                
                for record in [folder for folder in os.listdir(os.path.join(mimic_cxr_path, folder, patient)) if not folder.endswith('html') and not folder.endswith('.txt') and not folder.startswith('.') and folder != '/']:
                    path_of_record_folder = os.path.join(mimic_cxr_path, folder, patient, record)
                    # Each folder has its corresponding txt file
                    txt_path = path_of_record_folder + '.txt'
                    # Some folders are empty and they are useless
                    if len(os.listdir(path_of_record_folder)) == 1 or len(os.listdir(path_of_record_folder)) == 0:
                        #print("No images in this folder",path_of_record_folder)
                        continue
                    else:
                        valid_samples.append(
                            {'folder_path': path_of_record_folder,
                            'txt_path': txt_path}
                        )
        return valid_samples
                
                    

