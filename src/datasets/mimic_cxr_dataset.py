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
from turbojpeg import TurboJPEG


class MIMICCXR(Dataset):
    """ MIMIC CXR Dataset"""
    def __init__(self, valid_indices, only_first_image=True, only_images=False,
                tokenizer='scibert', tokenizer_add_special_tokens=True, token_max_len=256,
                return_pil=False, transforms=None, image_type='dcm', preprocessed=True):

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
        sample = self.valid_indices[idx]
        if self.preprocessed:
            folder_path = sample['folder_path'].replace("mimic-cxr-jpg","mimic-cxr-jpg-resized")
            txt_path = sample['txt_path']#.replace("mimic-cxr","mimic-cxr-txt")
        else:
            folder_path = sample['folder_path']
            txt_path = sample['txt_path']

        #images_path = [os.path.join(folder_path, image) for image in os.listdir(folder_path) if image.endswith('.dcm')]
        images = []

        if self.image_type == 'dcm':
            dcom_images = get_dicom_files(folder_path)
            # Take the first image as the image to be predicted
            if self.only_first_image is True:
                dcom_images = dcom_images[:1]

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
                    img_2d_scaled_process = np.concatenate([img_2d_scaled_process,
                                                            img_2d_scaled_process,
                                                            img_2d_scaled_process], axis=2)
                img_2d_scaled_process = Image.fromarray(img_2d_scaled_process)

                if self.transforms is not None:
                    img_2d_scaled_process = self.transforms(img_2d_scaled_process)
                else:
                    if not self.return_pil:
                        img_2d_scaled_process = T.transforms.ToTensor()(img_2d_scaled_process)

                images.append(img_2d_scaled_process)

        if self.image_type == 'jpg':
            images_path = [x for x in os.listdir(folder_path) if x.endswith(('jpg', 'png'))]
            if self.only_first_image is True:
                images_path = images_path[:1]
            # TODO slow for loop -> make it faster?  
            for image in images_path:
                #in_file = open(os.path.join(folder_path, image), 'rb')
                #img = self.jpeg.decode(in_file.read())
                #np.moveaxis(img,-1,0)                       # make it (3,224,224)
                #in_file.close()
                img = Image.open(os.path.join(folder_path,image)).convert('RGB')
                if self.transforms is not None:
                    img_2d_scaled_process = self.transforms(img)
                else:
                    if not self.return_pil:
                        img_2d_scaled_process = T.transforms.ToTensor()(img)
                        
                images.append(img_2d_scaled_process)

        if self.only_first_image is True:
            images = images[0]

        # Text Processing
        if self.only_images is  False:
            with open(txt_path, 'r') as f:
                lines = f.readlines()

            lines = [line.strip().rstrip() for line in lines ]
            lines = [line for line in lines if line!='']
            # Remove hidden patient information.
            lines = [line.replace('___', '')for line in lines]
            #lines = lines[1:]
            report = " ".join(lines)

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
        else:
            return images

                


class MIMICCXRDataModule(pl.LightningDataModule):
    def __init__(self, cxr_dcm_path: str, cxr_jpg_path: str,  batch_size: int = 32, transforms=None, only_first_image=True,
                 only_images=False, return_pil=False, limit_num_samples=None, num_data_workers=4,
                 tokenizer='scibert', image_type="jpg", split_path=None, preprocessed = True):
        super().__init__()
        self.cxr_dcm_path = cxr_dcm_path
        self.cxr_jpg_path = cxr_jpg_path
        self.batch_size = batch_size
        self.transforms = transforms
        self.limit_num_samples = limit_num_samples
        self.only_first_image = only_first_image
        self.only_images = only_images
        self.return_pil = return_pil
        self.num_data_workers = num_data_workers
        self.tokenizer = tokenizer
        self.image_type = image_type
        self.split_path = split_path
        self.preprocessed = preprocessed

        self.setup()

    def setup(self):
        # Reads the Library and creates the paths to the images

        if self.split_path == None:
            self.data_points = self.return_valid_samples(self.cxr_dcm_path, self.cxr_jpg_path)
            if self.limit_num_samples != None:
                self.data_points = self.data_points[:self.limit_num_samples]

            self.train_split, val = model_selection.train_test_split(self.data_points, test_size=0.2)
            self.val_split, self.test_split = model_selection.train_test_split(val, test_size=0.5)

        else:
            all_datapoints_path = self.split_path + "mimic_cxr_all.json"
            train_split_path = self.split_path + "mimic_cxr_train.json"
            val_split_path = self.split_path + "mimic_cxr_val.json"
            test_split_path = self.split_path + "mimic_cxr_test.json"

            #with open(all_datapoints_path, 'r') as f:
                #all_datapoints = json.load(f)

            with open(train_split_path, 'r') as f:
                self.train_split = json.load(f)
            
            with open(val_split_path, 'r') as f:
                self.val_split = json.load(f)
            
            with open(test_split_path, 'r') as f:
                self.test_split = json.load(f)

        # Limit all predefined paths to the number of limited samples
        if self.limit_num_samples is not None:
            self.train_split = self.train_split[:self.limit_num_samples]
            self.val_split = self.val_split[:self.limit_num_samples]
            self.test_split = self.test_split[:self.limit_num_samples]

        self.train_dataset = MIMICCXR(self.train_split, transforms=self.transforms["train"],
                                     tokenizer=self.tokenizer, image_type=self.image_type,
                                     preprocessed = self.preprocessed)
        self.validation_dataset = MIMICCXR(self.val_split, transforms=self.transforms["val"],
                                           tokenizer=self.tokenizer, image_type=self.image_type,
                                           preprocessed = self.preprocessed)
        self.test_dataset = MIMICCXR(self.test_split, tokenizer=self.tokenizer,
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

    # Jpeg files does not include the patient information.
    def return_valid_samples(self, cxr_dcm_path, cxr_jpg_path):
        if self.image_type == 'dcm':
            main_path = cxr_dcm_path
        elif self.image_type =='jpg':
            main_path = cxr_jpg_path

        valid_samples = []
        for folder in tqdm([dir for dir in os.listdir(main_path) if not dir.startswith(".") and not dir.endswith('.html')]):
            for patient in [pat_fold for pat_fold in os.listdir(os.path.join(main_path, folder)) if not pat_fold.endswith('.html') and not pat_fold.startswith(".")]:
                for record in [cur_file for cur_file in os.listdir(os.path.join(main_path, folder, patient)) if not cur_file.endswith('html') and not cur_file.endswith('.txt') and not cur_file.startswith('.') and cur_file != '/']:
                    # Text is only in the record folder
                    path_of_record_folder_text = os.path.join(cxr_dcm_path, folder, patient, record)
                    path_of_record_folder = os.path.join(main_path, folder, patient, record)
                    # Each folder has its corresponding txt file
                    txt_path = path_of_record_folder_text + '.txt'
                    # Some folders are empty and they are useless
                    if len(glob.glob1(path_of_record_folder,"*.jpg")) == 0:
                        #print("No images in this folder",path_of_record_folder)
                        continue
                    else:
                        valid_samples.append(
                            {'folder_path': path_of_record_folder,
                            'txt_path': txt_path}
                        )
        return valid_samples
