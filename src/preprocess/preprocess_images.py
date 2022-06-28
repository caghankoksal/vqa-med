import os
from tqdm import tqdm as tqdm
import glob
import json
from os import makedirs, path, getcwd
from PIL import Image

if os.getcwd().startswith('/home/mlmi-matthias'):
    MIMIC_CXR_DCM_PATH = '/home/mlmi-matthias/physionet.org/files/mimic-cxr/2.0.0/files/'
    MIMIC_CXR_JPG_PATH = "/home/mlmi-matthias/physionet.org/files/mimic-cxr-jpg/2.0.0/files/"

elif os.getcwd().startswith('/Users/caghankoksal'):
    MIMIC_CXR_DCM_PATH = '/Users/caghankoksal/Desktop/development/Flamingo-playground/physionet.org/files/mimic-cxr/2.0.0/files/'
    MIMIC_CXR_JPG_PATH = '/Users/caghankoksal/Desktop/development/physionet.org/files/mimic-cxr-jpg/2.0.0/files/'


DATA_JSON_PATH = "/home/mlmi-matthias/Caghan/mlmi-vqa/data/external/mimic_cxr_all.json"
with open(DATA_JSON_PATH, "r") as f:
   valid_data_points = json.load(f)




def resize_images_all(valid_data_points):
    for file in tqdm(valid_data_points):
        folder_path = file['folder_path']
        new_folder_path = folder_path.replace("mimic-cxr-jpg","mimic-cxr-jpg-resized")
        if not path.isdir(new_folder_path):
            makedirs(new_folder_path)

        for img_path in glob.glob1(folder_path,"*.jpg"):
            cur_img_full_path = os.path.join(folder_path, img_path)
            try:
                cur_img = Image.open(cur_img_full_path)
                cur_img_resized = cur_img.resize((224,224))
                cur_img_resized.save(fp=os.path.join(new_folder_path,img_path),format="JPEG",quality=100)
            except:
                print("Error in resizing",cur_img_full_path)
                continue


    