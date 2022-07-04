import numpy as np
import pandas as pd
import json
import pickle as pkl
import os

from PIL import Image
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Process, Manager

paths = Path('/home/mlmi-matthias/Data')
with open(paths / 'vqa_rad_chest_xrays.pkl', 'rb') as f:
        jsons = pkl.load(f)

output_folder_path = Path('/home/mlmi-matthias/Data/VQA_RAD_preprocessed')

result = []

for json_sample in tqdm(jsons):
    image_path = json_sample['image_path']

    # resize image
    cur_img = Image.open(image_path)
    cur_img_resized = cur_img.resize((224, 224))
    cur_img_resized.save(
        fp= str(output_folder_path / json_sample['image_name']),
        format="JPEG",
        quality=100,
    )

    # change image path in json
    json_sample['image_path'] = str(output_folder_path / json_sample['image_name'])
    result.append(json_sample)

# store the jsons again in the preprocessed
output_path = Path('/home/mlmi-matthias/Data/VQA_RAD_preprocessed')
with open(output_path / 'vqa_rad_chest_paths.pkl', 'wb') as f:
        pkl.dump(result, f, pkl.DEFAULT_PROTOCOL)