import numpy as np
import pandas as pd
import json
import pickle as pkl
import os

from PIL import Image
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Process, Manager

path = Path('/home/mlmi-matthias/VQA-RAD')
output_folder_path = Path('/home/mlmi-matthias/VQA-RAD/images_preprocessed')

with open(path / 'VQA-RAD_public.json', 'r') as f:
        jsons = json.load(f)

print(f'len images: {len(jsons)}')

for json_sample in tqdm(jsons):
    image_path = path / 'images' / json_sample['image_name']

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
