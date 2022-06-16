import numpy as np
import pandas as pd
import json
import pickle as pkl

from pathlib import Path
from tqdm import tqdm
from multiprocessing import Process, Manager

dataset_path = Path('/home/mlmi-matthias/VQA-RAD/')

with open(dataset_path / 'VQA-RAD_public.json', 'r') as f:
    data = json.load(f)

result = []

for sample in tqdm(data):
    if not(sample['image_organ'] == 'CHEST'):
        continue
    image_path = str(dataset_path / 'images' / sample['image_name'])
    sample['image_path'] = image_path
    result.append(sample)

print(f'Result length: {len(result)}')

output_path = Path('/home/mlmi-matthias/Data')
with open(output_path / 'vqa_rad_chest_xrays.pkl', 'wb') as f:
        pkl.dump(result, f, pkl.DEFAULT_PROTOCOL)