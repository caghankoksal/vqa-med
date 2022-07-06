import os
import glob
import json
import numpy as np
import multiprocessing as mp
from os import makedirs, path
from PIL import Image
from tqdm import tqdm as tqdm

if os.getcwd().startswith("/home/mlmi-matthias"):

    MIMIC_CXR_DCM_PATH = (
        "/home/mlmi-matthias/physionet.org/files/mimic-cxr/2.0.0/files/"
    )
    MIMIC_CXR_JPG_PATH = (
        "/home/mlmi-matthias/physionet.org/files/mimic-cxr-jpg/2.0.0/files/"
    )
    FOLDER_PROJECT = "/home/mlmi-matthias/Caghan/mlmi-vqa/"

elif os.getcwd().startswith("/Users/caghankoksal"):
    MIMIC_CXR_DCM_PATH = "/Users/caghankoksal/Desktop/development/Flamingo-playground/physionet.org/files/mimic-cxr/2.0.0/files/"
    MIMIC_CXR_JPG_PATH = "/Users/caghankoksal/Desktop/development/physionet.org/files/mimic-cxr-jpg/2.0.0/files/"
    FOLDER_PROJECT = "/Users/caghankoksal/Desktop/SS2022/mlmi-vqa/"


DATA_JSON_PATH = FOLDER_PROJECT + "data/external/mimic_cxr_all.json"


with open(DATA_JSON_PATH, "r") as f:
    valid_data_points = json.load(f)


def resize_images_all(data_points, slice):
    """
    Resize images and store them, keeping the structure of the dataset
    """
    for file in tqdm(data_points[slice]):
        folder_path = file["folder_path"]
        new_folder_path = folder_path.replace("mimic-cxr-jpg", "mimic-cxr-jpg-resized")
        if not path.isdir(new_folder_path):
            makedirs(new_folder_path)

        for img_path in glob.glob1(folder_path, "*.jpg"):
            cur_img_full_path = os.path.join(folder_path, img_path)
            try:
                cur_img = Image.open(cur_img_full_path)
                cur_img_resized = cur_img.resize((224, 224))
                cur_img_resized.save(
                    fp=os.path.join(new_folder_path, img_path),
                    format="JPEG",
                    quality=100,
                )
            except:
                print("Error in resizing", cur_img_full_path)
                continue


# setup parallelization parameters
nr_samples = len(valid_data_points)
n_proc = 1
offset = 0

chunksize = nr_samples // n_proc
proc_slices = []

for i_proc in range(n_proc):
    chunkstart = int(offset + (i_proc * chunksize))
    # make sure to include the division remainder for the last process
    chunkend = int(offset + (i_proc + 1) * chunksize) if i_proc < n_proc - 1 else int(offset + nr_samples)
    proc_slices.append(np.s_[chunkstart:chunkend])

print(proc_slices)

params = [(valid_data_points, slice) for slice in proc_slices]

running_tasks = [mp.Process(target=resize_images_all, args=param) for param in params]
for running_task in running_tasks:
    running_task.start()
for running_task in running_tasks:
    running_task.join()
