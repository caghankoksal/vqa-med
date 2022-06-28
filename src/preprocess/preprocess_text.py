import os
from os import path, makedirs
import shutil
import json
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


with open(DATA_JSON_PATH, "r", encoding="utf-8") as f:
    valid_data_points = json.load(f)


def move_all_txt(data_points):
    """_summary_

    Args:
        data_points (String): Path to the data points
    """
    for file in tqdm(data_points):
        txt_file_path = file["txt_path"]
        new_folder_path = "/" + os.path.join(*txt_file_path.split("/")[:-1]).replace(
            "mimic-cxr", "mimic-cxr-txt"
        )
        # print(new_folder_path)
        if not path.isdir(new_folder_path):
            makedirs(new_folder_path)

        shutil.copyfile(
            txt_file_path, os.path.join(new_folder_path, txt_file_path.split("/")[-1])
        )


move_all_txt(valid_data_points)
