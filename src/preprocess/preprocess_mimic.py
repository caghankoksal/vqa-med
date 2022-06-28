import os
from tqdm import tqdm as tqdm
import glob

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

# Jpeg files does not include the patient information.
# Jpeg files does not include the patient information.
def return_valid_samples(cxr_dcm_path, cxr_jpg_path):

    main_path = cxr_jpg_path

    valid_samples = []
    print(os.listdir(cxr_jpg_path))
    for folder in tqdm(
        [
            dir
            for dir in os.listdir(main_path)
            if not dir.startswith(".") and not dir.endswith(".html")
        ]
    ):
        print("Current Folder : ", folder)
        for patient in [
            pat_fold
            for pat_fold in os.listdir(os.path.join(main_path, folder))
            if not pat_fold.endswith(".html") and not pat_fold.startswith(".")
        ]:
            for record in [
                cur_file
                for cur_file in os.listdir(os.path.join(main_path, folder, patient))
                if not cur_file.endswith("html")
                and not cur_file.endswith(".txt")
                and not cur_file.startswith(".")
                and cur_file != "/"
            ]:
                # Text is only in the record folder
                path_of_record_folder_text = os.path.join(
                    cxr_dcm_path, folder, patient, record
                )
                path_of_record_folder = os.path.join(main_path, folder, patient, record)
                # Each folder has its corresponding txt file
                txt_path = path_of_record_folder_text + ".txt"
                # Some folders are empty and they are useless
                if len(glob.glob1(path_of_record_folder, "*.jpg")) == 0:
                    # print("No images in this folder",path_of_record_folder)
                    pass
                else:
                    valid_samples.append(
                        {"folder_path": path_of_record_folder, "txt_path": txt_path}
                    )
    return valid_samples


valid_list = return_valid_samples(MIMIC_CXR_DCM_PATH, MIMIC_CXR_JPG_PATH)


import json

with open(FOLDER_PROJECT + "data/external/mimic_cxr_all.json", "w") as final:
    json.dump(valid_list, final, indent=2)
with open(FOLDER_PROJECT + "data/external/mimic_cxr_all.json", "r") as f:
    valid_data_points = json.load(f)


import torch
from sklearn import model_selection

train_split, val = model_selection.train_test_split(valid_data_points, test_size=0.2)
val_split, test_split = model_selection.train_test_split(val, test_size=0.5)

with open(FOLDER_PROJECT + "data/external/mimic_cxr_train.json", "w") as final:
    json.dump(train_split, final, indent=2)

with open(FOLDER_PROJECT + "data/external/mimic_cxr_val.json", "w") as final:
    json.dump(val_split, final, indent=2)

with open(FOLDER_PROJECT + "data/external/mimic_cxr_test.json", "w") as final:
    json.dump(test_split, final, indent=2)

print("Train Set len :  ", len(train_split))
print("Validation Set len : ", len(val_split))
print("Test Set len : ", len(test_split))
