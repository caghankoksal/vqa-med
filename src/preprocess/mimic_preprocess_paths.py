import numpy as np
import csv
import pickle as pkl

from pathlib import Path
from tqdm import tqdm
from multiprocessing import Process, Manager

dataset_path = Path('/home/mlmi-matthias/physionet.org/files/mimic-cxr/2.0.0/')
dataset_path_jpg = Path('/home/mlmi-matthias/physionet.org/files/mimic-cxr-jpg/2.0.0/')

# be careful here, loading both amounts to ~10GB of memory
image_list = list(csv.DictReader(open(dataset_path / 'cxr-record-list.csv', 'r'), delimiter=','))         # list of dictionaries
report_list = list(csv.DictReader(open(dataset_path / 'cxr-study-list.csv', 'r'), delimiter=','))           # list of dictionaries

# get the csv to only keep the 
position_list = list(csv.DictReader(open(dataset_path_jpg / 'mimic-cxr-2.0.0-metadata.csv', 'r'), delimiter=','))


print(f'pos list: {len(position_list)}\n image list {len(image_list)}\nreport list: {len(report_list)}')


# split workload among processes
dataset_length = len(image_list)
n_proc = 20
offset = 0

chunksize = dataset_length // n_proc
proc_slices = []

for i_proc in range(n_proc):
        chunkstart = int(offset + (i_proc * chunksize))
        # make sure to include the division remainder for the last process
        chunkend = int(offset + (i_proc + 1) * chunksize) if i_proc < n_proc - 1 else int(offset + dataset_length)
        proc_slices.append(np.s_[chunkstart:chunkend])

print(f'Number of slices: {len(proc_slices)}\n{proc_slices}')


# process function to create the dictionaries for each image-report pair
def process(data, slice, rank):                    # split it up into slices
    # preprocess reports and images
    # iterate through images and find corresponding report
    for image in tqdm(image_list[slice]):
        image_path = image['path']
        
        # get corresponding position
        corresponding_position_entry = None
        for sample_postion in position_list:
                if image['dicom_id'] == sample_postion['dicom_id']:
                        corresponding_position_entry = sample_postion
                        break

        if corresponding_position_entry is None:
                print(f"Did not find corect position for image path {image_path}")

        # filter out all lateral images
        if not (corresponding_position_entry['ViewPosition'] == 'PA' or corresponding_position_entry['ViewPosition'] == 'AP'):
                continue

        id = image['study_id']
        # find corresponding report
        report_path = None
        for report in report_list:
            if id in report['study_id']:
                report_path = report['path']
        
        if report is None:
                print(f'Not found report for image with path {image_path}, study id: {id}')
                continue


        # adjust for jpeg path
        image_path = image_path[:-3] + 'jpg'

        entry = {'subject_id': report['subject_id'],
                'study_id': id,
                'dicom_id': image['dicom_id'],
                'report_path': str(dataset_path) + '/' + report_path,
                'image_path': str(dataset_path_jpg) + '/' + image_path,
                }

        data.append(entry)


data = Manager().list()
processes = []
for i in range(n_proc):
        p = Process(target=process, args=(data, proc_slices[i], i))  # Passing the list
        p.start()
        processes.append(p)
for p in processes:
        p.join()

data_list = list(data)
print(len(data_list))
output_path = Path('/home/mlmi-matthias/Data')
with open(output_path / 'mimic_images2reports_mlmi.pkl', 'wb') as f:
        pkl.dump(data_list, f, pkl.DEFAULT_PROTOCOL)
