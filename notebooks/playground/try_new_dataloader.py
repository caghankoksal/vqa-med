import sys 
sys.path.append('..')
import glob
import pyvips
import timeit
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning import Trainer, seed_everything
import torchvision.transforms as T
from PIL import Image
import torchvision.transforms as T

from turbojpeg import TurboJPEG

MIMIC_CXR_JPG_RESIZED_PATH = "/home/mlmi-matthias/physionet.org/files/mimic-cxr-jpg-resized/2.0.0/files/p18/*/*/*.jpg"
folder_path = glob.glob(MIMIC_CXR_JPG_RESIZED_PATH)

jpeg = TurboJPEG()

def usingPIL(path):
    img = T.ToTensor()(Image.open(path).convert("RGB"))
    # print(f'PIL shape: {img.shape}')
    return img

def usingPyvips(path):
    image = pyvips.Image.new_from_file(path, access="sequential") 
    mem_img = image.write_to_memory() 
    img=np.frombuffer(mem_img, dtype=np.uint8)
    # print(f'vips shape: {img.shape}')
    return img

def usingJPEGTurbo(path):
    in_file = open(path, 'rb')
    img = jpeg.decode(in_file.read())
    img = Image.fromarray(img)
    in_file.close()
    # print(f'turbo shape: {img.shape}')
    return img




def bench(name):
    result = timeit.timeit(f"using{name}('{folder_path[0]}')", 
                           setup=f"from __main__ import using{name}",
                           number=10)
    print(f"using{name}: {result * 10} ms")

if __name__ == "__main__":
    bench('PIL')
    bench('Pyvips')
    bench('JPEGTurbo')