# %%
import torchxrayvision as xrv

# %%
import pydicom as dicom
import matplotlib.pylab as plt

# %%
image_path1 = '/home/andrei/Downloads/02aa804e-bde0afdd-112c0b34-7bc16630-4e384014.dcm' 
image_path2 = '/home/andrei/Downloads/174413ec-4ec4c1f7-34ea26b7-c5f994f8-79ef1962.dcm' 

# %%
ds1 = dicom.dcmread(image_path1)
ds2 = dicom.dcmread(image_path2)


# %%
plt.subplot(1,2,1)
plt.imshow(ds1.pixel_array, cmap='gray')
plt.title('PA')

plt.subplot(1,2,2)
plt.imshow(ds2.pixel_array, cmap='gray')
plt.title('LA')

plt.show()
# %%
# folder = '/home/andrei/TUM/SoSe2022/MLMI/mimic-cxr/physionet.org/files/mimic-cxr/2.0.0/files/p10/p10000032/s50414267.txt'
# os.path.exists(folder)

# %%



