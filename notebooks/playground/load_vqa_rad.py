
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# %%
image_path1 = '/home/andrei/TUM/SoSe2022/MLMI/vqa-rad/images/synpic100132.jpg' 
image_path2 = '/home/andrei/TUM/SoSe2022/MLMI/vqa-rad/images/synpic100176.jpg'
image_path3 = '/home/andrei/TUM/SoSe2022/MLMI/vqa-rad/images/synpic20626.jpg'

# %%
img1 = mpimg.imread(image_path1)
img2 = mpimg.imread(image_path2)
img3 = mpimg.imread(image_path3)

# print(f'img3')

plt.subplot(1,2,1)
plt.imshow(img3, cmap='gray')
plt.title('20626')

# plt.subplot(1,2,1)
# plt.imshow(img1, cmap='gray')
# plt.title('100132')

# plt.subplot(1,2,2)
# plt.imshow(img2, cmap='gray')
# plt.title('100176')

plt.show()
