import PIL
from PIL import Image
import os 
import matplotlib.pyplot as plt

img_path = r'data//test\10001\live\496120.png'
test_path = r'D:\study\Projects\train_model_baseline\data\data_face_224\496120.png'
img = Image.open(test_path)
plt.imshow(img)
plt.show()