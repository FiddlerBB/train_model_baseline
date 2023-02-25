import PIL
from PIL import Image
import os 
import matplotlib.pyplot as plt
import pandas as pd

img_path = r'data//test\10001\live\496120.png'
test_path = 'D:\\study\\Projects\\train_model_baseline\\data\\data_kaggle_face_224\\spoof_973_089523.jpg'
# img = Image.open(test_path)
# plt.imshow(img)
# plt.show()
# csv = r'D:\study\Projects\train_model_baseline\processing_data\data_kaggle.csv'
# data = pd.read_csv(csv)
# gb = data.groupby('labels').count()
# print(gb)

csv_path = r'D:\study\Projects\train_model_baseline\processing_data\train_data.csv'
df = pd.read_csv(csv_path)
live = df[df['liveness_score'] == 1][:5000]
spoof = df[df['liveness_score'] == 0][:5000]

result = pd.concat([live, spoof])
result.sample(frac=1)
result= result.sample(frac=1)

df.to_csv('processing_data/shuffled.csv', index=False)