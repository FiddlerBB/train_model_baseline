import pandas as pd
import os
from glob import glob
import cv2

CSV_PATH = r'D:\study\Projects\train_model_baseline\processing_data\data_kaggle.csv'
CLEANED_IMAGE_PATH = r'D:\study\Projects\train_model_baseline\data\data_kaggle_face_224\*.jpg'
# DATA_PATH = r'D:\justscan\liveness-detection\New folder\CelebA_Spoof\Data\data_face'
OUTPUT_PATH = 'processing_data/filtered_clean_kaggle.csv'


def cleaned_images(img_path):
    filter = []
    image_path = []
    for image in glob(img_path):
        image_id = os.path.basename(image)
        filter.append(image_id)
        image_path.append(image)
    dict = {
        'image_id': filter, 'cleaned_image_path': image_path
    }
    df = pd.DataFrame(dict)
    return df

def image_csv(csv_path, cleaned_image_path, output_path):
    df = pd.read_csv(csv_path)
    # df.sort_values(by=['image_id'])
    # image_id = df['image_id']
    cleaned_image = cleaned_images(cleaned_image_path)
    # filtered = df[df['image_id'].isin(image_id)]
    # filtered['image_path'] = image_path
    filtered = pd.merge(df, cleaned_image, on='image_id', how='inner')
    filtered = filtered.sample(frac=1)
    filtered.to_csv(output_path, index=False)
    

image_csv(CSV_PATH, CLEANED_IMAGE_PATH, OUTPUT_PATH)

df = pd.read_csv(OUTPUT_PATH)
gb = df.groupby('labels').count()
print(gb)