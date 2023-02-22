import pandas as pd
import os
from glob import glob
import luigi
import logging

FILE_PATH = 'data//test//{}//*'
IMAGE_PATH = 'data//test//{}//{}//*.png'
IMAGES_PATH = 'data/test/{}/{}/'
LABEL_PATH = 'data//test//{}//{}//*.txt'
FOLDERS_PATH = 'data//test//*'


def get_image_info(path=FILE_PATH):
    image_id = []
    image_path = []
    labels_name = []
    labels_path = []

    for image_folder in glob(FOLDERS_PATH):
        folder_path = os.path.basename(image_folder)
        for folder in glob(FILE_PATH.format(folder_path)):
            folder_name = os.path.basename(folder)
            for image in glob(IMAGE_PATH.format(folder_path, folder_name)):
                image_basename = os.path.basename(image)
                images_name = image_basename[:-4]
                images_path = os.path.join(IMAGES_PATH.format(folder_path, folder_name), image_basename)
                labels = 1 if folder_name == 'live' else 0

                labels_name.append(labels)
                image_id.append(images_name)
                image_path.append(images_path)

            for bbox_file in glob(LABEL_PATH.format(folder_path, folder_name)):
                bbox_file_name = os.path.basename(bbox_file)

                label_path = os.path.join(IMAGES_PATH.format(folder_path, folder_name), bbox_file_name)
                labels_path.append(label_path)

    dict = {
        'image_id': image_id, 'image_path': image_path, 'label_path': labels_path, 'labels': labels_name
    }

    df = pd.DataFrame(dict)
    df.to_csv('data.csv', index=False)



def data_info_csv(path):
    images_id = []
    images_path = []
    labels_name = []
    labels_path = []
    for data_folder in glob(path):
        true_folder = r'{}/*'.format(data_folder)
        for label_folder in glob(true_folder):
            image_path = r'{}/*.png'.format(label_folder)
            label_path = r'{}/*.txt'.format(label_folder)
            label_name = os.path.basename(label_folder)
            for image in glob(image_path):
                image_basename = os.path.basename(image)
                images_name = image_basename[:-4]
                labels = 1 if label_name == 'live' else 0
                images_path.append(image)
                labels_name.append(labels)
                images_id.append(images_name)

            for label in glob(label_path):
                labels_path.append(label)

    dict = {
        'image_id': images_id, 'image_path': images_path, 'label_path': labels_path, 'labels': labels_name
    }


    df = pd.DataFrame(dict)
    print(df.head())
    df.to_csv('processing_data/data.csv', index=False)

    return

data_info_csv(FOLDERS_PATH)

