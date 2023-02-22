import numpy as np
from face_detection.ScrFd.scrfd import SCRFD
import cv2
import pandas as pd
import os

MODEL_PATH = r'processing_data\face_detection\det_10g.onnx'
detector = SCRFD(MODEL_PATH)
detector.prepare(0)

image_path = r'D:\study\Projects\train_model_baseline\testing\istockphoto-155137500-612x612.jpg'

def extract_face(image):
    # image = cv2.imread(img_path)[:,:,::-1]
    bbox, _ = detector.autodetect(image)
    bbox_total = []
    confidence_total = []
    for face in bbox:
        facial_area = list(face[0:4].astype(int))
        bbox_total.append(facial_area)
        confidence = face[4]
        confidence_total.append(confidence)
    return bbox_total, confidence_total


def croping_face(image_path,output_path):
    image = cv2.imread(image_path)
    bbox, confidence = extract_face(image)
    if bbox:
        if confidence[0] > 0.55:
            try:
                facial_image = image[bbox[0][1]: bbox[0][3], bbox[0][0]: bbox[0][2]]
                return cv2.imwrite('{}'.format(output_path), facial_image)
            except: 
                pass


df = pd.read_csv('data.csv')


for idx, row in df.iterrows():
    print(row['image_id'])
    image_name = str(row['image_id']) + '.png'
    image_path = row['image_path']
    prepared_data_path = os.path.join('Data/data_face', image_name)
    
    croping_face(image_path, prepared_data_path)

# croping_face(image_path, 'testing/tesing.png')

