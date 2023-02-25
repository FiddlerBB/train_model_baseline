import numpy as np
from face_detection.ScrFd.scrfd import SCRFD
import cv2
import pandas as pd
import os
import asyncio


MODEL_PATH = r'processing_data\face_detection\det_10g.onnx'
detector = SCRFD(MODEL_PATH)
detector.prepare(0)

# image_path = r'D:\study\Projects\train_model_baseline\testing\istockphoto-155137500-612x612.jpg'

async def extract_face(image):
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


async def croping_face(image_path, output_path):
    loop = asyncio.get_running_loop()

    image = cv2.imread(r'{}'.format(image_path))
    bbox, confidence = await extract_face(image)
    if bbox:
        if confidence[0] > 0.55:
            try:
                facial_image = image[bbox[0][1]: bbox[0][3], bbox[0][0]: bbox[0][2]]
                facial_image = cv2.resize(facial_image, (224,224))
                return await loop.run_in_executor(None, cv2.imwrite, output_path, facial_image)
            except:
                pass


async def main():
    df = pd.read_csv('processing_data/data_kaggle.csv')
    # df = df[:10]
    tasks = []
    for _, row in df.iterrows():
        # image_name = str(row['image_id']) + '.png'
        image_path = row['image_path']
        prepared_data_path = os.path.join('data/data__kaggle_face_224', row['image_id'])
        if not os.path.exists(prepared_data_path):
            task = asyncio.create_task(croping_face(image_path, prepared_data_path))
            tasks.append(task)
        

    await asyncio.gather(*tasks)

asyncio.run(main())

# croping_face(image_path, 'testing/tesing.png')

