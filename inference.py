import cv2
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from processing_data.cropping_face import extract_face
from training_data.model import anti_spoof_face


model_path = 'checkpoint/spoof_face_classification.hdf5'
model = anti_spoof_face()
model.load_weights(model_path)
# model = anti_spoof_face()
# model.load_weights(model_path)
# model = load_model(model_path)
image_path = r'D:\study\Projects\train_model_baseline\testing\images.jpeg'


def pre_processing(image):
    image = cv2.imread(image)[:,:,::-1]
    face = []
    x_total = []
    y_total = []
    w_total = []
    h_total = []
    bbox, confi = extract_face(image)
    for i in bbox:
        x = i[0]
        y = i[1]
        w = i[2]
        h = i[3]
        face_image = image[i[1]: i[3], i[0]: i[2]]
        # face_image = face_image[:,:,::-1]
        face_image = cv2.resize(image, (128,128))
        result_img = np.expand_dims(face_image, axis=0)
        face.append(result_img)
        x_total.append(x)
        y_total.append(y)
        w_total.append(w)
        h_total.append(h)
    return face, x_total[0], y_total[0],w_total[0],h_total[0]


def image_pre_processing(image):
    image = cv2.imread(image)[:,:,::-1]
    bbox, confi = extract_face(image)
    face_image = image[bbox[0][1]: bbox[0][3], bbox[0][0]: bbox[0][2]]
    face_image = cv2.resize(image, (128,128))
    face_image = face_image/255
    result_img = np.expand_dims(face_image, axis=0)
    return result_img



processed = image_pre_processing(image_path)
result = model.predict(processed)[0]
label = np.argmax(result)
confidence = np.max(result)
print(label, confidence)
