import cv2
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from processing_data.cropping_face import extract_face
from training_data.model import anti_spoof_face


# model_path = 'checkpoint/spoof_face_classification.hdf5'
# model = anti_spoof_face()
# model.load_weights(model_path)
model_path = r'models\anti_spoof_224_1677482405.3738997.h5'
model = load_model(model_path, compile=False)
image_path = r'D:\study\Projects\train_model_baseline\testing\spoof_1446_043043.jpg'



def image_pre_processing(image):
    image = cv2.imread(image)
    bbox, confi = extract_face(image)
    face_image = image[bbox[0][1]: bbox[0][3], bbox[0][0]: bbox[0][2]]
    face_image = cv2.resize(face_image, (224,224))
    # face_image = face_image/255
    result_img = np.expand_dims(face_image, axis=0)
    return result_img



processed = image_pre_processing(image_path)
result = model.predict(processed)[0]
label = np.argmax(result)
confidence = np.max(result)
print(label, confidence)
