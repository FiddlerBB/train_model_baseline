import cv2 
import numpy as np 
from tensorflow.keras.models import load_model
cap = cv2.VideoCapture(0)
from training_data.model import anti_spoof_face
from processing_data.cropping_face import extract_face

# model_path = 'models/anti_spoof_128_1677228839.5275097.h5'

model_path = 'checkpoint/spoof_face_classification.hdf5'

model = anti_spoof_face()
model.load_weights(model_path)
# model = load_model(model_path)
def pre_processing(image):
    image = cv2.imread(image)
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
        face_image = cv2.resize(image, (224,224))
        result_img = np.expand_dims(face_image, axis=0)
        face.append(result_img)
        x_total.append(x)
        y_total.append(y)
        w_total.append(w)
        h_total.append(h)
    return face, x_total[0], y_total[0],w_total[0],h_total[0]


while True: 
    ret, frame = cap.read()
    bbox, confi = extract_face(frame)
    for i in bbox:
        facial_img = frame[i[1]: i[3], i[0]: i[2]]
        try:
            face_image = cv2.resize(facial_img, (224,224))
        except:
            continue
        face_image = np.expand_dims(face_image, axis=0)
        result = model.predict(face_image)[0]
        label = np.argmax(result)
        confidence = np.max(result)
        if label == 0: 
            labels = 'real'
            color = (0,0,255)
        else:
            labels = 'spoof'
            color = (0,255,0)
        cv2.putText(frame, labels, (i[0], i[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)  
        cv2.rectangle(frame, (i[0],i[1]), (i[2], i[3]), (0,255,255), 1)
        cv2.imshow('a', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()



