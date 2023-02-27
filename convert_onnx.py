from tensorflow.keras.models import load_model
import tensorflow as tf
import cv2
import tf2onnx
import numpy as np


model_path = r'models\anti_spoof_224_1677482405.3738997.h5'
input_signature = (tf.TensorSpec((None,224,224,3), tf.float32, name='input'),)
model = load_model(model_path, compile=False)
onnx_model, _ = tf2onnx.convert.from_keras(model,input_signature=input_signature,opset=12, output_path='models/liveness_detection.onnx')