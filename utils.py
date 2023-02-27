import numpy as np
import cv2
import onnx
import onnxruntime

model_path = r'D:\study\Projects\train_model_baseline\models\liveness_detection.onnx'
image_path = r'D:\study\Projects\train_model_baseline\testing\istockphoto-155137500-612x612.jpg'


class LivenessONNX:
    def __init__(self, model_file=model_path, session=None):
        assert model_file is not None
        self.model_file = model_file
        self.session = session

    def preprocessing(self, image):
        img = cv2.resize(image, (224, 224))
        return np.expand_dims(img, axis=0)

    def predict(self, image):
        img = self.preprocessing(image)
        ort = onnxruntime.InferenceSession(self.model_file)
        input = ort.get_inputs()[0].name
        output = ort.get_outputs()
        output_names = []
        for out in output:
            output_names.append(out.name)

        net = ort.run(output_names, {input: img.astype('float32')})[0]
        return np.argmax(net), np.max(net[0])
    
model = LivenessONNX()
image = cv2.imread(image_path)
result = model.predict(image)
print(result)