from tensorflow.keras.applications import MobileNetV2, VGG16, Xception
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import sys
sys.path.append('../../')
# from config import IMG_SIZE, LR

def anti_spoof_face():
    base_model = Xception(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
    # base_model = Xception(weights='imagenet', include_top=False, input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3)))
    head_model = base_model.output

    head_model = GlobalAveragePooling2D()(head_model)
    head_model = Flatten(name='flatten')(head_model)
    head_model = Dense(512, activation='relu', kernel_regularizer=l2(0.01))(head_model)
    head_model = Dropout(0.5)(head_model)
    head_model = Dense(2, activation='softmax', kernel_regularizer=l2(0.01))(head_model)

    xception = Model(base_model.inputs, head_model)
    for layer in base_model.layers:
        layer.trainable = True
    return xception

