import os
import time
import tensorflow as tf
import pandas as pd
from config import (
    MODELS,
    CPS,
    LR,
    BATCH_SIZE,
    LOG,
    IMG_SIZE,
    EPOCHS
)

from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
import numpy as np
from dataset import FaceAntiSpoofDataGenerator
from tensorflow.keras.metrics import Precision, Recall
from sklearn.model_selection import train_test_split
from aug import AUGMENTATION_TRAIN
from model import anti_spoof_face

checkpoint_filepath = "spoof_face_classification.hdf5"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
CSV_PATH = r'processing_data\filtered_clean_kaggle.csv'

metrics = ['accuracy', Precision(), Recall()]

def load_data(db, val_split=0.2):
    if db == "train":
        data = pd.read_csv(CSV_PATH)
        # data.sort_values('labels', ascending=False)
        # data = data[:10000]
        img_path = data['cleaned_image_path'].values
        label = data['labels'].values
        # label = np.array(label, dtype='uint8')
        X_train, X_test, y_train, y_test = train_test_split(
            img_path, label, test_size=val_split, random_state=42)
        if val_split > 0:

            train_data = FaceAntiSpoofDataGenerator(X_train, y_train, no_classes=2, batch_size=BATCH_SIZE, img_size=(
                IMG_SIZE, IMG_SIZE), shuffle=True, augment=AUGMENTATION_TRAIN)
            val_data = FaceAntiSpoofDataGenerator(X_test, y_test, no_classes=2, batch_size=BATCH_SIZE, img_size=(
                IMG_SIZE, IMG_SIZE), shuffle=False, augment=None)

            return train_data, val_data


def trainer(train_generator, val_generator):

    vgg_model = anti_spoof_face()
    print(vgg_model.count_params(), vgg_model.inputs, vgg_model.outputs)
    print(len(train_generator), len(val_generator))

    checkpoint = callbacks.ModelCheckpoint(
        filepath=os.path.join(CPS, checkpoint_filepath),
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        monitor='val_loss',
        mode='min')

    early = callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=5, verbose=1)
    redonplat = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, mode="min", patience=3, verbose=1
    )
    csv_logger = callbacks.CSVLogger(
        os.path.join(LOG, 'anti_spoof{}_{}.csv'.format(
            IMG_SIZE, time.time()
        )),
        append=False, separator=','
    )

    callbacks_list = [
        checkpoint,
        early,
        redonplat,
        csv_logger,
    ]

    optim = optimizers.Adam(learning_rate=LR)
    vgg_model.compile(loss='binary_crossentropy', optimizer=optim,
                      metrics=metrics)

    history = vgg_model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=EPOCHS,
        callbacks=callbacks_list,
        validation_data=val_generator,
        validation_steps=len(val_generator)
    )

    vgg_model.save_weights(
        os.path.join(
            MODELS,
            "anti_spoof_weight_{}_{}.h5".format(IMG_SIZE, time.time())
        )
    )
    vgg_model.save(
        os.path.join(MODELS,
                     "anti_spoof_{}_{}.h5".format(IMG_SIZE, time.time()
                                                  )
                     )
    )

    with open(os.path.join(MODELS, "anti_spoof.json"), "w") as f:
        f.write(vgg_model.to_json())

    return history


if __name__ == '__main__':
    train_data, val_data = load_data("train", val_split=0.1)
    trainer(train_generator=train_data, val_generator=val_data)
