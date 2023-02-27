import tensorflow
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications.vgg16 import preprocess_input

class FaceAntiSpoofDataGenerator(tensorflow.keras.utils.Sequence):
    """Data generator for classification model
    """

    def __init__(self,
                 img_fps,
                 labels,
                 batch_size=64,
                 img_size=(64, 64),
                 no_channels=3,
                 no_classes=101,
                 augment=None,
                 pad=True,
                 shuffle=True,
                 debug=False):

        self.img_size = img_size
        self.no_channels = no_channels
        self.batch_size = batch_size
        print(">>> Batch_size: {} images".format(self.batch_size))

        self.img_fps = img_fps
        self.labels = labels
        assert len(self.img_fps) == len(self.labels)
        self.ids = range(len(self.img_fps))

        self.no_classes = no_classes
        self.augment = augment
        self.pad = pad
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index *
                               self.batch_size:(index + 1) * self.batch_size]

        temp_ids = [self.ids[k] for k in indexes]
        X, y = self.__data_generation(temp_ids)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, ids):
        X = np.empty((0, *self.img_size, self.no_channels))
        y = []

        for index, id_ in enumerate(ids):
            img = cv2.imread(self.img_fps[id_])
            label = self.labels[id_]
            # try:
            #     img = cv2.imread(self.img_fps[id_])
            #     label = self.labels[id_]
            # except:
            #     continue
            if img is None:
                continue
            img = cv2.resize(img, self.img_size)
            # img = preprocess_input(img)
            if self.augment:
                aug = self.augment(image=img)
                img = aug["image"]
            img = img.astype('float32')

            X = np.vstack((X, np.expand_dims(img, axis=0)))
            y.append(label)
        lb = LabelEncoder()
        y = lb.fit_transform(y)
        # y = np.array(y)

        y = to_categorical(y, num_classes=self.no_classes)
        y = np.array(y)
        return X, y