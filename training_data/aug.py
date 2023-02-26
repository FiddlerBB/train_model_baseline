import albumentations as A
import cv2
import tensorflow as tf
from config import IMG_SIZE

AUGMENTATION_TRAIN = A.Compose([
       A.ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
       A.GaussNoise(p=0.1),
       A.GaussianBlur(blur_limit=3, p=0.05),
       A.RandomCrop(width=IMG_SIZE, height=IMG_SIZE, p=1.0),
       A.HorizontalFlip(),
       A.PadIfNeeded(min_height=64, min_width=64, border_mode=cv2.BORDER_CONSTANT),
       A.OneOf([A.RandomBrightnessContrast(), A.FancyPCA(), A.HueSaturationValue()], p=0.7),
       A.RGBShift(),
       A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
   ])