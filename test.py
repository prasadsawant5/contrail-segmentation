import os
import cv2
import numpy as np
import tensorflow as tf
from config import BAND_11, BAND_14, BAND_15, TEST, TRAIN_IMG, SAVED_MODEL
from utils import get_image, read_npy_file

if __name__ == '__main__':
    testPaths = [os.path.join(TEST, img) for img in os.listdir(os.path.join(TEST))]

    model = tf.keras.models.load_model(SAVED_MODEL, compile=False)

    for testPath in testPaths:
        record_id = testPath.split('/')[-1]
        band11 = read_npy_file(os.path.join(testPath, BAND_11))
        band14 = read_npy_file(os.path.join(testPath, BAND_14))
        band15 = read_npy_file(os.path.join(testPath, BAND_15))

        img = get_image(band11, band14, band15)
        inputs = np.float32(img / 255.)

        mask = model.predict(inputs[None, ...])[0]
        mask = np.uint8(mask * 255)

        cv2.imshow('Image', img)
        cv2.moveWindow('Image', 100, 0)

        if os.path.exists(os.path.join(TRAIN_IMG, record_id, 'human_pixel_masks.npy')):
            ground_truth = np.uint8(read_npy_file(os.path.join(TRAIN_IMG, record_id, 'human_pixel_masks.npy')) * 255)

            cv2.imshow('Ground truth', ground_truth)
            cv2.moveWindow('Ground Truth', 500, 0)

        cv2.imshow('Prediction', mask)
        cv2.moveWindow('Ground Truth', 800, 0)

        key = cv2.waitKey(0)

        if key == 27:
            break

    cv2.destroyAllWindows()
