import os
import cv2
import random
import glob
import numpy as np
from tqdm import tqdm
from config import BAND_11, BAND_14, BAND_15, MASK_NPY, TRAIN_IMG, DATA, IMAGES, GROUND_TRUTH, IMAGE_NPY, MASK_NPY, VALIDATION_RAW, VAL
from utils import read_npy_file, get_image

if __name__ == '__main__':
    is_jpg = True
    if not os.path.exists(DATA):
        os.mkdir(DATA)

        train_images_dir = os.path.join(DATA, IMAGES)
        mask_images_dir = os.path.join(DATA, GROUND_TRUTH)
        os.mkdir(train_images_dir)
        os.mkdir(mask_images_dir)

        for record_id in tqdm(os.listdir(TRAIN_IMG)):
            if is_jpg:
                img = np.uint8(read_npy_file(os.path.join(TRAIN_IMG, record_id, IMAGE_NPY)) * 255)
                ground_truth = np.uint8(read_npy_file(os.path.join(TRAIN_IMG, record_id, MASK_NPY)) * 255)

                cv2.imwrite(os.path.join(train_images_dir, record_id + '.jpg'), img)
                cv2.imwrite(os.path.join(mask_images_dir, record_id + '.jpg'), ground_truth)

            else:
                img = read_npy_file(os.path.join(TRAIN_IMG, record_id, IMAGE_NPY))
                ground_truth = np.float32(read_npy_file(os.path.join(TRAIN_IMG, record_id, MASK_NPY)))

                np.save(os.path.join(train_images_dir, record_id + '.npy'), img)
                np.save(os.path.join(mask_images_dir, record_id + '.npy'), ground_truth)

    if not os.path.exists(VAL) and is_jpg:
        os.mkdir(VAL)

        train_images_dir = os.path.join(DATA, IMAGES)
        val_images_dir = os.path.join(VAL, IMAGES)
        mask_images_dir = os.path.join(DATA, GROUND_TRUTH)
        val_mask_images_dir = os.path.join(VAL, GROUND_TRUTH)
        os.mkdir(val_images_dir)
        os.mkdir(val_mask_images_dir)

        raw_val_path = os.listdir(VALIDATION_RAW)
        # random.shuffle(raw_val_path)
        # val_ds = raw_val_path[0:256]
        # train_ds = raw_val_path[256:]

        for record_id in tqdm(raw_val_path):
            if '.json' not in record_id:
                band_11 = read_npy_file(os.path.join(VALIDATION_RAW, record_id, BAND_11))
                band_14 = read_npy_file(os.path.join(VALIDATION_RAW, record_id, BAND_14))
                band_15 = read_npy_file(os.path.join(VALIDATION_RAW, record_id, BAND_15))
                ann = read_npy_file(os.path.join(VALIDATION_RAW, record_id, MASK_NPY))

                img = get_image(band_11, band_14, band_15)
                ground_truth = np.uint8(ann * 255)

                cv2.imwrite(os.path.join(val_images_dir, record_id + '.jpg'), img)
                cv2.imwrite(os.path.join(val_mask_images_dir, record_id + '.jpg'), ground_truth)

