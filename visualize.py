import os
import cv2
import numpy as np
from tqdm import tqdm
from config import TRAIN_IMG
from utils import read_npy_file

if __name__ == '__main__':
    for record_id in tqdm(os.listdir(TRAIN_IMG)):
        img = np.uint8(read_npy_file(os.path.join(TRAIN_IMG, record_id, 'image.npy')) * 255)
        ground_truth = np.uint8(read_npy_file(os.path.join(TRAIN_IMG, record_id, 'human_pixel_masks.npy')) * 255)

        cv2.imshow('Image', img)
        cv2.moveWindow('Image', 100, 0)
        cv2.imshow('Ground truth', ground_truth)
        cv2.moveWindow('Ground Truth', 500, 0)

        key = cv2.waitKey(0)

        if key == 27:
            break

    cv2.destroyAllWindows()
