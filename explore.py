import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from config import TRAIN_RAW, BAND_META_DATA
from utils import read_npy_file

if __name__ == '__main__':

    if not os.path.exists(BAND_META_DATA):
        records = []
        bands = []
        min = []
        max = []

        for record_id in tqdm(os.listdir(TRAIN_RAW)):
            if record_id != '905857553501724345':
                for band in glob.glob(os.path.join(TRAIN_RAW, record_id, 'band_*.npy')):
                    data = read_npy_file(band)
                    band_id = (band.split('/')[-1]).split('.')[0]
                    records.append(record_id)
                    bands.append(band_id)
                    min.append(np.min(data))
                    max.append(np.max(data))

        df = pd.DataFrame({'record_id': records, 'band': bands, 'min': min, 'max': max}, index=None)
        df.to_csv(BAND_META_DATA)


    # record_id = os.listdir(TRAIN_RAW)[0]
    # directory = os.path.join(TRAIN_RAW, record_id)
    #
    # print('Record ID: {}'.format(record_id))
    #
    # with open(os.path.join(directory, 'band_08.npy'), 'rb') as f:
    #     band8 = np.load(f)
    # with open(os.path.join(directory, 'band_09.npy'), 'rb') as f:
    #     band9 = np.load(f)
    # with open(os.path.join(directory, 'band_10.npy'), 'rb') as f:
    #     band10 = np.load(f)
    # with open(os.path.join(directory, 'band_11.npy'), 'rb') as f:
    #     band11 = np.load(f)
    # with open(os.path.join(directory, 'band_12.npy'), 'rb') as f:
    #     band12 = np.load(f)
    # with open(os.path.join(directory, 'band_13.npy'), 'rb') as f:
    #     band13 = np.load(f)
    # with open(os.path.join(directory, 'band_14.npy'), 'rb') as f:
    #     band14 = np.load(f)
    # with open(os.path.join(directory, 'band_15.npy'), 'rb') as f:
    #     band15 = np.load(f)
    # with open(os.path.join(directory, 'band_16.npy'), 'rb') as f:
    #     band16 = np.load(f)
    # with open(os.path.join(directory, 'human_pixel_masks.npy'), 'rb') as f:
    #     human_pixel_mask = np.load(f)
    # with open(os.path.join(directory, 'human_individual_masks.npy'), 'rb') as f:
    #     human_individual_mask = np.load(f)
    #
    # print('Band 8: {}'.format(band8.dtype))
    # print('Band 9: {}'.format(band9.dtype))
    # print('Band 10: {}'.format(band10.dtype))
    # print('Band 11: {}'.format(band11.dtype))
    # print('Band 12: {}'.format(band12.dtype))
    # print('Band 13: {}'.format(band13.dtype))
    # print('Band 14: {}'.format(band14.dtype))
    # print('Band 15: {}'.format(band15.dtype))
    # print('Band 16: {}'.format(band16.dtype))

        