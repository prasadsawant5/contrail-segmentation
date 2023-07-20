# https://www.kaggle.com/datasets/thejavanka/google-research-identify-contrails-preprocessing

WIDTH = 256
HEIGHT = 256
BATCH_SIZE = 32
DROPOUT = 0.3
LR = 1e-3
EPOCHS = 30
OUTPUT_CHANNELS = 1

N_TIMES_BEFORE = 4
N_TIMES_after = 3

# Used for normalization
# https://www.kaggle.com/code/inversion/visualizing-contrails
_T11_BOUNDS = (243, 303)
_CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
_TDIFF_BOUNDS = (-4, 2)

VALIDATION_RAW = 'validation'
TRAIN_RAW = 'train'
BAND_META_DATA = 'band_metadata.csv'
TRAIN_IMG = 'train_images'

BAND_11 = 'band_11.npy'
BAND_14 = 'band_14.npy'
BAND_15 = 'band_15.npy'

IMAGE_NPY = 'image.npy'
MASK_NPY = 'human_pixel_masks.npy'

VAL = 'val_data'
DATA = 'data'
IMAGES = 'images'
GROUND_TRUTH = 'ground_truth'
TEST = 'test'
SAVED_MODEL = 'saved_model'

# Record ID 905857553501724345 is incomplete
