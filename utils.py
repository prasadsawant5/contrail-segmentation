import tensorflow as tf
import tensorflow_io as tfio
import numpy as np
from config import _CLOUD_TOP_TDIFF_BOUNDS, _TDIFF_BOUNDS, _T11_BOUNDS, N_TIMES_BEFORE


def normalize_range(data, bounds):
    """Maps data to the range [0, 1]."""
    return (data - bounds[0]) / (bounds[1] - bounds[0])


def read_npy_file(directory: str) -> np.array:
    """
    Parameters
    ----------
    directory : str
        The directory to read from
    """
    with open(directory, 'rb') as f:
        arr = np.load(f)
        return arr


@tf.function
def process_input(img_dir, mask_dir) -> tuple:
    """
    Parameters
    ----------
    img_dir : str
        The directory to read from
    """
    img = tf.io.read_file(img_dir)
    img = tf.image.decode_jpeg(img, 3)
    img = tfio.experimental.color.rgb_to_bgr(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.clip_by_value(img, 0.0, 1.0)

    mask = tf.io.read_file(mask_dir)
    mask = tf.image.decode_jpeg(mask, 1)
    # mask = tfio.experimental.color.rgb_to_bgr(mask)
    mask = tf.image.convert_image_dtype(mask, tf.float32)
    mask = tf.clip_by_value(mask, 0.0, 1.0)

    return (img, mask)

@tf.function
def read_image_as_yuv(img_dir, mask_dir) -> tuple:
    """
    Parameters
    ----------
    img_dir : str
        The directory to read from
    """
    img = tf.io.read_file(img_dir)
    img = tf.image.decode_jpeg(img, 3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    yuv = tf.image.rgb_to_yuv(img)
    (Y, _, _) = tf.split(yuv, 3, axis=-1)
    Y = tf.clip_by_value(Y, 0.0, 1.0)

    mask = tf.io.read_file(mask_dir)
    mask = tf.image.decode_jpeg(mask, 1)
    # mask = tfio.experimental.color.rgb_to_bgr(mask)
    mask = tf.image.convert_image_dtype(mask, tf.float32)
    mask = tf.clip_by_value(mask, 0.0, 1.0)

    return (Y, mask)


def rle_encode(x, fg_val=1):
    """
    Args:
        x:  numpy array of shape (height, width), 1 - mask, 0 - background
    Returns: run length encoding as list
    """

    dots = np.where(
        x.T.flatten() == fg_val)[0]  # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def list_to_string(x):
    """
    Converts list to a string representation
    Empty list returns '-'
    """
    if x: # non-empty list
        s = str(x).replace("[", "").replace("]", "").replace(",", "")
    else:
        s = '-'
    return s


def rle_decode(mask_rle, shape=(256, 256)):
    '''
    mask_rle: run-length as string formatted (start length)
              empty predictions need to be encoded with '-'
    shape: (height, width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''

    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    if mask_rle != '-':
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
    return img.reshape(shape, order='F')  # Needed to align to RLE direction


def get_image(band11: np.array, band14: np.array, band15: np.array):
    r = normalize_range(band15 - band14, _TDIFF_BOUNDS)
    g = normalize_range(band14 - band11, _CLOUD_TOP_TDIFF_BOUNDS)
    b = normalize_range(band14, _T11_BOUNDS)

    false_color = np.clip(np.stack([r, g, b], axis=2), 0, 1)
    return false_color[..., N_TIMES_BEFORE]
