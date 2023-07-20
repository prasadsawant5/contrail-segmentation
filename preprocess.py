import os
import tensorflow as tf
from tensorflow.python.ops.gen_dataset_ops import MapDataset
from tensorflow.keras.preprocessing import image_dataset_from_directory
from config import DATA, IMAGES, GROUND_TRUTH, WIDTH, HEIGHT, BATCH_SIZE


def configure_for_performance(ds: MapDataset) -> MapDataset:
    ds = ds.cache()
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def get_dataset() -> MapDataset:
    normalization = tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255)

    images_dir = os.path.join(DATA, IMAGES)
    masks_dir = os.path.join(DATA, GROUND_TRUTH)

    if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
        print('Dir does not exists')
        return

    train_images = image_dataset_from_directory(
        images_dir,
        label_mode=None,
        image_size=(HEIGHT, WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    train_ground_truth = image_dataset_from_directory(
        masks_dir,
        label_mode=None,
        image_size=(HEIGHT, WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    dataset = tf.data.Dataset.zip((train_images, train_ground_truth))
    dataset = dataset.map(lambda img, mask: (normalization(img), normalization(mask)))
    dataset = configure_for_performance(dataset)

    return dataset


