import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, Adagrad, SGD, RMSprop
from tensorflow.keras.metrics import BinaryIoU
from model.unet import UNet
from model.psp_net import PSPNet
from model.deeplab_v3 import DeeplabV3
from imutils import paths
from config import BAND_11, BAND_14, BAND_15, DATA, DROPOUT, LR, EPOCHS, IMAGES, GROUND_TRUTH, BATCH_SIZE, TEST, SAVED_MODEL, VAL
from utils import get_image, process_input, read_npy_file, read_image_as_yuv
from losses import DiceLoss


def scheduler(epoch: int, lr: float) -> float:
    if epoch <= 10:
        return lr
    elif epoch > 10 and epoch <= 20:
        return 1e-4
    else:
        return 1e-5



if __name__ == '__main__':
    if not os.path.exists(SAVED_MODEL):
        os.mkdir(SAVED_MODEL)

    logs_dir = './logs'
    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)

    idx = os.listdir(logs_dir)
    if len(idx) == 0:
        idx = '1'
    else:
        idx = str(int(idx[-1]) + 1)

    tb_logs_dir = os.path.join(logs_dir, idx)
    if not os.path.exists(tb_logs_dir):
        os.mkdir(tb_logs_dir)

    tf.keras.saving.get_custom_objects().clear()

    AUTO = tf.data.AUTOTUNE
    # imagePaths = list(paths.list_files(os.path.join(DATA, IMAGES)))
    # maskPaths = list(paths.list_files(os.path.join(DATA, GROUND_TRUTH)))
    val_image_paths = list(paths.list_files(os.path.join(VAL, IMAGES)))
    val_mask_paths = list(paths.list_files(os.path.join(VAL, GROUND_TRUTH)))

    # dataset = tf.data.Dataset.from_tensor_slices((imagePaths, maskPaths))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_image_paths, val_mask_paths))

    # trainDS = dataset.shuffle(21000, reshuffle_each_iteration=True).map(process_input, num_parallel_calls=AUTO).batch(BATCH_SIZE).prefetch(AUTO)
    val_ds = val_dataset.shuffle(2000, reshuffle_each_iteration=True).map(process_input, num_parallel_calls=AUTO).batch(BATCH_SIZE).prefetch(AUTO)

    mirrored_strategy = tf.distribute.MirroredStrategy()

    with mirrored_strategy.scope():
        # unet = UNet(DROPOUT)
        # psp_net = PSPNet()
        deeplab = DeeplabV3()
        model = deeplab.build_model()

        optimizer = Adam(LR)
        metric = BinaryIoU()

        loss = DiceLoss(is_sigmoid=False)
        model.compile(optimizer=optimizer, loss=loss, metrics=metric)

    model.summary()

    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_logs_dir, profile_batch=5)

    model.fit(val_ds, validation_data=None, epochs=EPOCHS, callbacks=[lr_scheduler, tensorboard_callback])
    model.save(SAVED_MODEL)
